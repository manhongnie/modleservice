"""Retrieval task adapters: BM42 attention weights and shared CLIP text/image space.

BM42 follows Qdrant FastEmbed's document/query algorithms. Its hashed dimensions
identify stemmed words; document values come from real MiniLM attentions. Corpus
IDF remains the responsibility of the sparse index, not this inference service.
"""
from __future__ import annotations
import io
import math
from pathlib import Path
import string
from threading import Lock

from model_service.contracts import Prepared, ServiceError
from .core import PassthroughTask, decode_base64, texts, bad
from .embeddings import DenseTask, RerankTask, artifact_dir, output_tensor, normalize


def _limits(config, *, default_batch=8, max_batch=128, max_length=8192):
    for key, default, maximum in (('max_batch_size', default_batch, max_batch), ('max_length', 512, max_length)):
        value=config.options.get(key, default)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ServiceError('invalid_config', f'{key} must be an integer within 1..{maximum}')


class BM42Task(DenseTask):
    def __init__(self, config):
        super().__init__(config)
        self._lexical = None

    def validate(self, payload, capability):
        if capability != 'sparse_embeddings':
            raise ServiceError('unsupported_capability', 'BM42 supports sparse_embeddings only')
        PassthroughTask.validate(self, payload, capability)
        _limits(self.config, default_batch=32, max_batch=32, max_length=512)
        if payload.get('input_type', 'document') not in ('query', 'document'):
            bad('input_type must be query or document')
        if len(texts(payload)) > int(self.config.options.get('max_batch_size', 32)):
            bad('BM42 input batch exceeds configured maximum')
        if self.config.options.get('language', 'english') != 'english':
            raise ServiceError('invalid_config', 'This BM42 checkpoint supports English only')
        alpha = self.config.options.get('alpha', 0.5)
        if type(alpha) not in (float, int) or not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ServiceError('invalid_config', 'BM42 alpha must be finite and within (0,1]')

    def prepare(self, payload):
        tokenizer = self.get_tokenizer()
        encoded = tokenizer(texts(payload), padding=True, truncation=True,
                            max_length=int(self.config.options.get('max_length', 512)), return_tensors='np')
        return Prepared(dict(encoded), {'input_ids': encoded['input_ids'],
                                        'attention_mask': encoded['attention_mask'],
                                        'input_type': payload.get('input_type', 'document')})

    def _lexicon(self):
        with self._init_lock:
            if self._lexical is None:
                from nltk.stem.snowball import SnowballStemmer
                stopwords_file = Path(artifact_dir(self.config)) / 'stopwords.txt'
                if not stopwords_file.is_file():
                    raise ServiceError('model_files_missing', 'BM42 bundle requires upstream stopwords.txt')
                self._lexical = (set(stopwords_file.read_text().splitlines()), SnowballStemmer('english'))
        return self._lexical

    def finish(self, outputs, context):
        import mmh3
        import numpy as np
        tokenizer = self.get_tokenizer()
        stopwords, stemmer = self._lexicon()
        attention = output_tensor(outputs, self.config.options.get('output_name', 'attention_6'))
        input_ids, mask = context['input_ids'], context['attention_mask']
        if attention.ndim != 4 or attention.shape[0] != input_ids.shape[0] or attention.shape[2:] != (input_ids.shape[1], input_ids.shape[1]) or not np.isfinite(attention).all():
            raise ServiceError('invalid_model_output', 'BM42 requires finite [batch,heads,tokens,tokens] attention_6')
        pooled = attention[:, :, 0, :].mean(axis=1) * mask
        vectors = []
        alpha = float(self.config.options.get('alpha', 0.5))
        special = set(tokenizer.all_special_tokens)
        for ids, weights in zip(input_ids, pooled):
            words = []
            for position, token in enumerate(tokenizer.convert_ids_to_tokens(ids.tolist())):
                if token in special: continue
                if token.startswith('##') and words:
                    words[-1][0] += token[2:]
                    words[-1][1].append(position)
                else:
                    words.append([token.removeprefix('##'), [position]])
            lexical = {}
            for word, positions in words:
                if word in stopwords or word in string.punctuation: continue
                stem = stemmer.stem(word)
                weight = float(sum(weights[position] for position in positions))
                if weight < 0:
                    raise ServiceError('invalid_model_output', 'BM42 attention weights must be nonnegative')
                lexical[stem] = max(lexical.get(stem, 0.0), weight)
            sparse = {}
            for stem, weight in lexical.items():
                key = abs(mmh3.hash(stem))
                sparse[key] = 1.0 if context['input_type'] == 'query' else math.log1p(weight) ** alpha
            indices = sorted(sparse)
            vectors.append({'indices': indices, 'values': [sparse[index] for index in indices]})
        return {'sparse_embeddings': vectors, 'requires_idf': True, 'language': 'english', 'algorithm': 'bm42_attention'}


class DualClipTask(PassthroughTask):
    """One instance owns both towers. Payloads may contain text, images or both."""
    def __init__(self, config):
        super().__init__(config)
        self.processor = None
        self._init_lock = Lock()

    def validate(self, payload, capability):
        if capability not in {'embeddings', 'image_embeddings'}:
            raise ServiceError('unsupported_capability', 'CLIP supports embeddings/image_embeddings')
        family=self.config.options.get('clip_family', 'chinese')
        if family not in ('chinese','openai'):
            raise ServiceError('invalid_config', 'clip_family must be chinese or openai')
        maximum=512 if family=='chinese' else 77
        length=self.config.options.get('max_length', 52)
        if type(length) is not int or not 1 <= length <= maximum:
            raise ServiceError('invalid_config', f'CLIP max_length must be within 1..{maximum}')
        batch=self.config.options.get('max_batch_size',16)
        if type(batch) is not int or not 1<=batch<=16:
            raise ServiceError('invalid_config', 'CLIP max_batch_size must be within 1..16')
        super().validate(payload, capability)
        if 'texts' in payload or 'input' in payload:
            values = texts(payload)
            if len(values) > int(self.config.options.get('max_batch_size', 16)):
                bad('CLIP input batch exceeds configured maximum')
        if 'images' in payload:
            PassthroughTask.validate(self, payload, 'image_embeddings')
            if len(payload['images']) > batch:
                bad('CLIP image batch exceeds configured maximum')

    def prepare(self, payload):
        with self._init_lock:
            if self.processor is None:
                from transformers import CLIPProcessor, ChineseCLIPProcessor
                processor_class = ChineseCLIPProcessor if self.config.options.get('clip_family', 'chinese') == 'chinese' else CLIPProcessor
                self.processor = processor_class.from_pretrained(artifact_dir(self.config), local_files_only=True)
        tensors, counts = {}, {}
        if 'texts' in payload or 'input' in payload:
            values = texts(payload)
            tensors['text'] = dict(self.processor(text=values, padding=True, truncation=True,
                                   max_length=int(self.config.options.get('max_length', 52)), return_tensors='np'))
            counts['text'] = len(values)
        if 'images' in payload:
            from PIL import Image, UnidentifiedImageError
            images = []
            for value in payload['images']:
                try:
                    with Image.open(io.BytesIO(decode_base64(value, 'image'))) as picture:
                        if picture.width * picture.height > int(self.config.options.get('max_image_pixels', 16777216)):
                            bad('Decoded image exceeds pixel limit')
                        images.append(picture.convert('RGB'))
                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    raise ServiceError('invalid_input', 'Image cannot be decoded', 422) from exc
            tensors['image'] = dict(self.processor(images=images, return_tensors='np'))
            counts['image'] = len(images)
        return Prepared(tensors, counts)

    def finish(self, outputs, context):
        result = {}
        for tower, count in context.items():
            tensor = output_tensor(outputs, f'{tower}_embeds')
            if tensor.ndim != 2 or tensor.shape[0] != count:
                raise ServiceError('invalid_model_output', 'CLIP projected embeddings have incorrect batch shape')
            result[f'{tower}_embeddings'] = normalize(tensor)
        if len(result) == 1: return {'embeddings': next(iter(result.values()))}
        return result


class BoundedDenseTask(DenseTask):
    def validate(self, payload, capability):
        super().validate(payload, capability)
        _limits(self.config)
        if len(texts(payload)) > int(self.config.options.get('max_batch_size', 8)):
            bad('Embedding batch exceeds configured maximum')


class BoundedRerankTask(RerankTask):
    def validate(self, payload, capability):
        super().validate(payload, capability)
        _limits(self.config)
        if len(payload['documents']) > int(self.config.options.get('max_batch_size', 8)):
            bad('Reranking batch exceeds configured maximum')



class QwenEmbeddingTask(BoundedDenseTask):
    def validate(self, payload, capability):
        super().validate(payload, capability)
        if payload.get('input_type', 'document') not in ('query', 'document'):
            bad('input_type must be query or document')

    def prepare(self, payload):
        tokenizer = self.get_tokenizer()
        values = texts(payload)
        if payload.get('input_type', 'document') == 'query':
            instruction = self.config.options.get('instruction', 'Given a web search query, retrieve relevant passages that answer the query')
            values = [f'Instruct: {instruction}\nQuery:{text}' for text in values]
        encoded = tokenizer(values, padding=True, truncation=True,
                            max_length=int(self.config.options.get('max_length', 512)), return_tensors='np')
        return Prepared(dict(encoded), {'attention_mask': encoded['attention_mask'], 'count': len(values)})

    def finish(self, outputs, context):
        import numpy as np
        tensor = output_tensor(outputs, 'last_hidden_state')
        mask = context['attention_mask']
        if tensor.ndim != 3 or tensor.shape[:2] != mask.shape:
            raise ServiceError('invalid_model_output', 'Qwen embedding output must align with input tokens')
        # Supports either left or right padding without pooling padding positions.
        last_positions = np.where(mask, np.arange(mask.shape[1])[None, :], -1).max(axis=1)
        if np.any(last_positions < 0):
            raise ServiceError('invalid_model_output', 'Qwen embedding input contains no active tokens')
        return {'embeddings': normalize(tensor[np.arange(tensor.shape[0]), last_positions]), 'pooling': 'last_token_l2'}


class QwenRerankerTask(BoundedRerankTask):
    def prepare(self, payload):
        import numpy as np
        tokenizer = self.get_tokenizer()
        # Padding configuration is set once before serving this concurrency=1 task.
        tokenizer.padding_side = 'left'
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        start = tokenizer.encode(prefix, add_special_tokens=False)
        end = tokenizer.encode(suffix, add_special_tokens=False)
        max_length = int(self.config.options.get('max_length', 512))
        if max_length <= len(start) + len(end):
            raise ServiceError('invalid_config', 'Qwen reranker max_length cannot contain its prompt template')
        instruction = self.config.options.get('instruction', 'Given a web search query, retrieve relevant passages that answer the query')
        pairs = [f'<Instruct>: {instruction}\n<Query>: {payload["query"]}\n<Document>: {doc}' for doc in payload['documents']]
        raw = tokenizer(pairs, add_special_tokens=False, padding=False, truncation=True,
                        max_length=max_length-len(start)-len(end), return_attention_mask=False)
        raw['input_ids'] = [start + ids + end for ids in raw['input_ids']]
        encoded = dict(tokenizer.pad(raw, padding=True, return_tensors='np'))
        encoded['position_ids'] = np.maximum(np.cumsum(encoded['attention_mask'], axis=1) - 1, 0)
        no_id, yes_id = tokenizer.convert_tokens_to_ids('no'), tokenizer.convert_tokens_to_ids('yes')
        return Prepared(encoded, {'count': len(pairs), 'no_id': no_id, 'yes_id': yes_id})

    def finish(self, outputs, context):
        import numpy as np
        logits = output_tensor(outputs, 'logits')
        if logits.ndim != 3 or logits.shape[0] != context['count'] or logits.shape[-1] <= max(context['yes_id'],context['no_id']):
            raise ServiceError('invalid_model_output', 'Qwen reranker must produce vocabulary logits for each document')
        pair = logits[:, -1, [context['no_id'], context['yes_id']]]
        if not np.isfinite(pair).all():
            raise ServiceError('invalid_model_output', 'Qwen reranker returned non-finite logits')
        scores = np.exp(pair - pair.max(axis=1, keepdims=True))
        scores /= scores.sum(axis=1, keepdims=True)
        return {'results': sorted([{'index': i, 'score':float(score)} for i, score in enumerate(scores[:,1])], key=lambda row: row['score'], reverse=True)}
