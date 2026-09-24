"""Bounded OpenVINO GenAI chat/vision inputs; optional runtimes stay in workers."""
import base64
import io
import json
import math
from pathlib import Path
from .core import validate_payload, decode_base64
from model_service.contracts import Prepared, ServiceError


class GenAITask:
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.vision_metadata = None

    def validate(self, payload, capability):
        allowed = {'messages', 'max_new_tokens'} if self.config.task == 'ov_chat' else {'prompt', 'images', 'max_new_tokens'}
        expected = 'chat' if self.config.task == 'ov_chat' else 'vision_chat'
        if capability != expected:
            raise ServiceError('unsupported_capability', f'This plugin supports {expected}', 422)
        if set(payload) - allowed:
            raise ServiceError('invalid_input', 'Unknown generation input fields', 422)
        if expected == 'chat':
            validate_payload(self.config, payload, capability)
            chars = sum(len(m['content']) for m in payload['messages'])
        else:
            prompt = payload.get('prompt')
            if not isinstance(prompt, str) or not prompt.strip():
                raise ServiceError('invalid_input', 'prompt must be nonempty text', 422)
            images = payload.get('images')
            if not isinstance(images, list) or not 1 <= len(images) <= int(self.config.options.get('max_images', 1)):
                raise ServiceError('invalid_input', 'images exceeds the configured image count', 422)
            for value in images:
                decode_base64(value, 'image')
            chars = len(prompt)
        if chars > int(self.config.options.get('max_text_chars', 4096)):
            raise ServiceError('input_too_large', 'Generation text exceeds configured limit', 413)
        tokens = payload.get('max_new_tokens', self.config.options.get('max_new_tokens', 128))
        if type(tokens) is not int or not 1 <= tokens <= int(self.config.options.get('max_new_tokens', 128)):
            raise ServiceError('invalid_input', 'max_new_tokens exceeds the configured limit', 422)

    def prepare(self, payload):
        import openvino_genai as genai
        if self.tokenizer is None:
            self.tokenizer = genai.Tokenizer(self.config.path)
        messages = payload.get('messages')
        if messages is None:
            # Keep image tags inside the user turn, before applying the chat template.
            images_prefix = ''.join(f'<ov_genai_image_{i}>\n' for i in range(len(payload['images'])))
            messages = [{'role': 'user', 'content': images_prefix + payload['prompt']}]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                     extra_context={'enable_thinking': False})
        encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_tokens = encoded.input_ids.shape[-1]
        if prompt_tokens > int(self.config.options.get('max_input_tokens', 1024)):
            raise ServiceError('input_too_large', 'Prompt exceeds configured token limit', 413)
        images = []
        visual_tokens = 0
        if 'images' in payload:
            import numpy as np
            from PIL import Image, UnidentifiedImageError
            for value in payload['images']:
                try:
                    with Image.open(io.BytesIO(base64.b64decode(value, validate=True))) as source:
                        if source.width * source.height > int(self.config.options.get('max_image_pixels', 4194304)):
                            raise ServiceError('input_too_large', 'Decoded image exceeds pixel limit', 413)
                        source.load()
                        img = source.convert('RGB')
                        edge = int(self.config.options.get('image_edge', 448))
                        img.thumbnail((edge, edge))
                        visual_tokens += self._image_token_bound(img.height, img.width)
                        if prompt_tokens + visual_tokens > int(self.config.options.get('max_input_tokens', 1024)):
                            raise ServiceError('input_too_large', 'Text and visual token budget exceeds configured limit', 413)
                        images.append(np.asarray(img, dtype=np.uint8)[None].copy())
                except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
                    raise ServiceError('invalid_input', 'Invalid encoded image', 422) from error
        return Prepared({'prompt': prompt, 'images': images,
                         'max_new_tokens': payload.get('max_new_tokens', int(self.config.options.get('max_new_tokens', 128)))},
                        {'prompt_tokens': prompt_tokens, 'visual_token_bound': visual_tokens,
                         'total_input_token_bound': prompt_tokens + visual_tokens})

    def _image_token_bound(self, height, width):
        if self.vision_metadata is None:
            root = Path(self.config.path).resolve()
            path = root / 'preprocessor_config.json'
            if not path.resolve().is_relative_to(root):
                raise ServiceError('path_forbidden', 'Visual processor metadata must stay in the model directory', 403)
            try:
                metadata = json.loads(path.read_text())
            except (OSError, ValueError) as error:
                raise ServiceError('invalid_model_metadata', 'VLM requires local preprocessor_config.json for token accounting', 422) from error
            self.vision_metadata = metadata
        return image_token_bound(height, width, self.vision_metadata)

    def finish(self, outputs, context):
        if not isinstance(outputs.get('text'), str):
            raise ServiceError('invalid_model_output', 'Generation returned no text', 502)
        return outputs

    def finish_chunk(self, outputs, context):
        if not isinstance(outputs.get('delta'), str):
            raise ServiceError('invalid_model_output', 'Generation chunk has no text delta', 502)
        return outputs


def image_token_bound(height, width, metadata):
    """Conservative bound for Qwen smart_resize followed by spatial merging.

    The ceil grid bounds native rounding; the second grid covers the minimum
    pixel upscaling that can enlarge even tiny or narrow supplied images.
    """
    if not isinstance(metadata, dict) or 'Qwen' not in str(metadata.get('image_processor_type', '')):
        raise ServiceError('unsupported_format', 'Visual token accounting currently requires Qwen processor metadata', 422)
    patch, merge = metadata.get('patch_size'), metadata.get('merge_size')
    size = metadata.get('size', {})
    if not isinstance(size, dict):
        raise ServiceError('invalid_model_metadata', 'Invalid VLM processor size', 422)
    minimum = metadata.get('min_pixels', size.get('shortest_edge'))
    maximum = metadata.get('max_pixels', size.get('longest_edge'))
    if any(type(value) is not int or value <= 0 for value in (patch, merge, minimum, maximum)) or minimum > maximum:
        raise ServiceError('invalid_model_metadata', 'VLM processor must declare positive patch, merge and pixel bounds', 422)
    if height <= 0 or width <= 0 or max(height,width) / min(height,width) > 200:
        raise ServiceError('invalid_input', 'Qwen images require positive dimensions and aspect ratio <= 200', 422)
    factor = patch * merge
    native_grid = math.ceil(height/factor) * math.ceil(width/factor)
    scale = math.sqrt(max(1.0, minimum/(height*width)))
    upscaled_grid = math.ceil(height*scale/factor) * math.ceil(width*scale/factor)
    # The formatted text already includes the generic marker. Keeping it and
    # adding native vision start/end tokens deliberately overestimates length.
    return max(native_grid, upscaled_grid) + 2
