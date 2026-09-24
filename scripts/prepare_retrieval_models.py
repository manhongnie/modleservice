"""Fetch pinned upstream retrieval artifacts and optionally export CLIP towers.

No remote Python code is executed. Downloads are local copies under --root.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

MODELS = {
    'bge-m3': ('EmbeddedLLM/bge-m3-int4-ov', 'd50a3650d1f64b3621cb4685557b3c96acbc173a', 'mit'),
    'bge-reranker': ('EmbeddedLLM/bge-reranker-v2-m3-int4-ov', '2252a9a291c587141642f1ac7d10be2aa3e43b3d', 'apache-2.0'),
    'bm42': ('Qdrant/all_miniLM_L6_v2_with_attentions', '9695632d760ba609397b32f56fb5b1b870cf92a5', 'apache-2.0'),
    'chinese-clip': ('OFA-Sys/chinese-clip-vit-base-patch16', '36e679e65c2a2fead755ae21162091293ad37834', 'not_declared_in_model_card'),
    'qwen3-embedding': ('OpenVINO/Qwen3-Embedding-0.6B-int8-ov', 'dfe67308c933b7d1a5d15147902213b48c85a5a0', 'apache-2.0'),
    'qwen3-reranker': ('OpenVINO/Qwen3-Reranker-0.6B-int8-ov', 'd7dd424f08cbafe70657ca391c619820a99c1ca9', 'apache-2.0'),
    'openai-clip': ('openai/clip-vit-base-patch16', '57c216476eefef5ab752ec549e440a49ae4ae5f3', 'mit'),
}
DEFAULT_MODELS = ['bge-m3', 'bge-reranker', 'bm42', 'chinese-clip']


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def download(name, root):
    import fnmatch
    try:
        from scripts.download_models import download as fetch_file, read_url
    except ModuleNotFoundError:
        from download_models import download as fetch_file, read_url
    repo, revision, license_name = MODELS[name]
    target = root / name
    target.mkdir(parents=True, exist_ok=True)
    upstream = json.loads(read_url(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true'))
    patterns = ['*.json', '*.txt', '*.model', 'README.md', 'LICENSE*', 'openvino_model.xml', 'openvino_model.bin']
    if name == 'bm42': patterns.append('model.onnx')
    if name.endswith('clip'): patterns += ['pytorch_model.bin', 'festival.jpg']
    files = {}
    for entry in upstream['siblings']:
        filename = entry['rfilename']
        if not any(fnmatch.fnmatch(filename, pattern) for pattern in patterns): continue
        artifact = (target / filename).resolve()
        if not artifact.is_relative_to(target.resolve()): raise ValueError('Upstream artifact escaped model directory')
        artifact.parent.mkdir(parents=True, exist_ok=True)
        expected_sha256 = entry.get('lfs', {}).get('sha256')
        fetch_file(f'https://huggingface.co/{repo}/resolve/{revision}/{filename}', artifact,
                   entry.get('size'), expected_sha256)
        files[filename] = {'bytes': artifact.stat().st_size, 'sha256': digest(artifact),
                           'upstream_sha256_verified': bool(expected_sha256)}
    manifest = {'repository': repo, 'revision': revision, 'license': license_name,
                'license_reference': f'https://huggingface.co/{repo}', 'files': files,
                'base_model_license_reference': {'bge-m3':'https://huggingface.co/BAAI/bge-m3',
                    'bge-reranker':'https://huggingface.co/BAAI/bge-reranker-v2-m3'}.get(name, f'https://huggingface.co/{repo}')}
    previous_path=target / 'source-manifest.json'
    if previous_path.exists():
        previous=json.loads(previous_path.read_text())
        if previous.get('revision')==revision and 'conversion' in previous:
            manifest['conversion']=previous['conversion']
            for converted in target.glob('openvino_*'):
                if converted.name.startswith(('openvino_text.', 'openvino_image.')):
                    manifest['files'][converted.name]={'bytes':converted.stat().st_size, 'sha256':digest(converted)}
    previous_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'downloaded': name, 'directory': str(target), 'revision': revision}), flush=True)
    return target


def export_clip(path: Path, chinese: bool):
    import torch
    import openvino as ov
    from transformers import CLIPModel, ChineseCLIPModel
    torch.set_num_threads(2)
    model_class = ChineseCLIPModel if chinese else CLIPModel
    model = model_class.from_pretrained(str(path), local_files_only=True, trust_remote_code=False).eval()

    class TextTower(torch.nn.Module):
        def __init__(self, source):
            super().__init__(); self.source = source
        def forward(self, input_ids, attention_mask, token_type_ids=None):
            args = {'input_ids': input_ids, 'attention_mask': attention_mask, 'return_dict': False}
            if chinese: args['token_type_ids'] = token_type_ids
            hidden = self.source.text_model(**args)
            pooled = hidden[0][:, 0, :] if chinese else hidden[1]
            return self.source.text_projection(pooled)

    class ImageTower(torch.nn.Module):
        def __init__(self, source):
            super().__init__(); self.source = source
        def forward(self, pixel_values):
            pooled = self.source.vision_model(pixel_values=pixel_values, return_dict=False)[1]
            return self.source.visual_projection(pooled)

    ids = torch.ones((1, 16), dtype=torch.long)
    with torch.no_grad():
        text_input = {'input_ids': ids, 'attention_mask': ids}
        if chinese: text_input['token_type_ids'] = torch.zeros_like(ids)
        text_ir = ov.convert_model(TextTower(model), example_input=text_input)
        for port, name in zip(text_ir.inputs, text_input): port.get_tensor().set_names({name})
        text_ir.output(0).get_tensor().set_names({'text_embeds'})
        text_ir.reshape({port: [-1, -1] for port in text_ir.inputs})
        ov.save_model(text_ir, path / 'openvino_text.xml', compress_to_fp16=True)
        image_ir = ov.convert_model(ImageTower(model), example_input={'pixel_values': torch.zeros(1, 3, 224, 224)})
        image_ir.input(0).get_tensor().set_names({'pixel_values'})
        image_ir.output(0).get_tensor().set_names({'image_embeds'})
        image_ir.reshape({image_ir.input(0): [-1, 3, 224, 224]})
        ov.save_model(image_ir, path / 'openvino_image.xml', compress_to_fp16=True)
    # Exercise batch/sequence shapes different from tracing inputs before the
    # exported bundle can be registered. FP16 storage should preserve direction.
    import numpy as np
    comparison = {}
    sample_ids = torch.arange(16, dtype=torch.long).reshape(2, 8) + 100
    sample_mask = torch.ones_like(sample_ids)
    sample_mask[1, -2:] = 0
    samples = {'text': {'input_ids': sample_ids, 'attention_mask': sample_mask},
               'image': {'pixel_values': torch.rand(2, 3, 224, 224, generator=torch.Generator().manual_seed(17))}}
    if chinese: samples['text']['token_type_ids'] = torch.zeros_like(sample_ids)
    with torch.no_grad():
        for tower, wrapper in [('text', TextTower(model)), ('image', ImageTower(model))]:
            expected = wrapper(**samples[tower]).numpy()
            compiled = ov.Core().compile_model(path / f'openvino_{tower}.xml', 'CPU',
                       {'INFERENCE_NUM_THREADS':2, 'INFERENCE_PRECISION_HINT':'f32'})
            actual = compiled({key:value.numpy() for key,value in samples[tower].items()})[0]
            norm_expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)
            norm_actual = actual / np.linalg.norm(actual, axis=1, keepdims=True)
            cosines = np.sum(norm_expected * norm_actual, axis=1)
            if not np.isfinite(cosines).all() or float(cosines.min()) < .999:
                raise RuntimeError(f'{tower} OpenVINO export parity failed: {cosines}')
            comparison[tower] = {'minimum_cosine':float(cosines.min()), 'batch_size':2,
                                  'sequence_length':8 if tower=='text' else None}
    manifest_path = path / 'source-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['conversion'] = {'runtime': ov.__version__, 'format': 'OpenVINO IR FP16 weights',
                              'method': 'Separate projected text and vision towers, dynamic batch and text sequence',
                              'torch_version': torch.__version__, 'export_parity':comparison}
    for artifact in path.glob('openvino_*'):
        manifest['files'][artifact.name] = {'bytes': artifact.stat().st_size, 'sha256': digest(artifact)}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'exported': str(path)}), flush=True)


def verify_model(name, root):
    import base64
    import resource
    from importlib.metadata import version
    from threading import Event
    import time
    import numpy as np
    from model_service.contracts import ModelConfig
    from model_service.backends.openvino import OpenVINOBackend
    from model_service.backends.retrieval_ov import DualClipOpenVINOBackend
    from model_service.tasks.retrieval_ov import (BM42Task, DualClipTask, BoundedDenseTask,
        BoundedRerankTask, QwenEmbeddingTask, QwenRerankerTask)
    classes = {'bge-m3': BoundedDenseTask, 'bge-reranker': BoundedRerankTask, 'bm42': BM42Task,
               'chinese-clip': DualClipTask, 'openai-clip': DualClipTask,
               'qwen3-embedding': QwenEmbeddingTask, 'qwen3-reranker': QwenRerankerTask}
    rows = json.loads(Path('examples/models.retrieval-migrated.json').read_text())
    row = next(row for row in rows if Path(row['path']).name == name)
    row['path'] = str(root / name)
    config = ModelConfig(**row)
    backend = DualClipOpenVINOBackend(config) if name.endswith('clip') else OpenVINOBackend(config)
    task = classes[name](config)
    record = {'model': config.model_id, 'backend': config.backend, 'device': config.device,
              'source_manifest': str(root / name / 'source-manifest.json'), 'timestamp_unix': time.time(),
              'runtime_versions': {package:version(package) for package in ['openvino','transformers','torch','numpy']}}
    started = time.monotonic()
    record['load'] = backend.load()
    record['load_seconds'] = time.monotonic() - started
    def execute(payload):
        prepared = task.prepare(payload)
        return task.finish(backend.infer(prepared.inputs, Event()), prepared.context)
    try:
        started = time.monotonic()
        if name in {'bge-m3', 'qwen3-embedding'}:
            queries = execute({'texts':['中国的首都是哪里？'], 'input_type':'query'})['embeddings']
            docs = execute({'texts':['北京是中国的首都。', '香蕉是一种水果。']})['embeddings']
            scores = np.asarray(queries) @ np.asarray(docs).T
            assert scores[0,0] > scores[0,1], scores
            record['quality'] = {'sample': 'Chinese capital retrieval', 'cosine_scores':scores[0].tolist(),
                                 'dimensions':len(docs[0]), 'correct_first':True}
        elif name in {'bge-reranker', 'qwen3-reranker'}:
            result = execute({'query':'中国的首都是哪里？','documents':['北京是中国的首都。','香蕉是一种水果。']})
            assert result['results'][0]['index'] == 0, result
            record['quality'] = {'sample':'Chinese capital reranking', **result, 'correct_first':True}
        elif name == 'bm42':
            result = execute({'texts':['You should stay, study and sprint.', 'History can only prepare us to be surprised yet again.']})
            expected = [{1881538586:.26399775,150760872:.24662513,1932363795:.47077307},
                        {733618285:.38320042,1849833631:.25453135,1008800696:.18017513,2090661150:.30432631,1117393019:.1373556}]
            errors=[]
            for got, reference in zip(result['sparse_embeddings'], expected):
                actual=dict(zip(got['indices'],got['values']))
                assert set(actual)==set(reference), (actual,reference)
                errors.append(max(abs(actual[key]-value) for key,value in reference.items()))
            assert max(errors)<.005, errors
            query = execute({'texts':['study history'], 'input_type':'query'})
            assert all(value==1 for value in query['sparse_embeddings'][0]['values'])
            record['quality']={'sample':'Official model-card examples', 'maximum_absolute_error':max(errors),
                               'matches_official_indices':True, **result}
        else:
            photo = root / 'chinese-clip' / 'festival.jpg'
            payload={'texts':['红色的中国灯笼','一辆汽车','一只猫'] if name=='chinese-clip' else ['red Chinese lanterns','a car','a cat'],
                     'images':[base64.b64encode(photo.read_bytes()).decode()]}
            result=execute(payload)
            scores=np.asarray(result['image_embeddings']) @ np.asarray(result['text_embeddings']).T
            assert int(scores[0].argmax())==0, scores
            record['quality']={'sample':'Upstream festival.jpg vs lantern/car/cat prompts',
                               'cosine_scores':scores[0].tolist(),'dimensions':len(result['image_embeddings'][0]),'correct_first':True}
        record['inference_seconds']=time.monotonic()-started
        record['rss_peak_mb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        record['status']='passed_real_weights_small_sample'
    finally:
        record['close']=backend.close()
    print(json.dumps(record,ensure_ascii=False),flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=DEFAULT_MODELS)
    parser.add_argument('--root', type=Path, default=Path('models/retrieval'))
    parser.add_argument('--export-clip', action='store_true')
    parser.add_argument('--download-workers', type=int, default=2)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--verify-one', choices=MODELS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.verify_one:
        verify_model(args.verify_one,args.root)
        return
    if args.verify_only:
        import subprocess
        import sys
        report_path=Path('docs/retrieval-migration.json')
        report=json.loads(report_path.read_text()) if report_path.exists() else {}
        failures=[]
        for name in args.models:
            result=subprocess.run([sys.executable,__file__,'--verify-one',name,'--root',str(args.root)],capture_output=True,text=True)
            if result.returncode:
                record={'status':'failed','stderr':result.stderr[-4000:],'stdout':result.stdout[-4000:]}
                failures.append(name)
            else:
                record=json.loads(result.stdout.splitlines()[-1])
            report[name]=record
            report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({name:record},ensure_ascii=False),flush=True)
        if failures: raise SystemExit('Verification failed for: '+', '.join(failures))
        return
    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        list(pool.map(lambda name: download(name, args.root), args.models))
    if args.export_clip:
        for name in args.models:
            if name.endswith('clip'): export_clip(args.root / name, name == 'chinese-clip')


if __name__ == '__main__': main()
