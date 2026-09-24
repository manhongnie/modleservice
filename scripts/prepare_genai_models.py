"""Download pinned OpenVINO bundles from the source project's model inventory."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

try:
    from scripts.download_models import download, read_url
except ModuleNotFoundError:
    from download_models import download, read_url

MODELS = {
    'chat': ('circulus/Qwen3.5-4B-ov-awq', '0baf3dbd8812260ab443653a8996a0801e22d42a', 'qwen3.5-4b-ov-awq'),
    'vision': ('OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov', '48494b1aceaf169fc276b912baa876aa0e43e535', 'qwen2.5-vl-7b-int4-ov'),
}
MODELSCOPE_VISION_REVISION = '4f233eca467d32069adad47ed36207d821778fde'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', nargs='*', choices=list(MODELS), default=list(MODELS))
    parser.add_argument('--metadata-only', action='store_true', help='Fetch pinned manifests and processor metadata without weights')
    parser.add_argument('--file-workers', type=int, choices=[1,2], default=1)
    parser.add_argument('--transport-attempts', type=int, choices=range(1,6), default=3,
                        help='Bounded whole-file transport attempts; resumes SHA-protected partial ranges')
    parser.add_argument('--vision-transport', choices=['huggingface', 'modelscope'], default='huggingface',
                        help='Optional official ModelScope transport, only for vision artifacts with identical upstream SHA256')
    args = parser.parse_args()
    for name in args.models:
        repo, revision, folder = MODELS[name]
        root = Path('models/imported') / folder
        root.mkdir(parents=True, exist_ok=True)
        cache = root / 'source-manifest.json'
        if cache.exists():
            manifest = json.loads(cache.read_text())
        else:
            manifest = json.loads(read_url(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true'))
            cache.write_text(json.dumps(manifest, indent=2) + '\n')
        if manifest.get('sha') != revision or str(manifest.get('id', '')).lower() != repo.lower():
            raise ValueError('Cached upstream manifest does not match the pinned model revision')
        alternate_files = {}
        if name == 'vision' and args.vision_transport == 'modelscope':
            alternate_cache = root / 'modelscope-manifest.json'
            if alternate_cache.exists():
                alternate = json.loads(alternate_cache.read_text())
            else:
                alternate = json.loads(read_url(f'https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision={MODELSCOPE_VISION_REVISION}&Recursive=true'))
                alternate_cache.write_text(json.dumps(alternate, indent=2) + '\n')
            if alternate.get('Code') != 200:
                raise ValueError('Official ModelScope manifest request failed')
            alternate_files = {entry['Path']: entry for entry in alternate['Data']['Files'] if entry.get('Type') == 'blob'}
        def fetch(entry):
            filename = entry['rfilename']
            if args.metadata_only and filename not in {'config.json', 'processor_config.json', 'preprocessor_config.json', 'tokenizer_config.json', 'generation_config.json'}:
                return None
            if not filename.endswith(('.json', '.xml', '.bin', '.txt', '.jinja', '.md')):
                return None
            target = (root / filename).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError('Untrusted artifact filename')
            target.parent.mkdir(parents=True, exist_ok=True)
            print('Downloading', name, filename, entry.get('size'), flush=True)
            expected = entry.get('lfs', {}).get('sha256')
            url = f'https://huggingface.co/{repo}/resolve/{revision}/{filename}'
            transport, transport_revision = 'huggingface.co', revision
            alternate_entry = alternate_files.get(filename, {})
            if expected and alternate_entry.get('Sha256') == expected and alternate_entry.get('Size') == entry.get('size'):
                url = f'https://modelscope.cn/models/{repo}/resolve/{MODELSCOPE_VISION_REVISION}/{filename}'
                transport, transport_revision = 'modelscope.cn', MODELSCOPE_VISION_REVISION
            cached = target.exists()
            resumed_partial = target.with_suffix(target.suffix + '.partial').exists()
            for attempt in range(args.transport_attempts):
                try:
                    download(url, target, entry.get('size'), expected)
                    break
                except OSError as error:
                    # A checksum/size failure needs inspection, not a blind retry.
                    if 'Artifact transport failed:' not in str(error) or attempt + 1 == args.transport_attempts:
                        raise
                    print(f'Resuming {name} {filename} after transport failure ({attempt + 1}/{args.transport_attempts})', flush=True)
                    time.sleep(1)
            with target.open('rb') as f:
                digest = hashlib.file_digest(f, 'sha256').hexdigest()
            return {'file': filename, 'size': target.stat().st_size, 'sha256': digest,
                    'transport': 'verified_local_cache' if cached else transport, 'transport_revision': transport_revision,
                    'upstream_sha256_verified': bool(expected), 'resumed_partial': resumed_partial}
        with ThreadPoolExecutor(max_workers=args.file_workers) as pool:
            files = [entry for entry in pool.map(fetch, manifest['siblings']) if entry is not None]
        if args.metadata_only:
            print('Metadata prepared', name, flush=True)
            continue
        provenance = {'repository': repo, 'revision': revision,
            'source_inventory': 'llm-server/src/toml/models.toml',
            'license_metadata': manifest.get('cardData', {}).get('license'), 'files': files}
        if (root / 'modelscope-manifest.json').exists():
            provenance['alternative_transport'] = {'repository': f'https://modelscope.cn/models/{repo}',
                'revision': MODELSCOPE_VISION_REVISION, 'manifest': 'modelscope-manifest.json',
                'note': 'Resumed ranges may include both origins; final large-file SHA256 must match the pinned Hugging Face LFS manifest.'}
        (root / 'source.json').write_text(json.dumps(provenance, indent=2) + '\n')
        print('Verified', name, sum(f['size'] for f in files), flush=True)


if __name__ == '__main__':
    main()
