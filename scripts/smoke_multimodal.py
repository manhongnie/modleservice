"""Real TCP smoke of the current registry; run with the production server stopped."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time

import httpx
from smoke_http_models import verify_chat_stream


def summarize_output(output, model):
    output = dict(output)
    for field, extension in [('audio_base64', 'wav'), ('image_base64', 'png'), ('video_base64', 'mp4')]:
        if field in output:
            data = base64.b64decode(output.pop(field), validate=True)
            path = Path('var/http-artifacts') / (model + '.' + extension)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            output[field] = {'artifact': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    for field in ('embeddings', 'text_embeddings', 'image_embeddings', 'embedding'):
        if field in output:
            values = output.pop(field)
            if isinstance(values[0], list):
                output[field] = {'count': len(values), 'dimension': len(values[0]), 'first_values': values[0][:4]}
            else:
                output[field] = {'dimension': len(values), 'first_values': values[:4]}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append')
    args = parser.parse_args()
    report_path = Path('docs/http-multimodal-validation.json')
    report = json.loads(report_path.read_text()) if report_path.exists() else {'results': {}}
    report.update(timestamp_unix=time.time(), transport='real TCP HTTP / production CLI', allow_mock=False)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, BUSINESS_API_KEY=secrets.token_hex(24), ADMIN_API_KEY=secrets.token_hex(24))
    business = {'Authorization': 'Bearer ' + env['BUSINESS_API_KEY']}
    admin = {'Authorization': 'Bearer ' + env['ADMIN_API_KEY']}
    log_path = Path('var/http-multimodal.log')
    with log_path.open('a') as log:
        process = subprocess.Popen([sys.executable, '-m', 'model_service', '--config', 'examples/service.multimodal.json',
                                    '--port', str(port)], env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=1000) as client:
                deadline = time.monotonic() + 20
                while True:
                    try:
                        if client.get('/health', timeout=.5).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError('CLI failed to start; inspect ' + str(log_path))
                    time.sleep(.1)
                assert client.get('/admin/models', headers=business).status_code == 401
                assert client.post('/v1/embeddings', headers=admin, json={'input': {}}).status_code == 401
                before = client.get('/admin/status', headers=admin).json()
                assert not before['workers'], 'Restart must restore configuration, not fake loaded processes'
                report['restart'] = {'restored_models': len(before['models']), 'workers_initially_empty': True,
                                     'aliases': before['aliases']}
                report['registered_model_ids'] = [row['model_id'] for row in before['models']]
                for row in before['models']:
                    config = row['config']
                    if args.model and config['name'] not in args.model:
                        continue
                    assert row['enabled'] and row['validated']
                    started = time.monotonic()
                    result_list = []
                    for capability in config['capabilities']:
                        response = client.post('/v1/' + capability, headers=business,
                            json={'model': row['model_id'], 'input': config['validation_input']})
                        response.raise_for_status()
                        body = response.json()
                        assert body['mock'] is False and body['done'] is True and body['model'] == row['model_id']
                        result_list.append({'capability': capability, 'output': summarize_output(body['output'], config['name'])})
                    if config['capabilities'] == ['chat']:
                        report['chat_streaming'] = verify_chat_stream(client, business, admin)
                    state = client.get('/admin/status', headers=admin).json()
                    workers = [value for value in state['workers'].values() if value['alive']]
                    assert len(workers) == 1 and workers[0]['load_count'] == 1
                    response = client.post('/admin/models/' + row['model_id'] + '/unload', headers=admin)
                    response.raise_for_status()
                    report['results'][row['model_id']] = {'status': 'passed_real_http', 'outputs': result_list,
                        'tested_at_unix': time.time(), 'elapsed_s': time.monotonic() - started,
                        'worker': workers[0], 'unload': response.json()['release']}
                    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
                    print(row['model_id'], 'passed_real_http', flush=True)
                after = client.get('/admin/status', headers=admin).json()
                assert after['scheduler']['active_count'] == after['scheduler']['reserved_mb'] == after['scheduler']['gpu_reserved_mb'] == 0
                report['scheduler_after_unload'] = after['scheduler']
                ready = client.get('/ready')
                ready.raise_for_status()
                metrics = client.get('/admin/metrics', headers=admin)
                metrics.raise_for_status()
                report['metrics'] = metrics.text
                report['separate_authentication'] = True
        finally:
            process.terminate()
            process.wait(timeout=120)
    assert process.returncode in (0, -signal.SIGTERM), process.returncode
    report['shutdown'] = {'returncode': process.returncode, 'graceful': True}
    report['status'] = 'passed_selected_real_models'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
