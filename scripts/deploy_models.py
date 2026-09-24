"""Offline controller: validate model lists, retire legacy registrations, persist defaults.

Stop the HTTP controller first. This uses the same management/coordinator/worker
contracts as the API and acquires its exclusive database lock. Weights stay intact.
"""
import argparse
import asyncio
import json
from pathlib import Path
import secrets
import time

from model_service.contracts import ModelConfig, ServiceError, Settings
from model_service.service import Service

LEGACY_NAMES = {'qwen3.5-0.8b', 'whisper-tiny', 'vits-aishell3', 'vits-zh-aishell3', 'vits-icefall-zh-aishell3'}


async def retire_legacy(service):
    retired = []
    for row in service.registry.all():
        config = row['config']
        if config.name not in LEGACY_NAMES:
            continue
        refs = service.registry.references(config.model_id)
        if refs['dependencies']:
            raise ServiceError('model_referenced', 'Reassign legacy business dependencies before migration', 409)
        for alias in refs['aliases']:
            await service.delete_alias(alias)
        retired.append(await service.operate(config.model_id, 'remove'))
    return retired


async def run(args):
    values = json.loads(Path(args.settings).read_text())
    values.update(business_keys=[secrets.token_hex(24)], admin_keys=[secrets.token_hex(24)])
    if args.database:
        values['database'] = args.database
    service = Service(Settings(**values))
    await service.start()
    report_path = Path(args.report)
    report = json.loads(report_path.read_text()) if report_path.exists() else {'models': {}}
    report['timestamp_unix'] = time.time()
    try:
        if args.retire_legacy:
            report['retired_legacy'] = await retire_legacy(service)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        default_targets = {}
        for file in args.files:
            for raw in json.loads(Path(file).read_text()):
                for capability in raw['capabilities']:
                    default_targets.setdefault(capability, raw['name'] + '@' + raw['version'])
        for file in args.files:
            for raw in json.loads(Path(file).read_text()):
                config = ModelConfig.model_validate(raw)
                if args.model and config.name not in args.model:
                    continue
                started = time.monotonic()
                print('Validating', config.model_id, flush=True)
                try:
                    try:
                        existing = service.registry.get(config.model_id)
                    except ServiceError as error:
                        if error.code != 'model_not_found':
                            raise
                        await service.add(config)
                    else:
                        if existing['config'] != config:
                            raise ValueError(f'{config.model_id}: configuration changed; use a new unique version')
                        await service.validate(config.model_id)
                    # Real registration smoke went through scheduler, process and plugins.
                    worker = (await service.status())['workers'].get(config.instance_key)
                    if args.defaults:
                        for capability in config.capabilities:
                            if default_targets[capability] == config.model_id:
                                await service.set_alias('default:' + capability, config.model_id)
                    release = await service.operate(config.model_id, 'unload')
                    report['models'][config.model_id] = {
                        'status': 'passed_real_registration', 'backend': config.backend,
                        'elapsed_s': time.monotonic() - started, 'worker': worker,
                        'release': release['release'], 'files_preserved': Path(config.path).exists(),
                        'scheduler_after_unload': service.scheduler.snapshot()}
                    print('Enabled and unloaded', config.model_id, flush=True)
                except Exception as error:
                    report['models'][config.model_id] = {
                        'status': 'failed', 'error': str(error), 'code': getattr(error, 'code', type(error).__name__)}
                    raise
                finally:
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        report['aliases'] = service.registry.aliases()
        report['registered_models'] = [row['model_id'] for row in service.registry.all()]
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    finally:
        await service.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='*')
    parser.add_argument('--settings', default='examples/service.multimodal.json')
    parser.add_argument('--database')
    parser.add_argument('--report', default='docs/deployment-validation.json')
    parser.add_argument('--model', action='append')
    parser.add_argument('--retire-legacy', action='store_true')
    parser.add_argument('--defaults', action='store_true', help='First model for each capability becomes its default')
    asyncio.run(run(parser.parse_args()))
