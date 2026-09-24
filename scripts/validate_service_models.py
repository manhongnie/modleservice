"""Verify real models through registration, aliases, scheduling and worker lifecycle.

The selected database is modified intentionally: models are validated, enabled,
assigned default aliases and finally unloaded. Weights are never deleted.
"""
import argparse
import asyncio
import base64
import json
from pathlib import Path
import secrets
import time
import uuid

from model_service.contracts import ModelConfig, ServiceError, Settings
from model_service.service import Service


async def run(args):
    values = json.loads(Path("examples/service.production.json").read_text())
    values.update(database=args.database, business_keys=[secrets.token_hex(24)], admin_keys=[secrets.token_hex(24)])
    service = Service(Settings(**values))
    await service.start()
    destination = Path("docs/real-service-validation.json")
    report = json.loads(destination.read_text()) if destination.exists() else {}
    try:
        configs = json.loads(Path("examples/legacy/models.real.json").read_text())
        for raw in configs:
            if args.models and raw["name"] not in args.models:
                continue
            cfg = ModelConfig(**raw)
            started = time.monotonic()
            try:
                service.registry.get(cfg.model_id)
            except ServiceError as exc:
                if exc.code != "model_not_found":
                    raise
                await service.add(cfg)
            else:
                await service.validate(cfg.model_id)
            for capability in cfg.capabilities:
                await service.set_alias("default:" + capability, cfg.model_id)
            outputs = [item async for item in service.infer(cfg.capabilities[0], None, cfg.validation_input,
                                                           asyncio.Event(), uuid.uuid4().hex)]
            worker = (await service.status())["workers"][cfg.instance_key]
            if cfg.capabilities == ["tts"]:
                wav = base64.b64decode(outputs[0]["output"].pop("audio_base64"))
                artifact = Path("var/service-tts-smoke.wav")
                artifact.parent.mkdir(exist_ok=True)
                artifact.write_bytes(wav)
                outputs[0]["output"]["artifact"] = str(artifact)
            release = await service.operate(cfg.model_id, "unload")
            state = (await service.status())["scheduler"]
            record = {"model": cfg.model_id, "status": "passed_real_service_pipeline", "timestamp_unix": time.time(),
                      "deployment_mode": "production", "allow_mock": False,
                      "elapsed_s": time.monotonic() - started, "output": outputs, "worker": worker,
                      "release": release["release"], "resources_after_unload": state,
                      "assertions": {"real_output": outputs[0]["mock"] is False,
                                     "reused_after_validation": worker["load_count"] == 1,
                                     "no_active_request": state["active_count"] == 0}}
            assert all(record["assertions"].values()), record
            if args.remove_after_validation:
                try:
                    await service.operate(cfg.model_id, "remove")
                except ServiceError as exc:
                    assert exc.code == "model_referenced"
                else:
                    raise AssertionError("Removal should reject a model with default aliases")
                for capability in cfg.capabilities:
                    await service.delete_alias("default:" + capability)
                removed = await service.operate(cfg.model_id, "remove")
                assert removed["files_deleted"] is False and Path(cfg.path).exists()
                try:
                    service.registry.get(cfg.model_id)
                except ServiceError as exc:
                    assert exc.code == "model_not_found"
                else:
                    raise AssertionError("Model registration was not removed")
                record["removal"] = {"references_checked": True, "registration_removed": True,
                                     "files_preserved": True, "result": removed}
            report[cfg.name] = record
            destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(cfg.model_id, record["status"], flush=True)
    finally:
        await service.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*")
    parser.add_argument("--database", default="var/validation.sqlite3")
    parser.add_argument("--remove-after-validation", action="store_true",
                        help="Remove only the tested registrations and default aliases after validation; keep weight files")
    asyncio.run(run(parser.parse_args()))
