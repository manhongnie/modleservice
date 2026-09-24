"""Run real local weights in reusable worker processes and save bounded evidence."""
from __future__ import annotations
import argparse
import asyncio
import base64
import json
from pathlib import Path
import time

from model_service.contracts import ModelConfig, ServiceError
from model_service.executor import ProcessExecutor
from model_service.tasks import validate_input


async def main(args):
    rows = json.loads(Path("examples/legacy/models.real.json").read_text())
    destination = Path(args.report)
    report = json.loads(destination.read_text()) if destination.exists() else {}
    for row in rows:
        if args.models and row["name"] not in args.models:
            continue
        cfg = ModelConfig(**row)
        payload = cfg.validation_input
        if args.audio and cfg.capabilities == ["asr"]:
            payload = {"audio_base64": base64.b64encode(Path(args.audio).read_bytes()).decode()}
        executor = ProcessExecutor(load_timeout_s=300)
        record = {"model": cfg.model_id, "backend": cfg.backend, "path": cfg.path,
                  "timestamp_unix": time.time(), "validation_input": args.audio if args.audio and cfg.capabilities == ["asr"] else "configured sample", "status": "failed"}
        source_file = Path(cfg.path) / "source.json"
        if source_file.exists():
            record["source"] = json.loads(source_file.read_text())
        try:
            validate_input(cfg, payload, cfg.capabilities[0])
            started = time.monotonic()
            record["load"] = await executor.load(cfg)
            record["load_elapsed_s"] = time.monotonic() - started
            started = time.monotonic()
            output = [chunk async for chunk in executor.stream(cfg.instance_key, payload, "real-smoke", False, asyncio.Event())]
            record["inference_elapsed_s"] = time.monotonic() - started
            record["worker"] = executor.snapshot()[cfg.instance_key]
            if cfg.capabilities == ["tts"]:
                wav = base64.b64decode(output[0].pop("audio_base64"))
                artifact = Path("var/real-tts-smoke.wav")
                artifact.parent.mkdir(exist_ok=True)
                artifact.write_bytes(wav)
                output[0]["artifact"] = str(artifact)
                output[0]["wav_bytes"] = len(wav)
            record["output"] = output
            record["status"] = "passed_real_weights_smoke"
            if args.cancel:
                event = asyncio.Event()

                async def cancelled_request():
                    return [item async for item in executor.stream(cfg.instance_key, payload, "cancel-smoke", True, event)]

                running = asyncio.create_task(cancelled_request())
                await asyncio.sleep(0.03)
                cancelled_at = time.monotonic()
                event.set()
                try:
                    await running
                    record["cancel_test"] = {"status": "already_completed_before_cancel"}
                except ServiceError as error:
                    if error.code != "cancelled":
                        raise
                    record["cancel_test"] = {"status": "cancelled_and_stopped", "ack_after_cancel_s": time.monotonic() - cancelled_at,
                                             "busy_after_ack": executor.snapshot()[cfg.instance_key]["busy"]}
        except Exception as error:
            record["status"] = "failed"
            record["error_type"] = type(error).__name__
            record["error"] = str(error)
        finally:
            await executor.close(timeout_s=300)
        report[cfg.name] = record
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*")
    parser.add_argument("--report", default="docs/models-validation.json")
    parser.add_argument("--audio", help="Optional PCM16 mono 16kHz WAV for the ASR smoke test")
    parser.add_argument("--cancel", action="store_true", help="After warm inference, verify cooperative cancellation stop acknowledgement")
    asyncio.run(main(parser.parse_args()))
