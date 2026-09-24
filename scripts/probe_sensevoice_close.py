#!/usr/bin/env python3
"""Compare direct backend.close() with process exit using three real ASR cycles.

Run from any directory with the repository's Python environment. This diagnostic
does not open a registry, change runtime providers, or call malloc_trim. A single
spawned process imports the runtime, prepares one fixed 30-second input, and then
loads, infers, and closes the same backend three times.
"""
from __future__ import annotations

import argparse
import base64
import ctypes
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import multiprocessing as mp
from pathlib import Path
import platform
import sys
from threading import Event
import time
import wave

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def memory(process):
    value = process.memory_full_info()
    return {
        "rss_bytes": value.rss,
        "uss_bytes": value.uss,
        "rss_mib": value.rss / 1048576,
        "uss_mib": value.uss / 1048576,
    }


def bundled_ort_version(sherpa):
    library = Path(sherpa.__file__).parent / "lib" / "libonnxruntime.so"
    ort = ctypes.CDLL(str(library))

    class ApiBase(ctypes.Structure):
        _fields_ = [
            ("get_api", ctypes.c_void_p),
            ("get_version_string", ctypes.CFUNCTYPE(ctypes.c_char_p)),
        ]

    ort.OrtGetApiBase.restype = ctypes.POINTER(ApiBase)
    return {
        "version": ort.OrtGetApiBase().contents.get_version_string().decode(),
        "library": str(library),
        "method": "Bundled libonnxruntime.so OrtGetApiBase()->GetVersionString()",
    }


def probe_child(connection, raw_config):
    backend = None
    try:
        import numpy as np
        import sherpa_onnx

        from model_service.backends.sherpa_models import SherpaONNXBackend
        from model_service.contracts import ModelConfig

        config = ModelConfig(**raw_config)
        raw = base64.b64decode(config.validation_input["audio_base64"])
        with wave.open(io.BytesIO(raw), "rb") as audio:
            if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
                raise ValueError("Expected a mono PCM16 reference")
            rate = audio.getframerate()
            pcm = audio.readframes(audio.getnframes())
        needed = rate * 30 * 2
        pcm = (pcm * math.ceil(needed / len(pcm)))[:needed]
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        inputs = {"sample_rate": rate, "samples": samples}
        ort_info = bundled_ort_version(sherpa_onnx)
        connection.send({
            "kind": "runtime",
            "runtime": {
                "sherpa_onnx_version": sherpa_onnx.__version__,
                "onnxruntime": ort_info,
                "provider": "cpu",
                "threads": config.options.get("threads", 2),
                "torch_imported": "torch" in sys.modules,
                "funasr_imported": "funasr" in sys.modules,
            },
            "input": {
                "duration_seconds": 30,
                "sample_rate": rate,
                "samples": len(samples),
                "reference_wav_sha256": hashlib.sha256(raw).hexdigest(),
                "tiled_pcm16_sha256": hashlib.sha256(pcm).hexdigest(),
                "construction": "Repeat and truncate the registered synthetic speech reference to 30 seconds",
            },
        })
        del raw, pcm
        process = psutil.Process()
        backend = SherpaONNXBackend(config)
        cancel = Event()
        for cycle in range(1, 4):
            result = {"cycle": cycle, "pid": process.pid, "before_load": memory(process)}
            began = time.monotonic()
            result["load_metadata"] = backend.load()
            result["load_seconds"] = time.monotonic() - began
            result["after_load"] = memory(process)
            began = time.monotonic()
            output = backend.infer(inputs, cancel)
            result["inference_seconds"] = time.monotonic() - began
            result["output_text_chars"] = len(output["text"])
            result["output_text_sha256"] = hashlib.sha256(output["text"].encode()).hexdigest()
            result["after_inference"] = memory(process)
            del output
            began = time.monotonic()
            result["close_result"] = backend.close()
            result["close_seconds"] = time.monotonic() - began
            result["model_reference_cleared"] = backend.model is None
            result["immediately_after_close"] = memory(process)
            time.sleep(1)
            result["one_second_after_close"] = memory(process)
            result["rss_remaining_above_cycle_baseline_mib"] = (
                result["one_second_after_close"]["rss_mib"] - result["before_load"]["rss_mib"]
            )
            result["uss_remaining_above_cycle_baseline_mib"] = (
                result["one_second_after_close"]["uss_mib"] - result["before_load"]["uss_mib"]
            )
            connection.send({"kind": "cycle", "cycle": result})
    except BaseException as exc:
        connection.send({"kind": "error", "type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if backend is not None and backend.model is not None:
            backend.close()
        connection.close()


def stop_probe(child):
    """Bound cleanup of this diagnostic's own process, including interruption."""
    child.terminate()
    child.join(5)
    if child.is_alive():
        child.kill()
        child.join(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "docs/sensevoice-close-validation.json")
    args = parser.parse_args()
    rows = json.loads((ROOT / "examples/models.sherpa.json").read_text())
    raw_config = next(row for row in rows if row["task"] == "sherpa_sensevoice")
    raw_config["path"] = str((ROOT / raw_config["path"]).resolve())
    controller = psutil.Process()
    report = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "scope": "Three real load/infer/close cycles in one spawned process; no production database or HTTP service",
        "configuration": {key: value for key, value in raw_config.items() if key != "validation_input"},
        "measurement_notes": [
            "RSS and USS are point-in-time psutil readings in MiB, not inference peak measurements.",
            "Each cycle baseline includes imported native libraries and the same retained 30-second float32 waveform.",
            "backend.close() clears its model reference and invokes gc.collect(); garbage collection does not guarantee RSS returns to the OS.",
            "No per-inference collection, malloc_trim, or provider/arena setting changes are used.",
            "Residual RSS after close may include allocator caches or fragmentation; it alone does not prove a leak.",
            "Process exit confirmation is separate from direct-close readings. Three cycles do not establish long-run leak absence.",
        ],
        "controller_before_spawn": memory(controller),
        "cycles": [],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    context = mp.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    child = context.Process(target=probe_child, args=(child_connection, raw_config), name="sensevoice-direct-close-probe")
    observed_child = None
    started = False
    forced_termination = False
    passed = False
    try:
        child.start()
        started = True
        child_connection.close()
        report["child_pid"] = child.pid
        # A failed import can exit before psutil captures the PID identity. The
        # multiprocessing handle still supplies exit status and reaps that child.
        try:
            observed_child = psutil.Process(child.pid)
        except psutil.NoSuchProcess:
            pass
        save()
        deadline = time.monotonic() + 180
        pipe_closed = False
        while child.is_alive() or (not pipe_closed and parent_connection.poll()):
            if not pipe_closed and parent_connection.poll(.2):
                try:
                    message = parent_connection.recv()
                except EOFError:
                    pipe_closed = True
                else:
                    if message["kind"] == "runtime":
                        report.update({key: message[key] for key in ("runtime", "input")})
                    elif message["kind"] == "cycle":
                        report["cycles"].append(message["cycle"])
                        cycle = message["cycle"]
                        print(json.dumps({"cycle": cycle["cycle"],
                                          "before_load": cycle["before_load"],
                                          "after_inference": cycle["after_inference"],
                                          "one_second_after_close": cycle["one_second_after_close"]}), flush=True)
                    else:
                        report["error"] = message
                    save()
            elif pipe_closed:
                child.join(.2)
            if time.monotonic() > deadline:
                forced_termination = True
                stop_probe(child)
                break
        child.join(5)
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        # Ctrl-C, failed report writes, and channel errors must not strand an
        # independently spawned native model process.
        try:
            if started:
                if child.is_alive():
                    forced_termination = True
                    stop_probe(child)
                child.join(0)
                identity_running = observed_child.is_running() if observed_child is not None else None
                report["process_exit"] = {
                    "pid": child.pid,
                    "exitcode": child.exitcode,
                    "multiprocessing_alive": child.is_alive(),
                    "original_pid_identity_running": identity_running,
                    "forced_termination": forced_termination,
                }
                passed = (
                    len(report["cycles"]) == 3 and child.exitcode == 0
                    and not child.is_alive() and identity_running is False
                    and not forced_termination and "error" not in report
                    and all(cycle["model_reference_cleared"] for cycle in report["cycles"])
                )
            report["controller_after_child_exit"] = memory(controller)
            report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            report["status"] = "completed" if passed else "failed"
            if passed:
                baseline = report["cycles"][0]["before_load"]
                final = report["cycles"][-1]["one_second_after_close"]
                report["observations"] = {
                    "same_child_pid_all_cycles": len({cycle["pid"] for cycle in report["cycles"]}) == 1,
                    "last_close_rss_above_initial_baseline_mib": final["rss_mib"] - baseline["rss_mib"],
                    "last_close_uss_above_initial_baseline_mib": final["uss_mib"] - baseline["uss_mib"],
                    "process_exit_confirmed": True,
                    "interpretation": "Report observed close residuals separately from confirmed process exit; no conclusion about unbounded leaks from three cycles alone.",
                }
            save()
        finally:
            parent_connection.close()
            child_connection.close()
            if not child.is_alive():
                child.close()
    print(f"{report['status']}: {args.report}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
