"""Smoke test the prepared real registry through a temporary CLI/TCP server."""
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


def verify_chat_stream(client, business, admin):
    payload = {"input": {"messages": [{"role": "user", "content": "请用中文介绍什么是机器学习。"}],
                         "max_new_tokens": 64}, "stream": True}
    frames = []
    began = time.monotonic()
    with client.stream("POST", "/v1/chat", headers=business, json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            frame = json.loads(line[6:])
            assert "error" not in frame, frame
            frames.append({"at_seconds": round(time.monotonic() - began, 3), **frame})
            if len(frames) == 1:
                status = client.get("/admin/status", headers=admin).json()
                assert status["scheduler"]["active_count"] == 1
    chunks = [frame["output"] for frame in frames if frame.get("output") is not None]
    assert len(chunks) > 2 and frames[-1]["done"]
    assert "".join(chunk["delta"] for chunk in chunks) == chunks[-1]["text"]
    assert frames[0]["at_seconds"] < frames[-1]["at_seconds"]
    before = client.get("/admin/status", headers=admin).json()
    active_worker = next(worker for worker in before["workers"].values() if worker["alive"])
    with client.stream("POST", "/v1/chat", headers=business, json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith("data: "):
                first = json.loads(line[6:])
                assert not first["done"] and first["mock"] is False
                break
        began = time.monotonic()
    deadline = time.monotonic() + 15
    while True:
        status = client.get("/admin/status", headers=admin).json()
        if status["scheduler"]["active_count"] == 0:
            break
        assert time.monotonic() < deadline, "Disconnect did not release request after stop"
        time.sleep(.02)
    cancelled = next(worker for worker in status["workers"].values() if worker["alive"])
    assert cancelled["busy"] == 0
    assert cancelled["pid"] == active_worker["pid"] and cancelled["load_count"] == 1
    assert status["scheduler"]["temporary_mb"] == 0
    disconnect_stop_seconds = round(time.monotonic() - began, 3)
    reuse = client.post("/v1/chat", headers=business,
                        json={"input": {"messages": [{"role": "user", "content": "请只回答：你好"}], "max_new_tokens": 8}})
    reuse.raise_for_status()
    assert reuse.json()["mock"] is False
    return {"frames": len(frames), "chunks": len(chunks), "first_chunk_seconds": frames[0]["at_seconds"],
            "terminal_seconds": frames[-1]["at_seconds"], "output": chunks[-1],
            "disconnect_stop_seconds": disconnect_stop_seconds,
            "cancelled_worker": cancelled, "reuse": reuse.json(), "permit_held_during_output": True}


def main():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, BUSINESS_API_KEY=secrets.token_hex(24), ADMIN_API_KEY=secrets.token_hex(24))
    log_path = Path("var/http-real-smoke.log")
    log_path.parent.mkdir(exist_ok=True)
    report = {"timestamp_unix": time.time(), "transport": "real TCP HTTP / uvicorn CLI",
              "deployment_mode": "production", "allow_mock": False, "results": {}}
    with log_path.open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "model_service", "--config", "examples/service.production.json",
                                    "--port", str(port)], env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=180) as client:
                deadline = time.monotonic() + 15
                while True:
                    try:
                        if client.get("/health", timeout=.2).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(f"CLI failed to start; inspect {log_path}")
                    time.sleep(.05)
                for config in json.loads(Path("examples/legacy/models.real.json").read_text()):
                    capability = config["capabilities"][0]
                    began = time.monotonic()
                    response = client.post("/v1/" + capability, headers={"Authorization": "Bearer " + env["BUSINESS_API_KEY"]},
                                           json={"input": config["validation_input"]})
                    response.raise_for_status()
                    result = response.json()
                    assert result["mock"] is False and result["done"] is True
                    if "audio_base64" in result["output"]:
                        result["output"]["audio_base64_chars"] = len(result["output"].pop("audio_base64"))
                    if capability == "chat":
                        report["chat_streaming"] = verify_chat_stream(client,
                            {"Authorization": "Bearer " + env["BUSINESS_API_KEY"]},
                            {"Authorization": "Bearer " + env["ADMIN_API_KEY"]})
                    release = client.post("/admin/models/" + result["model"] + "/unload",
                                          headers={"Authorization": "Bearer " + env["ADMIN_API_KEY"]})
                    release.raise_for_status()
                    report["results"][capability] = {"elapsed_s": time.monotonic() - began, "response": result,
                                                     "unload": release.json()["release"]}
                status = client.get("/admin/status", headers={"Authorization": "Bearer " + env["ADMIN_API_KEY"]}).json()
                assert status["scheduler"]["active_count"] == status["scheduler"]["reserved_mb"] == 0
                report["scheduler_after_unload"] = status["scheduler"]
                ready = client.get("/ready")
                ready.raise_for_status()
                metrics = client.get("/admin/metrics", headers={"Authorization": "Bearer " + env["ADMIN_API_KEY"]})
                metrics.raise_for_status()
                assert "model_service_active_requests 0" in metrics.text
                report["ready"] = ready.json()
                report["metrics"] = metrics.text
        finally:
            process.terminate()
            process.wait(timeout=30)
    # Uvicorn can re-raise the handled termination signal after graceful shutdown.
    assert process.returncode in (0, -signal.SIGTERM), process.returncode
    assert "Application shutdown complete." in log_path.read_text()
    report["shutdown"] = {"returncode": process.returncode, "application_shutdown_completed": True}
    report["status"] = "passed_real_http_models"
    Path("docs/http-real-smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(report["status"], ", ".join(report["results"]))


if __name__ == "__main__":
    main()
