"""Register and validate an example list through the authenticated admin API."""
import argparse
import json
import os

import httpx
from model_service.contracts import ModelConfig


def verify_existing_config(client, config):
    requested = ModelConfig.model_validate(config)
    response = client.get("/admin/models")
    response.raise_for_status()
    existing = next((row["config"] for row in response.json()["models"]
                     if row["model_id"] == requested.model_id), None)
    if existing is None or ModelConfig.model_validate(existing) != requested:
        raise SystemExit(f"{requested.model_id}: existing configuration differs; use a new unique version")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--defaults", action="store_true", help="Set default aliases for listed capabilities")
    parser.add_argument("--unload-after-validation", action="store_true")
    args = parser.parse_args()
    key = os.environ.get("ADMIN_API_KEY")
    if not key:
        parser.error("ADMIN_API_KEY is required")
    with open(args.file, encoding="utf-8") as source:
        configs = json.load(source)
    with httpx.Client(base_url=args.url, headers={"Authorization": f"Bearer {key}"}, timeout=600) as client:
        for config in configs:
            model_id = config["name"] + "@" + config["version"]
            response = client.post("/admin/models", json={"config": config, "enable": True})
            if response.status_code == 409 and response.json().get("error", {}).get("code") == "version_exists":
                # Re-run the real validation rather than treating an existing row as healthy.
                verify_existing_config(client, config)
                response = client.post(f"/admin/models/{model_id}/validate")
            if response.is_error:
                raise SystemExit(f"{model_id}: {response.status_code} {response.text}")
            print(model_id, "validated", "MOCK" if config["backend"] == "mock" else config["backend"], flush=True)
            if args.defaults:
                for capability in config["capabilities"]:
                    response = client.put(f"/admin/aliases/default:{capability}", json={"model_id": model_id})
                    response.raise_for_status()
            if args.unload_after_validation:
                response = client.post(f"/admin/models/{model_id}/unload")
                response.raise_for_status()


if __name__ == "__main__":
    main()
