"""Run exactly one control process; model execution uses reusable child processes."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .contracts import Settings


def load_settings(config_path: str | None = None) -> Settings:
    path = config_path or os.environ.get("MODEL_SERVICE_CONFIG") or "examples/service.json"
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for environment, field in (("BUSINESS_API_KEY", "business_keys"), ("ADMIN_API_KEY", "admin_keys")):
        if environment in os.environ:
            data[field] = [os.environ[environment]]
    # Settings rejects absent, empty or shared keys; no public default credentials.
    return Settings.model_validate(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-host multi-model inference service")
    parser.add_argument("--config", help="JSON settings path; defaults to MODEL_SERVICE_CONFIG")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", choices=["debug", "info", "warning", "error"], default="info")
    parser.add_argument("--check-config", action="store_true", help="Validate settings and exit without starting workers")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        settings = load_settings(args.config)
    except (OSError, ValueError) as error:
        # Pydantic errors may contain secrets; do not print the invalid settings.
        parser.error(f"Cannot read valid settings ({type(error).__name__}); configure distinct BUSINESS_API_KEY and ADMIN_API_KEY.")
    if args.check_config:
        print("Configuration is valid; no server or model processes started.")
        return
    import uvicorn
    from .api import create_app
    uvicorn.run(create_app(settings), host=args.host, port=args.port, workers=1, log_level=args.log_level,
                access_log=False, server_header=False, timeout_keep_alive=5,
                limit_concurrency=settings.max_http_requests + 16,
                # Never impose a hard server shutdown cutoff on native inference.
                timeout_graceful_shutdown=None)


if __name__ == "__main__":
    main()
