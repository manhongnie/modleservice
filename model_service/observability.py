"""Bounded audit events and process-lifetime Prometheus counters."""
from collections import Counter
import json
import logging
import time

log = logging.getLogger("model_service")


class Observability:
    def __init__(self, registry):
        self.registry = registry
        self.started_at = time.time()
        self.requests = Counter()
        self.request_seconds = 0.0
        self.execution_seconds = 0.0
        self.executions = 0

    def event(self, kind: str, **data):
        self.registry.event(kind, **data)
        log.info(json.dumps({"event": kind, **data}, ensure_ascii=False))
        if kind == "request_finished":
            self.requests[data["outcome"]] += 1
            self.request_seconds += data["duration_seconds"]
        elif kind == "execution_finished":
            self.executions += 1
            self.execution_seconds += data["execution_seconds"]

    def metrics(self, scheduler: dict, workers: dict) -> str:
        lines = ["# TYPE model_service_up gauge", "model_service_up 1",
                 "# TYPE model_service_started_timestamp_seconds gauge",
                 f"model_service_started_timestamp_seconds {self.started_at}",
                 "# TYPE model_service_requests_total counter"]
        for outcome, count in sorted(self.requests.items()):
            escaped = outcome.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            lines.append(f'model_service_requests_total{{outcome="{escaped}"}} {count}')
        gauges = {"queue_length": scheduler["queue_length"], "active_requests": scheduler["active_count"],
                  "resident_bytes": scheduler["resident_mb"] * 1048576,
                  "temporary_bytes": scheduler["temporary_mb"] * 1048576,
                  "budget_bytes": scheduler["budget_mb"] * 1048576,
                  "gpu_resident_bytes": scheduler["gpu_resident_mb"] * 1048576,
                  "gpu_temporary_bytes": scheduler["gpu_temporary_mb"] * 1048576,
                  "gpu_budget_bytes": scheduler["gpu_budget_mb"] * 1048576,
                  "quarantined_requests": len(scheduler["quarantined_requests"]),
                  "worker_rss_bytes": sum(worker.get("rss_bytes", 0) for worker in workers.values()),
                  "workers": sum(bool(worker.get("alive")) for worker in workers.values())}
        for name, value in gauges.items():
            lines.extend([f"# TYPE model_service_{name} gauge", f"model_service_{name} {value}"])
        for name, value in {"request_duration_seconds_sum": self.request_seconds,
                            "execution_duration_seconds_sum": self.execution_seconds,
                            "executions_total": self.executions}.items():
            lines.extend([f"# TYPE model_service_{name} counter", f"model_service_{name} {value}"])
        return "\n".join(lines) + "\n"
