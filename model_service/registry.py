"""Persistent model configuration, independent of transient execution state."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .contracts import ModelConfig, ServiceError


class Registry:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=5)
        self.db.row_factory = sqlite3.Row
        try:
            self.db.execute("PRAGMA busy_timeout=5000")
            version = self.db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise RuntimeError(f"Unsupported registry schema version: {version}")
            self.db.execute("PRAGMA foreign_keys=ON")
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript("""
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY, config TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 0, validated INTEGER NOT NULL DEFAULT 0,
                validation_error TEXT, management_state TEXT NOT NULL DEFAULT 'ready',
                created_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS aliases (
                alias TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models(model_id));
            CREATE TABLE IF NOT EXISTS dependencies (
                dependency TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models(model_id));
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL NOT NULL,
                kind TEXT NOT NULL, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS uncertainties (
                request_id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models(model_id),
                reason TEXT NOT NULL, timestamp REAL NOT NULL);
            PRAGMA user_version=1;
            """)
        except BaseException:
            self.db.close()
            raise

    def _row(self, row) -> dict[str, Any]:
        return {**dict(row), "config": ModelConfig.model_validate_json(row["config"]),
                "enabled": bool(row["enabled"]), "validated": bool(row["validated"])}

    def get(self, model_id: str) -> dict[str, Any]:
        row = self.db.execute("SELECT * FROM models WHERE model_id=?", (model_id,)).fetchone()
        if row is None:
            raise ServiceError("model_not_found", "Model is not registered", 404)
        return self._row(row)

    def all(self) -> list[dict[str, Any]]:
        return [self._row(r) for r in self.db.execute("SELECT * FROM models ORDER BY model_id")]

    def resolve(self, reference: str | None, capability: str) -> dict[str, Any]:
        reference = reference or f"default:{capability}"
        alias = self.db.execute("SELECT model_id FROM aliases WHERE alias=?", (reference,)).fetchone()
        if alias:
            return self.get(alias[0])
        if "@" in reference:
            return self.get(reference)
        matches = [row for row in self.all() if row["config"].name == reference]
        if not matches:
            raise ServiceError("model_not_found", "Model or default alias is not registered", 404)
        if len(matches) != 1:
            raise ServiceError("ambiguous_model", "Use a versioned model name or alias", 409)
        return matches[0]

    def register(self, config: ModelConfig) -> None:
        try:
            with self.db:
                inserted = self.db.execute("""
                    INSERT INTO models(model_id,config,created_at)
                    SELECT ?,?,? WHERE NOT EXISTS (SELECT 1 FROM aliases WHERE alias=?)
                """, (config.model_id, config.model_dump_json(), time.time(), config.name))
                if not inserted.rowcount:
                    raise ServiceError("alias_conflict", "Model name conflicts with an existing alias", 409)
        except sqlite3.IntegrityError:
            raise ServiceError("version_exists", "Model version already exists; register a new version", 409) from None

    def validation(self, model_id: str, success: bool, error: str | None = None) -> None:
        with self.db:
            updated = self.db.execute("UPDATE models SET validated=?, validation_error=?, enabled=CASE WHEN ? THEN enabled ELSE 0 END WHERE model_id=?",
                                      (success, error, success, model_id))
            if not updated.rowcount:
                raise ServiceError("model_not_found", "Model is not registered", 404)

    def enable(self, model_id: str, enabled: bool) -> None:
        row = self.get(model_id)
        if enabled and not row["validated"]:
            raise ServiceError("not_validated", "Model must pass validation before enabling", 409)
        with self.db:
            self.db.execute("UPDATE models SET enabled=? WHERE model_id=?", (enabled, model_id))

    def management(self, model_id: str, state: str) -> None:
        with self.db:
            updated = self.db.execute("UPDATE models SET management_state=? WHERE model_id=?", (state, model_id))
            if not updated.rowcount:
                raise ServiceError("model_not_found", "Model is not registered", 404)

    def references(self, model_id: str) -> dict[str, list[str]]:
        return {
            "aliases": [r[0] for r in self.db.execute("SELECT alias FROM aliases WHERE model_id=?", (model_id,))],
            "dependencies": [r[0] for r in self.db.execute("SELECT dependency FROM dependencies WHERE model_id=?", (model_id,))],
        }

    def aliases(self) -> dict[str, str]:
        return dict(self.db.execute("SELECT alias,model_id FROM aliases"))

    def dependencies(self) -> dict[str, str]:
        return dict(self.db.execute("SELECT dependency,model_id FROM dependencies"))

    def set_alias(self, alias: str, model_id: str) -> None:
        row = self.get(model_id)
        if not row["enabled"] or row["management_state"] != "ready":
            raise ServiceError("model_unavailable", "Alias target must be enabled and accepting requests", 409)
        if "@" in alias or any(r["config"].name == alias for r in self.all()):
            raise ServiceError("alias_conflict", "Alias must not shadow a model name or version", 409)
        if alias.startswith("default:") and alias.split(":", 1)[1] not in row["config"].capabilities:
            raise ServiceError("unsupported_capability", "Default alias capability does not match model", 409)
        with self.db:
            self.db.execute("INSERT INTO aliases VALUES(?,?) ON CONFLICT(alias) DO UPDATE SET model_id=excluded.model_id", (alias, model_id))

    def delete_alias(self, alias: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM aliases WHERE alias=?", (alias,))

    def add_dependency(self, dependency: str, model_id: str) -> None:
        self.get(model_id)
        with self.db:
            self.db.execute("INSERT INTO dependencies VALUES(?,?) ON CONFLICT(dependency) DO UPDATE SET model_id=excluded.model_id", (dependency, model_id))

    def delete_dependency(self, dependency: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM dependencies WHERE dependency=?", (dependency,))

    def remove(self, model_id: str) -> None:
        self.get(model_id)
        refs = self.references(model_id)
        if any(refs.values()):
            raise ServiceError("model_referenced", "Reassign or delete aliases and registered dependencies first", 409)
        if self.db.execute("SELECT 1 FROM uncertainties WHERE model_id=? LIMIT 1", (model_id,)).fetchone():
            raise ServiceError("execution_unknown", "Confirm outstanding remote executions stopped before removal", 409)
        with self.db:
            self.db.execute("DELETE FROM models WHERE model_id=?", (model_id,))

    def uncertain(self, request_id: str, model_id: str, reason: str) -> None:
        self.get(model_id)
        with self.db:
            updated = self.db.execute("""
                INSERT INTO uncertainties VALUES(?,?,?,?)
                ON CONFLICT(request_id) DO UPDATE SET reason=excluded.reason,timestamp=excluded.timestamp
                WHERE uncertainties.model_id=excluded.model_id
            """, (request_id, model_id, reason, time.time()))
            if not updated.rowcount:
                raise ServiceError("duplicate_request", "Request ID already belongs to another model's uncertain execution", 409)

    def uncertainties(self) -> list[dict]:
        return [dict(row) for row in self.db.execute("SELECT * FROM uncertainties")]

    def clear_uncertainty(self, request_id: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM uncertainties WHERE request_id=?", (request_id,))

    def event(self, kind: str, **data: Any) -> None:
        with self.db:
            self.db.execute("INSERT INTO events(timestamp,kind,data) VALUES(?,?,?)", (time.time(), kind, json.dumps(data, ensure_ascii=False)))
            self.db.execute("DELETE FROM events WHERE seq <= (SELECT COALESCE(MAX(seq),0)-10000 FROM events)")

    def events(self, limit: int = 100) -> list[dict]:
        return [{**dict(r), "data": json.loads(r["data"])} for r in self.db.execute("SELECT * FROM events ORDER BY seq DESC LIMIT ?", (limit,))]

    def close(self) -> None:
        self.db.close()
