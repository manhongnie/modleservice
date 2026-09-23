"""Persistence boundary regressions; no model processes or external services."""
from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import closing

import pytest

from model_service.contracts import ModelConfig, ServiceError
from model_service.registry import Registry


def model(**changes):
    values = {"name": "registered-model", "version": "1", "capabilities": ["chat"],
              "backend": "mock", "task": "mock", "resident_mb": 20, "request_mb": 5,
              "validation_input": {"messages": [{"role": "user", "content": "test"}]}}
    return ModelConfig(**(values | changes))


def enable(registry, config):
    registry.register(config)
    registry.validation(config.model_id, True)
    registry.enable(config.model_id, True)


def track_connections(monkeypatch, *, factory=sqlite3.Connection):
    original_connect = sqlite3.connect
    opened = []

    def connect(*args, **kwargs):
        connection = original_connect(*args, factory=factory, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr("model_service.registry.sqlite3.connect", connect)
    return opened, original_connect


def test_unknown_schema_closes_connection_without_changing_journal_or_schema(tmp_path, monkeypatch):
    path = tmp_path / "future.db"
    with closing(sqlite3.connect(path)) as db:
        db.execute("PRAGMA user_version=99")
        db.execute("CREATE TABLE future_data (value TEXT)")
        journal = db.execute("PRAGMA journal_mode").fetchone()[0]
    opened, connect = track_connections(monkeypatch)
    with pytest.raises(RuntimeError, match="Unsupported registry schema version: 99"):
        Registry(str(path))
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")
    with closing(connect(path)) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 99
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == journal
        assert db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [("future_data",)]


def test_schema_initialization_failure_also_closes_connection(tmp_path, monkeypatch):
    class FailedSchemaConnection(sqlite3.Connection):
        def executescript(self, script):
            raise sqlite3.DatabaseError("Simulated schema initialization failure")

    opened, _ = track_connections(monkeypatch, factory=FailedSchemaConnection)
    with pytest.raises(sqlite3.DatabaseError, match="initialization failure"):
        Registry(str(tmp_path / "broken.db"))
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")


def test_registry_sets_bounded_busy_timeout_and_durable_wal_options(tmp_path, monkeypatch):
    opened, _ = track_connections(monkeypatch)
    registry = Registry(str(tmp_path / "registry.db"))
    try:
        db = opened[0]
        assert db.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert db.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1
    finally:
        registry.close()


def test_registry_write_waits_for_short_lived_database_writer(tmp_path):
    path = tmp_path / "registry.db"
    registry = Registry(str(path))
    locked = threading.Event()
    failures = []

    def briefly_hold_writer():
        try:
            with closing(sqlite3.connect(path)) as db:
                db.execute("BEGIN IMMEDIATE")
                locked.set()
                time.sleep(.05)
                db.rollback()
        except BaseException as exc:
            failures.append(exc)
            locked.set()

    writer = threading.Thread(target=briefly_hold_writer)
    writer.start()
    try:
        assert locked.wait(2)
        registry.register(model())
        assert registry.get(model().model_id)["config"] == model()
    finally:
        writer.join(2)
        registry.close()
    assert not writer.is_alive()
    assert failures == []


def test_register_refuses_to_shadow_existing_alias_and_preserves_target(tmp_path):
    path = tmp_path / "registry.db"
    registry = Registry(str(path))
    target = model()
    enable(registry, target)
    registry.set_alias("stable", target.model_id)
    try:
        with pytest.raises(ServiceError) as error:
            registry.register(model(name="stable"))
        assert error.value.code == "alias_conflict"
        assert len(registry.all()) == 1
        assert registry.resolve("stable", "chat")["config"] == target
    finally:
        registry.close()
    restarted = Registry(str(path))
    try:
        assert len(restarted.all()) == 1
        assert restarted.resolve("stable", "chat")["config"] == target
    finally:
        restarted.close()


def test_duplicate_version_does_not_replace_original_configuration(tmp_path):
    registry = Registry(str(tmp_path / "registry.db"))
    original = model(resident_mb=20)
    try:
        registry.register(original)
        with pytest.raises(ServiceError) as error:
            registry.register(model(resident_mb=100))
        assert error.value.code == "version_exists"
        assert registry.get(original.model_id)["config"] == original
    finally:
        registry.close()


@pytest.mark.parametrize("operation", [
    lambda registry: registry.validation("missing@1", True),
    lambda registry: registry.management("missing@1", "ready"),
    lambda registry: registry.remove("missing@1"),
    lambda registry: registry.uncertain("request", "missing@1", "remote intent"),
])
def test_unknown_model_mutations_report_not_found(tmp_path, operation):
    registry = Registry(str(tmp_path / "registry.db"))
    try:
        with pytest.raises(ServiceError) as error:
            operation(registry)
        assert error.value.code == "model_not_found"
        assert registry.all() == []
        assert registry.uncertainties() == []
    finally:
        registry.close()


def test_configuration_aliases_dependencies_and_uncertainty_survive_restart(tmp_path):
    path = tmp_path / "registry.db"
    registry = Registry(str(path))
    config = model()
    enable(registry, config)
    registry.set_alias("default:chat", config.model_id)
    registry.add_dependency("business:chat", config.model_id)
    registry.uncertain("remote-request", config.model_id, "completion unknown")
    registry.management(config.model_id, "reconciliation_required")
    registry.event("model_registered", model=config.model_id)
    registry.close()
    restarted = Registry(str(path))
    try:
        row = restarted.get(config.model_id)
        assert row["config"] == config
        assert row["enabled"] and row["validated"]
        assert row["management_state"] == "reconciliation_required"
        assert restarted.resolve(None, "chat")["config"] == config
        assert restarted.references(config.model_id) == {
            "aliases": ["default:chat"], "dependencies": ["business:chat"]}
        assert restarted.uncertainties()[0]["request_id"] == "remote-request"
        assert restarted.events()[0]["data"] == {"model": config.model_id}
    finally:
        restarted.close()


def test_failed_revalidation_disables_model_persistently(tmp_path):
    path = tmp_path / "registry.db"
    registry = Registry(str(path))
    config = model()
    enable(registry, config)
    registry.validation(config.model_id, False, "artifact no longer loads")
    registry.close()
    restarted = Registry(str(path))
    try:
        row = restarted.get(config.model_id)
        assert not row["validated"]
        assert not row["enabled"]
        assert row["validation_error"] == "artifact no longer loads"
        with pytest.raises(ServiceError) as error:
            restarted.enable(config.model_id, True)
        assert error.value.code == "not_validated"
    finally:
        restarted.close()


def test_uncertain_execution_cannot_be_overwritten_by_another_model(tmp_path):
    registry = Registry(str(tmp_path / "registry.db"))
    first, second = model(), model(name="other-model")
    try:
        registry.register(first)
        registry.register(second)
        registry.uncertain("same-request", first.model_id, "first intent")
        registry.uncertain("same-request", first.model_id, "unknown completion")
        with pytest.raises(ServiceError) as error:
            registry.uncertain("same-request", second.model_id, "conflicting intent")
        assert error.value.code == "duplicate_request"
        record, = registry.uncertainties()
        assert record["model_id"] == first.model_id
        assert record["reason"] == "unknown completion"
    finally:
        registry.close()


def test_removal_requires_remote_reconciliation_and_preserves_artifact(tmp_path):
    registry = Registry(str(tmp_path / "registry.db"))
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"synthetic artifact")
    config = model(path=str(artifact))
    try:
        registry.register(config)
        registry.uncertain("remote-request", config.model_id, "unknown completion")
        with pytest.raises(ServiceError) as error:
            registry.remove(config.model_id)
        assert error.value.code == "execution_unknown"
        assert registry.get(config.model_id)["config"] == config
        registry.clear_uncertainty("remote-request")
        registry.remove(config.model_id)
        assert registry.all() == []
        assert artifact.read_bytes() == b"synthetic artifact"
    finally:
        registry.close()
