from __future__ import annotations

import os
import sqlite3
from contextlib import closing

import pytest

from scripts.backup_registry import backup_registry


def test_online_backup_includes_committed_wal_and_keeps_source_usable(tmp_path):
    source, target = tmp_path / "live.db", tmp_path / "backup.db"
    with closing(sqlite3.connect(source)) as db:
        assert db.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        db.execute("PRAGMA wal_autocheckpoint=0")
        db.execute("CREATE TABLE data (value TEXT)")
        db.execute("INSERT INTO data VALUES ('committed')")
        db.commit()
        assert (tmp_path / "live.db-wal").stat().st_size > 0
        backup_registry(source, target)
        with closing(sqlite3.connect(target)) as copy:
            assert copy.execute("SELECT value FROM data").fetchall() == [("committed",)]
            assert copy.execute("PRAGMA quick_check").fetchone() == ("ok",)
        db.execute("INSERT INTO data VALUES ('after-backup')")
        db.commit()
        with closing(sqlite3.connect(target)) as copy:
            assert copy.execute("SELECT COUNT(*) FROM data").fetchone() == (1,)
    assert target.stat().st_mode & 0o077 == 0


def test_backup_refuses_source_alias_and_existing_destination(tmp_path):
    source = tmp_path / "live.db"
    with closing(sqlite3.connect(source)) as db:
        db.execute("CREATE TABLE data (value TEXT)")
    with pytest.raises(ValueError):
        backup_registry(source, source)
    symlink = tmp_path / "source-link"
    symlink.symlink_to(source)
    with pytest.raises(ValueError):
        backup_registry(source, symlink)
    existing = tmp_path / "existing.db"
    existing.write_text("precious")
    with pytest.raises(FileExistsError):
        backup_registry(source, existing)
    assert existing.read_text() == "precious"
    hardlink = tmp_path / "source-hardlink"
    os.link(source, hardlink)
    with pytest.raises(FileExistsError):
        backup_registry(source, hardlink)
