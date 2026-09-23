"""Create an online SQLite snapshot, including committed WAL data, without overwrite."""
from __future__ import annotations

import argparse
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path


def backup_registry(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    destination = destination.absolute()
    if destination.resolve() == source:
        raise ValueError("Backup destination must differ from the source database")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Backup destination already exists; refusing to overwrite")
    if not source.is_file():
        raise ValueError("Source must be a SQLite database file")
    fd, temporary = tempfile.mkstemp(prefix=".registry-backup-", suffix=".sqlite3", dir=destination.parent)
    os.close(fd)  # mkstemp creates owner-only permissions.
    try:
        with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as reader:
            with closing(sqlite3.connect(temporary)) as writer:
                reader.backup(writer)
                if writer.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                    raise ValueError("SQLite backup integrity check failed")
        # Publish the complete file atomically and refuse a concurrent overwrite.
        os.link(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path, help="New file in an existing directory; never overwritten")
    args = parser.parse_args()
    try:
        backup_registry(args.source, args.destination)
    except (OSError, sqlite3.Error, ValueError) as error:
        parser.error(f"Backup failed ({type(error).__name__}); source must exist and destination must be a new file")
    print(f"Verified backup created: {args.destination}")


if __name__ == "__main__":
    main()
