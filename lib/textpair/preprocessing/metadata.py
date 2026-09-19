"""PhiloLogic metadata lookup.

Metadata for a text object is assembled by walking up the OHCO hierarchy from
the object's own level to the document, taking the first non-empty value for
each field. Results are cached per level, so in practice only the most specific
level costs a query.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any

PHILO_LEVELS: dict[str, int] = {
    "doc": 1,
    "div1": 2,
    "div2": 3,
    "div3": 4,
    "para": 5,
    "sent": 6,
    "word": 7,
}
LEVEL_NAMES: dict[int, str] = {level: name for name, level in PHILO_LEVELS.items()}


class _Database:
    """A toms.db, opened once per process and shared by every file beside it."""

    __slots__ = ("cursor", "levels_present", "rows", "text_path")

    def __init__(self, db_path: str, text_path: str):
        self.text_path = text_path
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        self.cursor = connection.cursor()
        self.rows: dict[str, dict[str, Any]] = {}
        # Which OHCO levels exist as objects. Asked one indexed lookup at a time:
        # SELECT DISTINCT philo_type scans a covering index, which is 175ms on a
        # 3M-row toms.db and was being paid once per document.
        self.levels_present = set()
        for name, level in PHILO_LEVELS.items():
            self.cursor.execute("SELECT 1 FROM toms WHERE philo_type = ? LIMIT 1", (name,))
            if self.cursor.fetchone() is not None:
                self.levels_present.add(level)


# Keyed by path. A worker handles many files from one database, and the row cache
# is worth sharing between them: every text object walks up to the same document.
_DATABASES: dict[str, _Database] = {}


class MetadataLookup:
    """Reads text-object metadata out of a PhiloLogic toms.db."""

    __slots__ = ("words_file", "db_path", "text_path", "available", "_database")

    def __init__(self, words_file: str):
        data_dir = os.path.abspath(os.path.join(words_file, os.pardir, os.pardir))
        self.text_path = os.path.join(data_dir, "TEXT")
        self.db_path = os.path.join(data_dir, "toms.db")
        self.words_file = words_file
        self.available = os.path.exists(self.db_path)
        self._database: _Database | None = None
        if self.available:
            database = _DATABASES.get(self.db_path)
            if database is None:
                database = _DATABASES[self.db_path] = _Database(self.db_path, self.text_path)
            self._database = database

    def __call__(self, position: str, object_type: str) -> dict[str, Any]:
        """Metadata for the text object at `position`, merged up the hierarchy."""
        database = self._database
        if database is None:
            return {"filename": os.path.basename(self.words_file)}
        cursor = database.cursor
        object_id = position.split()
        level = PHILO_LEVELS[object_type]
        metadata: dict[str, Any] = {"parsed_filename": self.words_file}
        while object_id:
            padding = " ".join("0" for _ in range(7 - level))
            current_id = f"{' '.join(object_id[:level])} {padding}"
            row = database.rows.get(current_id)
            if row is None and level in database.levels_present:
                cursor.execute("SELECT * from toms WHERE philo_id = ?", (current_id,))
                result = cursor.fetchone()
                if result is not None:
                    row = self._row_to_fields(result, level)
                    database.rows[current_id] = row
            if row is not None:
                for field, value in row.items():
                    if field not in metadata or not metadata[field]:
                        metadata[field] = value
                # Only levels that exist as objects get an id. Sentences have no
                # toms row, so there is no philo_sent_id, as before.
                philo_object_id = f"philo_{LEVEL_NAMES[level]}_id"
                if not metadata.get(philo_object_id):
                    metadata[philo_object_id] = " ".join(object_id[:level])
            object_id.pop()
            level -= 1
        return metadata

    def _row_to_fields(self, result: sqlite3.Row, level: int) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        for field in result.keys():
            value = result[field]
            # At the document level keep every field, even empty ones, so the
            # key exists downstream; above it, empty values must not mask a
            # populated value from a higher level.
            if not value and level != 1:
                continue
            if field == "filename" and value:
                value = os.path.join(self.text_path, value)
            fields[field] = "" if value is None else value
        return fields
