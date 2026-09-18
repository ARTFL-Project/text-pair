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


class MetadataLookup:
    """Reads text-object metadata out of a PhiloLogic toms.db."""

    def __init__(self, words_file: str):
        data_dir = os.path.abspath(os.path.join(words_file, os.pardir, os.pardir))
        self.text_path = os.path.join(data_dir, "TEXT")
        self.db_path = os.path.join(data_dir, "toms.db")
        self.words_file = words_file
        self.available: bool = os.path.exists(self.db_path)
        self._cache: dict[str, dict[str, Any]] = {}
        self._cursor: sqlite3.Cursor | None = None
        # Levels with no rows at all cannot match any query. Sentences are never
        # in toms.db, so a sent-level run would otherwise pay one guaranteed-miss
        # query per sentence.
        self._levels_present: set[int] = set()
        if self.available:
            connection = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            self._cursor = connection.cursor()
            self._cursor.execute("SELECT DISTINCT philo_type FROM toms")
            self._levels_present = {
                PHILO_LEVELS[row[0]] for row in self._cursor.fetchall() if row[0] in PHILO_LEVELS
            }

    def __call__(self, position: str, object_type: str) -> dict[str, Any]:
        """Metadata for the text object at `position`, merged up the hierarchy."""
        if not self.available or self._cursor is None:
            return {"filename": os.path.basename(self.words_file)}
        object_id = position.split()
        level = PHILO_LEVELS[object_type]
        metadata: dict[str, Any] = {"parsed_filename": self.words_file}
        while object_id:
            padding = " ".join("0" for _ in range(7 - level))
            current_id = f"{' '.join(object_id[:level])} {padding}"
            row = self._cache.get(current_id)
            if row is None and level in self._levels_present:
                self._cursor.execute("SELECT * from toms WHERE philo_id = ?", (current_id,))
                result = self._cursor.fetchone()
                if result is not None:
                    row = self._row_to_fields(result, level)
                    self._cache[current_id] = row
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
