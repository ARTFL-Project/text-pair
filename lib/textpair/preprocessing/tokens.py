"""Columnar text objects.

Parallel lists rather than a sequence of token objects, because that is the shape
every consumer wants: the n-gram writer takes three columns and the VSA token
cache four.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass(slots=True)
class TextObject:
    """Normalized tokens for one PhiloLogic text object.

    An empty string in `forms` is a token that was filtered but kept in place,
    so byte offsets still line up with the source text. `surface_forms` and
    `positions` are only populated when the caller asks for them.
    """

    forms: list[str] = field(default_factory=list)
    start_bytes: list[int] = field(default_factory=list)
    end_bytes: list[int] = field(default_factory=list)
    surface_forms: list[str] = field(default_factory=list)
    positions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # True at each token that opens a new sentence. Only populated when a spaCy
    # pipeline needs sentence boundaries.
    sent_starts: list[bool] = field(default_factory=list)
    # The object's own philo position, and how many words it held before
    # filtering -- both go into sentence-level metadata, which toms.db lacks.
    first_position: str = ""
    raw_length: int = 0
    # The object's full extent, before filtering dropped any tokens. Sentence
    # metadata needs the true sentence span, not the surviving tokens' span.
    raw_start_byte: int = 0
    raw_end_byte: int = 0

    def __len__(self) -> int:
        return len(self.forms)

    def __bool__(self) -> bool:
        return bool(self.forms)

    def __iter__(self) -> Iterator[tuple[str, int, int]]:
        return zip(self.forms, self.start_bytes, self.end_bytes)

    @property
    def text(self) -> list[str]:
        """Non-empty normalized forms, for vectorizers."""
        return [form for form in self.forms if form]

    def extend(self, other: "TextObject") -> None:
        """Append another text object's tokens, keeping this one's metadata."""
        self.forms.extend(other.forms)
        self.start_bytes.extend(other.start_bytes)
        self.end_bytes.extend(other.end_bytes)
        self.surface_forms.extend(other.surface_forms)
        self.positions.extend(other.positions)
        self.sent_starts.extend(other.sent_starts)
        self.raw_length += other.raw_length
        if not self.metadata:
            self.metadata = dict(other.metadata)
        elif other.end_bytes:
            self.metadata["end_byte"] = other.end_bytes[-1]

    def purge(self) -> None:
        """Drop filtered tokens, then resync the metadata byte range."""
        keep = [index for index, form in enumerate(self.forms) if form and form != " "]
        if len(keep) != len(self.forms):
            self.forms = [self.forms[i] for i in keep]
            self.start_bytes = [self.start_bytes[i] for i in keep]
            self.end_bytes = [self.end_bytes[i] for i in keep]
            if self.surface_forms:
                self.surface_forms = [self.surface_forms[i] for i in keep]
            if self.positions:
                self.positions = [self.positions[i] for i in keep]
            if self.sent_starts:
                self.sent_starts = [self.sent_starts[i] for i in keep]
        self.sync_byte_range()

    def sync_byte_range(self) -> None:
        if self.start_bytes:
            self.metadata["start_byte"] = self.start_bytes[0]
            self.metadata["end_byte"] = self.end_bytes[-1]
        else:
            self.metadata["start_byte"] = 0
            self.metadata["end_byte"] = 0
