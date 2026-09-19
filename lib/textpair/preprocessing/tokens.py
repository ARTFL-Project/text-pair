"""Columnar text objects.

Parallel lists rather than a sequence of token objects, because that is the shape
every consumer wants: the n-gram writer takes three columns and the VSA token
cache four.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np


@dataclass(slots=True)
class FormTable:
    """The distinct normalized forms of a file, as bytes a kernel can index.

    `offsets[i]:offsets[i+1]` is form i in `data`. Shared by every text object
    read from the same file, since they were interned together.
    """

    data: np.ndarray
    offsets: np.ndarray
    longest: int

    @classmethod
    def from_vocabulary(cls, vocabulary) -> "FormTable":
        """A view of a worker vocabulary's packed forms.

        No copy: the arrays are the vocabulary's own, sliced to what it has
        filled. They only ever grow, so a view taken now stays valid for the ids
        it was taken for.
        """
        return cls(vocabulary.form_data[: vocabulary.form_used],
                   vocabulary.form_offsets[: vocabulary.count + 1],
                   vocabulary.longest)

    @classmethod
    def build(cls, forms: list[str]) -> "FormTable":
        encoded = [form.encode("utf8") for form in forms]
        offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
        if encoded:
            offsets[1:] = np.cumsum([len(item) for item in encoded], dtype=np.int64)
        joined = b"".join(encoded)
        data = np.frombuffer(joined, dtype=np.uint8) if joined else np.zeros(0, dtype=np.uint8)
        return cls(data, offsets, max((len(item) for item in encoded), default=0))


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
    # Which distinct form each surviving token is, and the table to resolve them
    # against. Set only by the scanning reader, and only when nothing filtered
    # out, so an n-gram's key can be built without its string.
    form_ids: Any = None
    form_table: Any = None
    # n-gram keys, when they were computed without building the n-gram strings.
    keys: Any = None
    # Whether the reader already applied purge's rule, so purge has nothing to
    # find and need not walk every form looking.
    prefiltered: bool = False

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
        if self.prefiltered:
            self.sync_byte_range()
            return
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
