"""Reading PhiloLogic words_and_philo_ids files.

One line per word, JSON. Decoded through a typed msgspec Struct, which skips the
intermediate dict, and from a whole-file decompression rather than lz4's
readline.
"""

from __future__ import annotations

import os
from typing import Iterator

import lz4.frame
import msgspec

import numpy as np

from . import philo_scan
from .metadata import PHILO_LEVELS
from .normalize import Normalizer
from .tokens import FormTable, TextObject

SENT_LEVEL = PHILO_LEVELS["sent"]


class Word(msgspec.Struct, gc=False):
    """A word as written by the PhiloLogic parser.

    gc=False because these are short-lived and hold no references the collector
    needs to trace; it keeps them out of its bookkeeping.
    """

    token: str
    position: str
    start_byte: int
    end_byte: int
    philo_type: str = "word"


_DECODE = msgspec.json.Decoder(Word).decode
_DECODE_LINES = msgspec.json.Decoder(Word).decode_lines


def read_blob(path: str) -> bytes:
    """The decompressed contents of a words_and_philo_ids file."""
    with open(path, "rb") as handle:
        raw = handle.read()
    return lz4.frame.decompress(raw) if path.endswith(".lz4") else raw


def read_words(path: str) -> list[Word]:
    """Every word object in a words_and_philo_ids file.

    One decode_lines call rather than one decode per line: msgspec has no way to
    reuse an output object, so the objects get allocated either way, but the
    per-call overhead does not have to be.
    """
    return _DECODE_LINES(read_blob(path))


def split_text_objects(
    path: str,
    text_object_type: str,
    normalizer: Normalizer,
    keep_all: bool = False,
    keep_surface: bool = False,
    defer_normalization: bool = False,
) -> Iterator[tuple[TextObject, str]]:
    """Group a file's words into text objects, normalizing as we go.

    Yields (text_object, object_id) with metadata left unset; the caller attaches
    it, so metadata lookup and text extraction stay independent.

    `keep_all` keeps a placeholder for every filtered token so byte offsets still
    line up with the source, which is what VSA passage reconstruction needs.
    `keep_surface` additionally retains raw surface forms and philo positions.

    With `defer_normalization`, `forms` holds modernized surface forms and nothing
    is filtered: the spaCy stage needs every token in place to tag it, and only
    then can normalization decide what to drop.
    """
    level = PHILO_LEVELS[text_object_type]
    modernize = normalizer.modernize
    if defer_normalization:
        normalize = None
    elif normalizer.config.memoizable:
        normalize = normalizer.__call__
    else:
        normalize = normalizer.normalize

    want_sent_starts = normalizer.config.needs_spacy
    # Sentence ids cost a split and a join per token, and are only needed when
    # something reads them. The punctuation workaround below rewrites a token's
    # position, which can only change the object it lands in when objects are
    # finer-grained than the document -- a punct token cannot be attributed to a
    # different document -- and the rewritten position itself is only kept when
    # keep_surface is on.
    track_sentences = level > 1 or keep_surface or want_sent_starts

    if not track_sentences and not defer_normalization:
        scanned = _scanned_objects(path, level, normalizer, keep_all)
        if scanned is not None:
            yield from scanned
            return

    current_object_id: str | None = None
    current = TextObject()
    previous_sent_id: str | None = None
    current_sent_id = ""

    for word in read_words(path):
        if not track_sentences:
            position = word.position
            object_id = position[: position.find(" ")]
        else:
            # Works around a PhiloLogic parser bug that assigns punctuation to
            # the following sentence: reattach it to the sentence just closed.
            if word.philo_type == "punct" and current_sent_id:
                position = f"{current_sent_id} 0"
            else:
                position = word.position
            # Capped split: only the first SENT_LEVEL fields are ever read, and
            # a philo position has at least seven -- some corpora write nine.
            philo_id = position.split(" ", SENT_LEVEL)
            object_id = philo_id[0] if level == 1 else " ".join(philo_id[:level])
            current_sent_id = " ".join(philo_id[:SENT_LEVEL])

        if current_object_id is None:
            current_object_id = object_id
            current.first_position = position
        elif object_id != current_object_id:
            # raw_length, not forms: a text object whose every token was filtered
            # still existed and still gets a metadata entry, as it did before.
            if current.raw_length:
                yield current, current_object_id
            current = TextObject(first_position=position)
            current_object_id = object_id

        is_sent_start = current_sent_id != previous_sent_id
        previous_sent_id = current_sent_id
        if not current.raw_length:
            current.raw_start_byte = word.start_byte
        current.raw_length += 1
        current.raw_end_byte = word.end_byte

        form = modernize(word.token)
        if normalize is not None:
            form = normalize(form)
            if not form and not keep_all:
                continue
        current.forms.append(form)
        current.start_bytes.append(word.start_byte)
        current.end_bytes.append(word.end_byte)
        if keep_surface:
            current.surface_forms.append(word.token)
            current.positions.append(position)
        if want_sent_starts:
            current.sent_starts.append(is_sent_start)

    if current.raw_length and current_object_id is not None:
        yield current, current_object_id


def _scanned_objects(path: str, level: int, normalizer: Normalizer, keep_all: bool):
    """Text objects read by scanning the buffer rather than parsing each line.

    Returns None when the scanner does not recognise the file's JSON, leaving the
    caller to parse it properly. Only the case with no sentence tracking is
    handled here: nothing needs a token's position beyond which object it is in,
    so the whole file reduces to columns and one pass over the distinct tokens.
    """
    if normalizer.vocabulary is None:
        normalizer.vocabulary = philo_scan.Vocabulary()
    vocabulary = normalizer.vocabulary

    columns = philo_scan.read_columns(read_blob(path), level, vocabulary=vocabulary)
    if columns is None:
        return None

    # The vocabulary spans the worker's whole run, so a token is turned into a
    # string, modernized and normalized once -- the first time this worker meets
    # it -- rather than once per document it appears in.
    normalize = normalizer.normalize if not normalizer.config.memoizable else None
    if normalize is None:
        vocabulary.resolve(normalizer.from_raw)
    else:
        vocabulary.resolve(lambda token: normalize(normalizer.modernize(token)))

    table = None if keep_all else FormTable.from_vocabulary(vocabulary)
    return _emit_objects(columns, vocabulary.forms, vocabulary.kept, keep_all, table)


def _emit_objects(columns, forms, kept, keep_all: bool, table=None):
    new_object = columns.new_object
    if new_object.size == 0:
        return
    starts = np.flatnonzero(new_object)
    stops = np.concatenate((starts[1:], [new_object.size]))
    buffer = columns.buffer
    token_ids = columns.token_ids
    start_byte = columns.start_byte
    end_byte = columns.end_byte

    for start, stop in zip(starts, stops):
        ids = token_ids[start:stop]
        if keep_all:
            selected = ids
            starts_out = start_byte[start:stop]
            ends_out = end_byte[start:stop]
        else:
            mask = kept[ids]
            selected = ids[mask]
            starts_out = start_byte[start:stop][mask]
            ends_out = end_byte[start:stop][mask]
        text_object = TextObject(
            forms=[forms[i] for i in selected],
            start_bytes=starts_out.tolist(),
            end_bytes=ends_out.tolist(),
            form_ids=selected if table is not None else None,
            form_table=table,
            first_position=buffer[columns.position_lo[start]:columns.position_hi[start]]
                .tobytes().decode("utf8"),
            prefiltered=not keep_all,
            raw_length=int(stop - start),
            raw_start_byte=int(start_byte[start]),
            raw_end_byte=int(end_byte[stop - 1]),
        )
        object_id = buffer[columns.position_lo[start]:columns.object_hi[start]].tobytes().decode("utf8")
        yield text_object, object_id


def sent_metadata(text_object: TextObject) -> dict[str, object]:
    """Extra metadata a sentence-level object needs, which toms.db has no row for."""
    return {
        "philo_id": " ".join(text_object.first_position.split()[:SENT_LEVEL] + ["0"]),
        "philo_type": "sent",
        "start_byte": text_object.raw_start_byte,
        "end_byte": text_object.raw_end_byte,
        "word_count": text_object.raw_length,
    }
