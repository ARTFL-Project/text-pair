"""Scanning PhiloLogic word JSON without parsing it.

A words file is one JSON object per line, and only four of its fields are ever
read. Handing each line to a JSON parser means allocating an object per token to
throw most of it away; on a 2.5M-word document that was 40% of the read.

This walks the raw buffer once in a numba kernel and writes out columns: the
byte span of each token, its start and end byte, whether the object it belongs
to differs from the previous line's, and whether it is punctuation. Tokens are
deduplicated as they are found, so the caller normalizes each distinct surface
form once -- 22.5k of them in a 2.5M-word document -- instead of once per token.

It is a scanner, not a parser: it assumes one object per line whose values are
scalars, and reads the fields it knows by name, in whatever order they appear,
ignoring the rest. Field order does vary between corpora -- frantext writes
pos before lemma, rousseau_holbach the other way -- and the number of fields
varies from five to nine, which is why nothing here is positional. What it does
not assume is that strings are free of escapes; `token_escaped` marks the ones
needing real JSON unescaping, a fraction of a percent of lines.

A line whose token or byte offsets it cannot find leaves the sentinel -1 in
place, and the caller falls back to a real parser for that file. So a shape this
does not understand is slow, never wrong.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Field names, as byte values, so the kernel can compare without building objects.
_TOKEN = np.frombuffer(b"token", dtype=np.uint8)
_POSITION = np.frombuffer(b"position", dtype=np.uint8)
_START_BYTE = np.frombuffer(b"start_byte", dtype=np.uint8)
_END_BYTE = np.frombuffer(b"end_byte", dtype=np.uint8)
_PHILO_TYPE = np.frombuffer(b"philo_type", dtype=np.uint8)
_PUNCT = np.frombuffer(b"punct", dtype=np.uint8)

QUOTE = 34       # "
BACKSLASH = 92   # \
COLON = 58       # :
COMMA = 44       # ,
OPEN = 123       # {
CLOSE = 125      # }
SPACE = 32
NEWLINE = 10


@njit(inline="always")
def _matches(buf, lo, hi, name):
    if hi - lo != name.size:
        return False
    for i in range(name.size):
        if buf[lo + i] != name[i]:
            return False
    return True


@njit(inline="always")
def _string_end(buf, start, limit):
    """Index of the closing quote of a JSON string whose content starts at `start`.

    Returns (end, escaped). `escaped` is set when a backslash appears inside, in
    which case the content needs real unescaping rather than a plain slice.
    """
    i = start
    escaped = False
    while i < limit:
        c = buf[i]
        if c == BACKSLASH:
            escaped = True
            i += 2
            continue
        if c == QUOTE:
            return i, escaped
        i += 1
    return limit, escaped


@njit(nogil=True, cache=True)
def scan_lines(buf, line_starts, level, want_punct,
               token_lo, token_hi, token_escaped, start_byte, end_byte,
               position_lo, position_hi, object_hi, is_punct):
    """Fill the output columns from one line per entry of `line_starts`.

    `object_hi` ends the first `level` space-separated fields of the position,
    which is the text object a token belongs to.

    Scanning a line stops once the wanted fields are in hand. Corpora carrying
    pos, tag, ent_type and lemma put roughly half of each line after the four
    fields that matter, and none of it is read.
    """
    wanted = 5 if want_punct else 4
    n = line_starts.size - 1
    for row in range(n):
        limit = line_starts[row + 1]
        i = line_starts[row]
        token_lo[row] = -1
        token_hi[row] = -1
        token_escaped[row] = 0
        start_byte[row] = -1
        end_byte[row] = -1
        position_lo[row] = -1
        position_hi[row] = -1
        object_hi[row] = -1
        is_punct[row] = 0

        found = 0
        while i < limit and buf[i] != OPEN:
            i += 1
        i += 1
        while i < limit and found < wanted:
            while i < limit and buf[i] != QUOTE and buf[i] != CLOSE:
                i += 1
            if i >= limit or buf[i] == CLOSE:
                break
            key_lo = i + 1
            key_hi, _ = _string_end(buf, key_lo, limit)
            i = key_hi + 1
            while i < limit and buf[i] != COLON:
                i += 1
            i += 1
            while i < limit and buf[i] == SPACE:
                i += 1
            if i >= limit:
                break

            if buf[i] == QUOTE:
                value_lo = i + 1
                value_hi, escaped = _string_end(buf, value_lo, limit)
                i = value_hi + 1
                if _matches(buf, key_lo, key_hi, _TOKEN):
                    token_lo[row] = value_lo
                    token_hi[row] = value_hi
                    token_escaped[row] = 1 if escaped else 0
                    found += 1
                elif _matches(buf, key_lo, key_hi, _POSITION):
                    position_lo[row] = value_lo
                    position_hi[row] = value_hi
                    found += 1
                    fields = 0
                    j = value_lo
                    while j < value_hi:
                        if buf[j] == SPACE:
                            fields += 1
                            if fields == level:
                                break
                        j += 1
                    object_hi[row] = j
                elif want_punct and _matches(buf, key_lo, key_hi, _PHILO_TYPE):
                    if _matches(buf, value_lo, value_hi, _PUNCT):
                        is_punct[row] = 1
                    found += 1
            else:
                value_lo = i
                negative = False
                if i < limit and buf[i] == 45:  # -
                    negative = True
                    i += 1
                value = 0
                digits = False
                while i < limit and 48 <= buf[i] <= 57:
                    value = value * 10 + (buf[i] - 48)
                    digits = True
                    i += 1
                if digits:
                    if negative:
                        value = -value
                    if _matches(buf, key_lo, key_hi, _START_BYTE):
                        start_byte[row] = value
                        found += 1
                    elif _matches(buf, key_lo, key_hi, _END_BYTE):
                        end_byte[row] = value
                        found += 1
                else:
                    while i < limit and buf[i] != COMMA and buf[i] != CLOSE:
                        i += 1

            while i < limit and buf[i] != COMMA and buf[i] != CLOSE:
                i += 1
            if i < limit and buf[i] == CLOSE:
                break
            i += 1


@njit(nogil=True, cache=True)
def intern_tokens(buf, token_lo, token_hi, table_keys, table_ids,
                  distinct_lo, distinct_hi, token_ids):
    """Map each token span to a dense id, deduplicating as we go.

    Open addressing over `table_keys`, which the caller sizes to a power of two
    comfortably larger than the number of distinct tokens it expects. Returns the
    number of distinct tokens found, or -1 if the table filled up.
    """
    mask = table_keys.size - 1
    distinct = 0
    for row in range(token_lo.size):
        lo = token_lo[row]
        hi = token_hi[row]
        # FNV-1a over the token bytes.
        h = np.uint64(14695981039346656037)
        for i in range(lo, hi):
            h ^= np.uint64(buf[i])
            h *= np.uint64(1099511628211)
        slot = np.int64(h & np.uint64(mask))
        while True:
            existing = table_keys[slot]
            if existing == -1:
                if distinct >= distinct_lo.size:
                    return -1
                table_keys[slot] = np.int64(h)
                table_ids[slot] = distinct
                distinct_lo[distinct] = lo
                distinct_hi[distinct] = hi
                token_ids[row] = distinct
                distinct += 1
                break
            if existing == np.int64(h):
                candidate = table_ids[slot]
                clo = distinct_lo[candidate]
                chi = distinct_hi[candidate]
                if chi - clo == hi - lo:
                    same = True
                    for i in range(hi - lo):
                        if buf[lo + i] != buf[clo + i]:
                            same = False
                            break
                    if same:
                        token_ids[row] = candidate
                        break
            slot = (slot + 1) & mask
    return distinct


# ----------------------------------------------------------------------
# Driving the kernels
# ----------------------------------------------------------------------

# Load factor for the interning table, as a power of two above the token count.
_TABLE_HEADROOM = 4


class Columns:
    """A words file as parallel arrays, with its tokens interned."""

    __slots__ = ("buffer", "token_ids", "start_byte", "end_byte", "object_ids",
                 "position_lo", "position_hi", "object_hi", "is_punct",
                 "distinct_lo", "distinct_hi", "distinct_escaped", "n_distinct")

    def __init__(self, **fields):
        for name, value in fields.items():
            setattr(self, name, value)

    def __len__(self):
        return self.token_ids.size

    def token(self, index: int) -> str:
        """The `index`th distinct token, unescaped if it needs to be."""
        raw = self.buffer[self.distinct_lo[index]:self.distinct_hi[index]].tobytes()
        if self.distinct_escaped[index]:
            import json
            return json.loads(b'"' + raw + b'"')
        return raw.decode("utf8")


def _line_ends(buffer: np.ndarray) -> np.ndarray:
    """Start of every line, plus a final sentinel at the end of the buffer."""
    breaks = np.flatnonzero(buffer == NEWLINE)
    ends = np.empty(breaks.size + 2, dtype=np.int64)
    ends[0] = 0
    ends[1:-1] = breaks + 1
    ends[-1] = buffer.size
    # A trailing newline leaves an empty last line; drop it.
    if ends.size >= 2 and ends[-2] >= buffer.size:
        ends = ends[:-1]
    return ends


def _intern(buffer, lo, hi, limit):
    """Dense ids for byte spans. Returns (ids, distinct_lo, distinct_hi, n) or None."""
    size = 1 << max(int(limit * _TABLE_HEADROOM - 1).bit_length(), 4)
    table_keys = np.full(size, -1, dtype=np.int64)
    table_ids = np.zeros(size, dtype=np.int64)
    distinct_lo = np.zeros(limit, dtype=np.int64)
    distinct_hi = np.zeros(limit, dtype=np.int64)
    ids = np.zeros(lo.size, dtype=np.int64)
    found = intern_tokens(buffer, lo, hi, table_keys, table_ids,
                          distinct_lo, distinct_hi, ids)
    if found < 0:
        return None
    return ids, distinct_lo[:found], distinct_hi[:found], found


def read_columns(blob: bytes, level: int, want_punct: bool = False) -> Columns | None:
    """Scan a decompressed words file, or None if the scanner does not recognise it.

    None means some line did not yield the fields we need, which is the signal to
    fall back to a real parser rather than to guess.
    """
    buffer = np.frombuffer(blob, dtype=np.uint8)
    ends = _line_ends(buffer)
    rows = ends.size - 1
    if rows <= 0:
        return None
    token_lo = np.empty(rows, dtype=np.int64)
    token_hi = np.empty(rows, dtype=np.int64)
    start_byte = np.empty(rows, dtype=np.int64)
    end_byte = np.empty(rows, dtype=np.int64)
    position_lo = np.empty(rows, dtype=np.int64)
    position_hi = np.empty(rows, dtype=np.int64)
    object_hi = np.empty(rows, dtype=np.int64)
    token_escaped = np.empty(rows, dtype=np.uint8)
    is_punct = np.empty(rows, dtype=np.uint8)
    scan_lines(buffer, ends, level, want_punct, token_lo, token_hi, token_escaped,
               start_byte, end_byte, position_lo, position_hi, object_hi, is_punct)

    if token_lo[0] < 0 or np.any(token_lo < 0) or np.any(start_byte < 0) or np.any(object_hi < 0):
        return None

    interned = _intern(buffer, token_lo, token_hi, rows)
    if interned is None:
        return None
    token_ids, distinct_lo, distinct_hi, n_distinct = interned

    objects = _intern(buffer, position_lo, object_hi, rows)
    if objects is None:
        return None
    object_ids = objects[0]

    # A token is escaped if any occurrence of it was.
    distinct_escaped = np.zeros(n_distinct, dtype=np.uint8)
    escaped_rows = np.flatnonzero(token_escaped)
    if escaped_rows.size:
        distinct_escaped[token_ids[escaped_rows]] = 1

    return Columns(buffer=buffer, token_ids=token_ids, start_byte=start_byte,
                   end_byte=end_byte, object_ids=object_ids,
                   position_lo=position_lo, position_hi=position_hi,
                   object_hi=object_hi, is_punct=is_punct, distinct_lo=distinct_lo,
                   distinct_hi=distinct_hi, distinct_escaped=distinct_escaped,
                   n_distinct=n_distinct)
