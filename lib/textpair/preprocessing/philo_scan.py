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
               position_lo, position_hi, object_hi, is_punct, new_object):
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
        new_object[row] = 1
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

        # Whether this row opens a new text object, by comparing its object id
        # with the previous row's. Interning them instead would mean hashing
        # every token's position to discover, in a document-level corpus, that
        # there is exactly one.
        if row > 0 and object_hi[row] >= 0 and object_hi[row - 1] >= 0:
            length = object_hi[row] - position_lo[row]
            if length == object_hi[row - 1] - position_lo[row - 1]:
                same = True
                for k in range(length):
                    if buf[position_lo[row] + k] != buf[position_lo[row - 1] + k]:
                        same = False
                        break
                if same:
                    new_object[row] = 0


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

    __slots__ = ("buffer", "token_ids", "start_byte", "end_byte", "new_object",
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


@njit(nogil=True, cache=True)
def _line_ends(buffer):
    """Start of every line, plus a final sentinel at the end of the buffer.

    Counting first and filling second walks the buffer twice, which is still
    less work than `flatnonzero(buffer == NEWLINE)`: that materialises a bool
    the size of the file -- 12MB a document, 45.7GB over a corpus -- and then
    scans it.
    """
    breaks = 0
    for i in range(buffer.size):
        if buffer[i] == NEWLINE:
            breaks += 1
    ends = np.empty(breaks + 2, dtype=np.int64)
    ends[0] = 0
    at = 1
    for i in range(buffer.size):
        if buffer[i] == NEWLINE:
            ends[at] = i + 1
            at += 1
    ends[at] = buffer.size
    # A trailing newline leaves an empty last line; drop it.
    if at >= 1 and ends[at - 1] >= buffer.size:
        return ends[:at]
    return ends[:at + 1]


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


def read_columns(blob: bytes, level: int, want_punct: bool = False,
                 vocabulary: "Vocabulary | None" = None) -> Columns | None:
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
    new_object = np.empty(rows, dtype=np.uint8)
    scan_lines(buffer, ends, level, want_punct, token_lo, token_hi, token_escaped,
               start_byte, end_byte, position_lo, position_hi, object_hi, is_punct,
               new_object)

    if token_lo[0] < 0 or np.any(token_lo < 0) or np.any(start_byte < 0) or np.any(object_hi < 0):
        return None

    if vocabulary is not None:
        token_ids = vocabulary.intern(buffer, token_lo, token_hi)
        distinct_lo = distinct_hi = np.zeros(0, dtype=np.int64)
        n_distinct = vocabulary.count
    else:
        interned = _intern(buffer, token_lo, token_hi, rows)
        if interned is None:
            return None
        token_ids, distinct_lo, distinct_hi, n_distinct = interned

    # A token is escaped if any occurrence of it was. The vocabulary decides
    # this from the bytes themselves, so it only matters for the local table.
    if vocabulary is None:
        distinct_escaped = np.zeros(n_distinct, dtype=np.uint8)
        escaped_rows = np.flatnonzero(token_escaped)
        if escaped_rows.size:
            distinct_escaped[token_ids[escaped_rows]] = 1
    else:
        distinct_escaped = np.zeros(0, dtype=np.uint8)

    return Columns(buffer=buffer, token_ids=token_ids, start_byte=start_byte,
                   end_byte=end_byte, new_object=new_object,
                   position_lo=position_lo, position_hi=position_hi,
                   object_hi=object_hi, is_punct=is_punct, distinct_lo=distinct_lo,
                   distinct_hi=distinct_hi, distinct_escaped=distinct_escaped,
                   n_distinct=n_distinct)


# ----------------------------------------------------------------------
# A worker's token vocabulary
# ----------------------------------------------------------------------


@njit(inline="always")
def _fnv(buf, lo, hi):
    h = np.uint64(14695981039346656037)
    for i in range(lo, hi):
        h ^= np.uint64(buf[i])
        h *= np.uint64(1099511628211)
    return h


@njit(nogil=True, cache=True)
def intern_into(buf, token_lo, token_hi, first_row, vocab_data, vocab_offsets,
                table_hash, table_id, out_ids, state):
    """Map token spans onto ids in a vocabulary that outlives this buffer.

    Resumable: when the vocabulary runs out of room it records its progress in
    `state` and returns the row it stopped at, so the caller can grow the arrays
    and call again from there without losing what was already interned.
    `state` is (count, bytes used).
    """
    count = state[0]
    used = state[1]
    mask = np.uint64(table_hash.size - 1)
    for row in range(first_row, token_lo.size):
        lo = token_lo[row]
        hi = token_hi[row]
        length = hi - lo
        digest = _fnv(buf, lo, hi)
        slot = digest & mask
        while True:
            at = table_id[slot]
            if at == -1:
                if used + length > vocab_data.size or count + 1 >= vocab_offsets.size:
                    state[0] = count
                    state[1] = used
                    return row
                for i in range(length):
                    vocab_data[used + i] = buf[lo + i]
                used += length
                table_hash[slot] = np.int64(digest)
                table_id[slot] = count
                out_ids[row] = count
                count += 1
                vocab_offsets[count] = used
                break
            if table_hash[slot] == np.int64(digest):
                start = vocab_offsets[at]
                if vocab_offsets[at + 1] - start == length:
                    same = True
                    for i in range(length):
                        if vocab_data[start + i] != buf[lo + i]:
                            same = False
                            break
                    if same:
                        out_ids[row] = at
                        break
            slot = (slot + np.uint64(1)) & mask
    state[0] = count
    state[1] = used
    return token_lo.size


@njit(nogil=True, cache=True)
def rehash_vocabulary(vocab_data, vocab_offsets, count, table_hash, table_id):
    """Refill a freshly enlarged table from the vocabulary."""
    mask = np.uint64(table_hash.size - 1)
    for entry in range(count):
        lo = vocab_offsets[entry]
        hi = vocab_offsets[entry + 1]
        digest = _fnv(vocab_data, lo, hi)
        slot = digest & mask
        while table_id[slot] != -1:
            slot = (slot + np.uint64(1)) & mask
        table_hash[slot] = np.int64(digest)
        table_id[slot] = entry


class Vocabulary:
    """Every distinct token a worker has seen, and its normalized form.

    Interning per file means materializing a Python string for each of a
    document's distinct tokens just to look up a form the worker already
    computed -- about 6,400 per document, 23M over a corpus, for maybe 2M
    genuinely distinct tokens. Carrying the table between files means a token is
    turned into a string, modernized and normalized once, the first time this
    worker meets it.

    The normalized forms are kept twice: as Python strings, which text objects
    hand to their consumers, and as the packed bytes the n-gram kernel hashes.
    """

    __slots__ = ("data", "offsets", "table_hash", "table_id", "state",
                 "forms", "form_data", "form_used", "form_offsets", "longest", "kept")

    def __init__(self):
        self.data = np.empty(1 << 20, dtype=np.uint8)
        self.offsets = np.zeros((1 << 16) + 1, dtype=np.int64)
        self.table_hash = np.zeros(1 << 17, dtype=np.int64)
        self.table_id = np.full(1 << 17, -1, dtype=np.int64)
        self.state = np.zeros(2, dtype=np.int64)
        self.forms: list[str] = []
        self.form_data = np.empty(1 << 20, dtype=np.uint8)
        self.form_used = 0
        self.form_offsets = np.zeros((1 << 16) + 1, dtype=np.int64)
        self.longest = 0
        # Whether each form is a token at all. Matches TextObject.purge: a form
        # of one space survives normalization but is not a token.
        self.kept = np.zeros(1 << 16, dtype=bool)

    @property
    def count(self) -> int:
        return int(self.state[0])

    def intern(self, buffer: np.ndarray, token_lo: np.ndarray,
               token_hi: np.ndarray) -> np.ndarray:
        """Ids for every token span, extending the vocabulary as needed."""
        out = np.empty(token_lo.size, dtype=np.int64)
        row = 0
        while True:
            row = intern_into(buffer, token_lo, token_hi, row, self.data, self.offsets,
                              self.table_hash, self.table_id, out, self.state)
            self._maybe_grow()
            if row >= token_lo.size:
                return out

    def _maybe_grow(self) -> None:
        count = self.count
        used = int(self.state[1])
        if count + 1 >= self.offsets.size:
            self.offsets = np.resize(self.offsets, self.offsets.size * 2)
            self.offsets[count + 1 :] = 0
        # Doubling the data when it is close to full, rather than exactly full,
        # keeps a single long token from forcing a resize per call.
        if used * 4 > self.data.size * 3:
            grown = np.empty(self.data.size * 2, dtype=np.uint8)
            grown[:used] = self.data[:used]
            self.data = grown
        if count * 2 > self.table_hash.size:
            size = self.table_hash.size * 2
            self.table_hash = np.zeros(size, dtype=np.int64)
            self.table_id = np.full(size, -1, dtype=np.int64)
            rehash_vocabulary(self.data, self.offsets, count, self.table_hash, self.table_id)

    def resolve(self, normalize) -> None:
        """Normalize every token interned since the last call."""
        data = self.data
        offsets = self.offsets
        for entry in range(len(self.forms), self.count):
            raw = data[offsets[entry] : offsets[entry + 1]].tobytes()
            if b"\\" in raw:
                import json

                token = json.loads(b'"' + raw + b'"')
            else:
                token = raw.decode("utf8")
            form = normalize(token)
            self.forms.append(form)
            encoded = form.encode("utf8")
            if self.form_used + len(encoded) > self.form_data.size:
                grown = np.empty(max(self.form_data.size * 2,
                                     self.form_used + len(encoded)), dtype=np.uint8)
                grown[: self.form_used] = self.form_data[: self.form_used]
                self.form_data = grown
            if entry + 1 >= self.form_offsets.size:
                self.form_offsets = np.resize(self.form_offsets, self.form_offsets.size * 2)
            if entry >= self.kept.size:
                self.kept = np.resize(self.kept, self.kept.size * 2)
            self.kept[entry] = bool(form) and form != " "
            if encoded:
                self.form_data[self.form_used : self.form_used + len(encoded)] = \
                    np.frombuffer(encoded, dtype=np.uint8)
            self.form_used += len(encoded)
            self.form_offsets[entry + 1] = self.form_used
            if len(encoded) > self.longest:
                self.longest = len(encoded)
