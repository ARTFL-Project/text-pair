#!/usr/bin/env python3
"""Binary columnar per-document files: the n-gram index, and n-grams in order.

Both replace a JSON file so a reader can mmap instead of parse. Little-endian
throughout.

`ngrams/{doc_id}.bin` -- the aligner's index, one CSR block per key:

    magic    8 bytes            b"TPNG0002"
    n_keys   int64
    n_pos    int64
    keys     int64[n_keys]      ngram hashes, ascending
    offsets  int32[n_keys + 1]  CSR bounds into the position columns, offsets[0] == 0
    idx      int32[n_pos]       ngram index within the document
    sb       int32[n_pos]       start byte
    eb       int32[n_pos]       end byte

`ngrams_in_order/{doc_id}.bin` -- every n-gram in document order, which the banality
filter reads by byte range:

    magic    8 bytes            b"TPIO0001"
    n        int64
    keys     int64[n]           ngram hashes, in ngram-index order
    sb       int32[n]           start byte, ascending

Headers are 24 and 16 bytes, both multiples of 8, and the int64 keys column comes first
in each, so every column starts on its own alignment in a page-aligned mmap and
`np.frombuffer` views it without a copy. No padding is needed and none is written.

Keys are 64 bits. `TPNG0001`, which held them in int32, still loads -- `columns` returns
that file's keys in their own width and the loader widens them on assignment -- but an
index written under either width can only be aligned against another built with the same
hash, since a 32-bit key and a 64-bit key of the same n-gram are unrelated values.

Ordering is load-bearing, since the point of the format is to let the loader skip its
sort. `keys` is ascending as a *signed* int64, which is what `loader.build_csr_sorted`
produces from JSON (np.argsort over the hash) and also the order the aligner's biased
`uint64(key) ^ 0x8000000000000000` comparison gives, that bias being monotone in the
signed key. Positions inside a key stay in increasing ngram-index order, which the stable
sort in `_csr` preserves from the ngram order they are collected in.

`convert_file` / `convert_directory` turn an existing JSON index into binary; they
exist for the validation tools under tests/, not as a supported migration path.
"""

import os
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import orjson

MAGIC = b"TPNG0002"
LEGACY_MAGIC = b"TPNG0001"          # same layout, int32 keys
HEADER_SIZE = 24
ORDER_MAGIC = b"TPIO0001"
ORDER_HEADER_SIZE = 16
_I4 = np.dtype("<i4")
_I8 = np.dtype("<i8")
_HEADER = struct.Struct("<8sqq")
_ORDER_HEADER = struct.Struct("<8sq")


def _pack(keys, offsets, idx, sb, eb):
    out = bytearray(_HEADER.pack(MAGIC, keys.shape[0], idx.shape[0]))
    for column in (keys, offsets, idx, sb, eb):
        out += column.tobytes()
    return bytes(out)


def _csr(hashes, idx, sb, eb):
    """Group one entry per position by its hash into the five columns.

    The sort is stable, so each key keeps its positions in the order they arrive.
    """
    keys = np.asarray(hashes, _I8)
    n = keys.shape[0]
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    first = np.ones(n, bool)
    first[1:] = ordered[1:] != ordered[:-1]
    starts = np.flatnonzero(first)
    offsets = np.empty(starts.shape[0] + 1, _I4)
    offsets[:-1] = starts
    offsets[-1] = n
    return (ordered[starts], offsets, np.asarray(idx, _I4)[order],
            np.asarray(sb, _I4)[order], np.asarray(eb, _I4)[order])


def dumps_positions(hashes, start_bytes, end_bytes):
    """Serialize a document whose positions are given in ngram-index order, so that
    the ngram index of position i is i."""
    return _pack(*_csr(hashes, np.arange(len(hashes), dtype=_I4), start_bytes, end_bytes))


def write_positions(path, hashes, start_bytes, end_bytes):
    """Write a document's positions, in ngram-index order, to `path`."""
    with open(path, "wb") as binary_file:
        binary_file.write(dumps_positions(hashes, start_bytes, end_bytes))


def dumps(text_index):
    """Serialize {int hash: [(index, start_byte, end_byte), ...]}."""
    hashes = []
    idx = []
    sb = []
    eb = []
    for key, positions in text_index.items():
        for index, start_byte, end_byte in positions:
            hashes.append(key)
            idx.append(index)
            sb.append(start_byte)
            eb.append(end_byte)
    return _pack(*_csr(hashes, idx, sb, eb))


def parse_header(head, path=""):
    """(n_keys, n_pos) from the first HEADER_SIZE bytes. Raises on a bad magic."""
    return _parse_header(head, path)[:2]


def _parse_header(head, path=""):
    """(n_keys, n_pos, key_dtype). Accepts both key widths."""
    if len(head) < HEADER_SIZE:
        raise ValueError(f"truncated binary ngram file: {path or '<buffer>'}")
    magic, n_keys, n_pos = _HEADER.unpack_from(head, 0)
    if magic == MAGIC:
        key_dtype = _I8
    elif magic == LEGACY_MAGIC:
        key_dtype = _I4
    else:
        raise ValueError(
            f"{path or '<buffer>'} is not a binary ngram file: "
            f"expected magic {MAGIC!r}, found {magic!r}"
        )
    if n_keys < 0 or n_pos < 0:
        raise ValueError(f"negative array length in binary ngram file: {path or '<buffer>'}")
    return n_keys, n_pos, key_dtype


def read_header(path):
    """(n_keys, n_pos) for one file, reading only the header."""
    with open(path, "rb") as binary_file:
        return parse_header(binary_file.read(HEADER_SIZE), path)


def columns(buffer, path=""):
    """(keys, offsets, idx, sb, eb) as zero-copy views into an mmap or bytes buffer.

    `keys` comes back in the file's own width, int64 or -- for a TPNG0001 index -- int32.
    Callers that concatenate it into an int64 array get the widening from numpy on
    assignment; callers that write it back out must coerce it themselves.
    """
    n_keys, n_pos, key_dtype = _parse_header(buffer[:HEADER_SIZE], path)
    expected = (HEADER_SIZE + key_dtype.itemsize * n_keys
                + 4 * (n_keys + 1 + 3 * n_pos))
    if len(buffer) < expected:
        raise ValueError(
            f"truncated binary ngram file {path or '<buffer>'}: "
            f"{len(buffer)} bytes, expected {expected}"
        )
    keys = np.frombuffer(buffer, key_dtype, n_keys, HEADER_SIZE)
    offset = HEADER_SIZE + key_dtype.itemsize * n_keys
    out = [keys]
    for count in (n_keys + 1, n_pos, n_pos, n_pos):
        out.append(np.frombuffer(buffer, _I4, count, offset))
        offset += 4 * count
    return tuple(out)


# ------------------------------------------------------------------ n-grams in order


def dumps_order(hashes, start_bytes):
    """Serialize a document's n-grams in ngram-index order, for the banality filter."""
    keys = np.asarray(hashes, _I8)
    return (_ORDER_HEADER.pack(ORDER_MAGIC, keys.shape[0]) + keys.tobytes()
            + np.asarray(start_bytes, _I4).tobytes())


def write_order(path, hashes, start_bytes):
    """Write one document's n-grams in order to `path`."""
    with open(path, "wb") as binary_file:
        binary_file.write(dumps_order(hashes, start_bytes))


def order_columns(buffer, path=""):
    """(keys, start_bytes) as zero-copy views into an mmap or bytes buffer."""
    if len(buffer) < ORDER_HEADER_SIZE:
        raise ValueError(f"truncated ngrams-in-order file: {path or '<buffer>'}")
    magic, count = _ORDER_HEADER.unpack_from(buffer, 0)
    if magic != ORDER_MAGIC:
        raise ValueError(
            f"{path or '<buffer>'} is not an ngrams-in-order file: "
            f"expected magic {ORDER_MAGIC!r}, found {magic!r}"
        )
    if count < 0:
        raise ValueError(f"negative length in ngrams-in-order file: {path or '<buffer>'}")
    expected = ORDER_HEADER_SIZE + 12 * count
    if len(buffer) < expected:
        raise ValueError(
            f"truncated ngrams-in-order file {path or '<buffer>'}: "
            f"{len(buffer)} bytes, expected {expected}"
        )
    keys = np.frombuffer(buffer, _I8, count, ORDER_HEADER_SIZE)
    start_bytes = np.frombuffer(buffer, _I4, count, ORDER_HEADER_SIZE + 8 * count)
    return keys, start_bytes


def convert_file(json_path, binary_path):
    """Convert one JSON ngram file. Returns the two file sizes."""
    with open(json_path, "rb") as json_file:
        raw = json_file.read()
    payload = dumps({int(key): value for key, value in orjson.loads(raw).items()})
    with open(binary_path, "wb") as binary_file:
        binary_file.write(payload)
    return len(raw), len(payload)


def convert_directory(source_dir, target_dir, workers=8):
    """Convert every .json file in `source_dir` into `target_dir`, which must not be
    `source_dir`. Returns (n_files, json_bytes, binary_bytes)."""
    if os.path.abspath(source_dir) == os.path.abspath(target_dir):
        raise ValueError("refusing to convert an ngram directory in place")
    names = sorted(name for name in os.listdir(source_dir) if name.endswith(".json"))
    if not names:
        raise ValueError(f"no .json ngram files in {source_dir}")
    os.makedirs(target_dir, exist_ok=True)
    totals = [0, 0]

    def convert(name):
        return convert_file(
            os.path.join(source_dir, name),
            os.path.join(target_dir, name[: -len(".json")] + ".bin"),
        )

    with ThreadPoolExecutor(workers) as pool:
        for json_bytes, binary_bytes in pool.map(convert, names):
            totals[0] += json_bytes
            totals[1] += binary_bytes
    return len(names), totals[0], totals[1]
