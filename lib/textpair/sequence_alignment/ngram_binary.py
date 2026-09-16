#!/usr/bin/env python3
"""Binary columnar ngram index: one `ngrams/{doc_id}.bin` per document.

Replaces `ngrams/{doc_id}.json` so the aligner's load phase is an mmap instead of a
parse. Little-endian throughout:

    magic    8 bytes            b"TPNG0001"
    n_keys   int64
    n_pos    int64
    keys     int32[n_keys]      ngram hashes, ascending
    offsets  int32[n_keys + 1]  CSR bounds into the position columns, offsets[0] == 0
    idx      int32[n_pos]       ngram index within the document
    sb       int32[n_pos]       start byte
    eb       int32[n_pos]       end byte

The header is 24 bytes, a multiple of 4, so every column starts 4-byte aligned in a
page-aligned mmap and `np.frombuffer` views it without a copy. No padding is needed,
and none is written: the file is exactly HEADER_SIZE + 4 * (2 * n_keys + 1 + 3 * n_pos)
bytes.

Ordering is load-bearing, since the point of the format is to let the loader skip its
sort. `keys` is ascending as a *signed* int32, which is what `loader.build_csr_sorted`
produces from JSON (np.argsort over the int32 hash) and also the order the aligner's
biased `uint32(key) ^ 0x80000000` comparison gives, that bias being monotone in the
signed key. Positions inside a key stay in increasing ngram-index order, which is the
order the JSON writer's dict insertion leaves them in.

`convert_file` / `convert_directory` turn an existing JSON index into binary; they
exist for the validation tools under tests/, not as a supported migration path.
"""

import os
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import orjson

MAGIC = b"TPNG0001"
HEADER_SIZE = 24
_I4 = np.dtype("<i4")
_HEADER = struct.Struct("<8sqq")


def columns_from_index(text_index):
    """(keys, offsets, idx, sb, eb) from {int hash: [(index, start_byte, end_byte), ...]}."""
    keys = sorted(text_index)
    offsets = np.empty(len(keys) + 1, _I4)
    offsets[0] = 0
    idx = []
    sb = []
    eb = []
    for slot, key in enumerate(keys):
        positions = text_index[key]
        for position in positions:
            idx.append(position[0])
            sb.append(position[1])
            eb.append(position[2])
        offsets[slot + 1] = len(idx)
    return (
        np.fromiter(keys, _I4, len(keys)),
        offsets,
        np.array(idx, _I4),
        np.array(sb, _I4),
        np.array(eb, _I4),
    )


def dumps(text_index):
    """Serialize an ngram index to the binary format."""
    keys, offsets, idx, sb, eb = columns_from_index(text_index)
    out = bytearray(_HEADER.pack(MAGIC, keys.shape[0], idx.shape[0]))
    for column in (keys, offsets, idx, sb, eb):
        out += column.tobytes()
    return bytes(out)


def write(path, text_index):
    """Write an ngram index to `path` in the binary format."""
    with open(path, "wb") as binary_file:
        binary_file.write(dumps(text_index))


def parse_header(head, path=""):
    """(n_keys, n_pos) from the first HEADER_SIZE bytes. Raises on a bad magic."""
    if len(head) < HEADER_SIZE:
        raise ValueError(f"truncated binary ngram file: {path or '<buffer>'}")
    magic, n_keys, n_pos = _HEADER.unpack_from(head, 0)
    if magic != MAGIC:
        raise ValueError(
            f"{path or '<buffer>'} is not a binary ngram file: "
            f"expected magic {MAGIC!r}, found {magic!r}"
        )
    if n_keys < 0 or n_pos < 0:
        raise ValueError(f"negative array length in binary ngram file: {path or '<buffer>'}")
    return n_keys, n_pos


def read_header(path):
    """(n_keys, n_pos) for one file, reading only the header."""
    with open(path, "rb") as binary_file:
        return parse_header(binary_file.read(HEADER_SIZE), path)


def columns(buffer, path=""):
    """(keys, offsets, idx, sb, eb) as zero-copy views into an mmap or bytes buffer."""
    n_keys, n_pos = parse_header(buffer[:HEADER_SIZE], path)
    expected = HEADER_SIZE + 4 * (2 * n_keys + 1 + 3 * n_pos)
    if len(buffer) < expected:
        raise ValueError(
            f"truncated binary ngram file {path or '<buffer>'}: "
            f"{len(buffer)} bytes, expected {expected}"
        )
    offset = HEADER_SIZE
    out = []
    for count in (n_keys, n_keys + 1, n_pos, n_pos, n_pos):
        out.append(np.frombuffer(buffer, _I4, count, offset))
        offset += 4 * count
    return tuple(out)


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
