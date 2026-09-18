#!/usr/bin/env python3
"""The ngrams_in_order binary format, and that the banality filter still sees what it saw.

    test_ngram_order.py

`ngrams_in_order/{doc}.bin` replaced a JSON array of `[start_byte, key]` pairs, which
banality_finder.NgramDoc used to parse with orjson and bisect. The parse was 4.7ms per
document against nothing measurable for two `np.frombuffer` views, and the file is 12
bytes an n-gram rather than 30 -- but only if the range lookup means exactly what it did
before, boundaries included. That is what this checks, against an independent
reimplementation of the old JSON path.
"""
import bisect
import os
import random
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from textpair.sequence_alignment import ngram_binary  # noqa: E402
from textpair.sequence_alignment.banality_finder import NgramDoc  # noqa: E402


def old_get_ngrams(pairs, start_byte, end_byte):
    """banality_finder.NgramDoc as it was: a list of [start_byte, key] and two bisects."""
    positions = [pair[0] for pair in pairs]
    start_index = bisect.bisect_left(positions, start_byte)
    end_index = bisect.bisect_left(positions, end_byte)
    return [key for _, key in pairs[start_index:end_index]]


def case(name, keys, start_bytes, probes):
    pairs = [[int(sb), int(key)] for sb, key in zip(start_bytes, keys)]
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "0.bin")
        ngram_binary.write_order(path, keys, start_bytes)
        document = NgramDoc(path)
        if document.name != "0.bin":
            raise AssertionError(f"{name}: name is {document.name!r}")
        stored_keys, stored_start_bytes = document.keys, document.start_bytes
        if stored_keys.dtype != np.dtype("<i8"):
            raise AssertionError(f"{name}: keys are {stored_keys.dtype}, not int64")
        if stored_keys.tolist() != [int(k) for k in keys]:
            raise AssertionError(f"{name}: keys did not round-trip")
        if stored_start_bytes.tolist() != [int(s) for s in start_bytes]:
            raise AssertionError(f"{name}: start bytes did not round-trip")
        for start_byte, end_byte in probes:
            expected = old_get_ngrams(pairs, start_byte, end_byte)
            got = document.get_ngrams(start_byte, end_byte)
            if got != expected:
                raise AssertionError(
                    f"{name}: get_ngrams({start_byte}, {end_byte}) gave {got}, "
                    f"expected {expected}"
                )
    print(f"  ok   {name}")


def main():
    random.seed(11)
    # A realistic document: ascending byte offsets, keys spanning the signed int64 range
    # including both extremes, which a 32-bit column could not hold.
    count = 4_000
    start_bytes = np.cumsum(np.random.default_rng(3).integers(1, 40, count)).astype(np.int32)
    keys = np.random.default_rng(4).integers(-2 ** 63, 2 ** 63 - 1, count, dtype=np.int64)
    keys[0] = -(2 ** 63)
    keys[1] = 2 ** 63 - 1
    keys[2] = 0
    probes = [(int(start_bytes[0]), int(start_bytes[-1])),          # the whole document
              (0, int(start_bytes[-1]) + 1000),                     # wider than it
              (int(start_bytes[-1]) + 1, int(start_bytes[-1]) + 2), # past the end
              (0, 0), (5, 5)]                                       # empty ranges
    for _ in range(300):
        lo = random.randrange(0, int(start_bytes[-1]) + 10)
        hi = lo + random.randrange(0, 400)
        probes.append((lo, hi))
    # Every exact boundary, where bisect_left's tie behaviour is the thing to preserve.
    for index in random.sample(range(count), 40):
        probes.append((int(start_bytes[index]), int(start_bytes[min(index + 5, count - 1)])))
    case("realistic document", keys, start_bytes, probes)

    case("single n-gram", np.array([42], np.int64), np.array([7], np.int32),
         [(0, 7), (7, 8), (0, 100), (8, 9)])
    case("repeated start bytes", np.array([1, 2, 3, 4], np.int64),
         np.array([10, 10, 20, 20], np.int32),
         [(10, 20), (0, 10), (10, 10), (20, 21), (0, 100)])
    case("empty document", np.array([], np.int64), np.array([], np.int32),
         [(0, 10), (0, 0)])

    blob = ngram_binary.dumps_order(np.array([1, 2], np.int64), np.array([0, 5], np.int32))
    if len(blob) != ngram_binary.ORDER_HEADER_SIZE + 12 * 2:
        raise AssertionError(f"unexpected size {len(blob)}")
    for damaged, why in ((blob[:8] + blob[9:], "truncated"),
                         (b"TPNG0002" + blob[8:], "wrong magic")):
        try:
            ngram_binary.order_columns(damaged, "damaged")
        except ValueError:
            pass
        else:
            raise AssertionError(f"{why} input was accepted")
    print("  ok   size and rejection of damaged input")
    print("all checks pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
