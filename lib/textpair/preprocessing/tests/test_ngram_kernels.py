#!/usr/bin/env python3
"""Checks the n-gram key kernel against mmh3 and against joining the strings.

    test_ngram_kernels.py

The kernel exists so the non-debug path never builds an n-gram string. That is
only allowed if the key is bit-for-bit what hashing the string would have given,
because it is the key on disk and in the aligner's inverted index.
"""
import random
import sys

import numpy as np
from mmh3 import hash64

from textpair.preprocessing.ngram_kernels import contiguous_keys, murmur3_low64

FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


def hash_of(text: str) -> int:
    """The kernel's key for `text`, as the signed int64 mmh3 would return."""
    data = np.frombuffer(text.encode("utf8"), dtype=np.uint8)
    padded = np.zeros(max(data.size, 1) + 16, dtype=np.uint8)
    padded[: data.size] = data
    unsigned = murmur3_low64(padded, data.size)
    return int(np.array(unsigned, dtype=np.uint64).view(np.int64))


def test_lengths_around_the_block_boundary():
    """x64_128 consumes 16 bytes at a time; every tail length must be right."""
    for length in range(0, 80):
        text = "".join(chr(97 + (i % 26)) for i in range(length))
        check(f"len {length}", hash_of(text), hash64(text)[0])


def test_multibyte_and_punctuation():
    for text in ("été", "œuvre", "français_très_élégant", "a_b_c", "_", "__",
                 "  _idem_avec", "ῥυθμὸς_peut_se", "日本語_テスト_文字"):
        check(f"{text!r}", hash_of(text), hash64(text)[0])


def test_random_strings():
    random.seed(4)
    alphabet = "abcdefghijklmnopqrstuvwxyzéèàçôûîœ_"
    for _ in range(4000):
        text = "".join(random.choice(alphabet) for _ in range(random.randint(1, 40)))
        if hash_of(text) != hash64(text)[0]:
            FAILURES.append(f"random {text!r}: {hash_of(text)} != {hash64(text)[0]}")
            return


def _table(forms):
    encoded = [form.encode("utf8") for form in forms]
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(e) for e in encoded])
    return np.frombuffer(b"".join(encoded), dtype=np.uint8), offsets


def test_matches_joining_the_strings():
    random.seed(9)
    forms = ["maison", "cheval", "é", "trèslongmotquidépasse", "a", "royaume", "œuvre"]
    form_bytes, offsets = _table(forms)
    for size in (2, 3, 5):
        ids = np.array([random.randrange(len(forms)) for _ in range(200)], dtype=np.int64)
        longest = int(np.max(np.diff(offsets))) * size + size
        scratch = np.zeros(longest + 16, dtype=np.uint8)
        out = np.zeros(max(ids.size - size + 1, 0), dtype=np.int64)
        contiguous_keys(form_bytes, offsets, ids, size, scratch, out)
        want = [hash64("_".join(forms[i] for i in ids[start:start + size]))[0]
                for start in range(ids.size - size + 1)]
        check(f"size {size}", out.tolist(), want)


def test_empty_and_short_inputs():
    form_bytes, offsets = _table(["a", "bb"])
    scratch = np.zeros(64, dtype=np.uint8)
    for count in (0, 1, 2):
        ids = np.zeros(count, dtype=np.int64)
        out = np.zeros(max(count - 3 + 1, 0), dtype=np.int64)
        produced = contiguous_keys(form_bytes, offsets, ids, 3, scratch, out)
        check(f"{count} forms, 3-grams", produced, max(count - 2, 0))


def main():
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    if FAILURES:
        print(f"test_ngram_kernels: {len(FAILURES)} failure(s)")
        for failure in FAILURES[:10]:
            print(f"  {failure}")
        return 1
    print("test_ngram_kernels: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
