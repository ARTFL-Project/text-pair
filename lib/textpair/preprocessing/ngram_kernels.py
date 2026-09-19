"""Building n-gram keys without building n-gram strings.

On the non-debug path an n-gram's text exists only to be hashed: nothing reads
it, and index.tab is written only when the tracer will. So the key can be
computed straight from the normalized forms, in a kernel, without the hundreds
of thousands of Python string joins a document would otherwise cost.

The hash has to stay exactly what mmh3.hash64(text)[0] gives, since it is the
key on disk and in the aligner's index, so this is MurmurHash3 x64_128 with
seed 0, returning its low 64 bits. tests/test_ngram_kernels.py checks it against
mmh3 itself.
"""

from __future__ import annotations

import numpy as np
from numba import njit

_C1 = np.uint64(0x87C37B91114253D5)
_C2 = np.uint64(0x4CF5AD432745937F)
_M1 = np.uint64(0xFF51AFD7ED558CCD)
_M2 = np.uint64(0xC4CEB9FE1A85EC53)
_UNDERSCORE = 95


@njit(inline="always")
def _rotl(value, bits):
    return (value << np.uint64(bits)) | (value >> np.uint64(64 - bits))


@njit(inline="always")
def _fmix(value):
    value ^= value >> np.uint64(33)
    value *= _M1
    value ^= value >> np.uint64(33)
    value *= _M2
    value ^= value >> np.uint64(33)
    return value


@njit(inline="always")
def murmur3_low64(data, length):
    """Low 64 bits of MurmurHash3 x64_128 over data[:length], seed 0."""
    h1 = np.uint64(0)
    h2 = np.uint64(0)
    blocks = length // 16

    for block in range(blocks):
        base = block * 16
        k1 = np.uint64(0)
        k2 = np.uint64(0)
        for byte in range(8):
            k1 |= np.uint64(data[base + byte]) << np.uint64(8 * byte)
            k2 |= np.uint64(data[base + 8 + byte]) << np.uint64(8 * byte)
        k1 *= _C1
        k1 = _rotl(k1, 31)
        k1 *= _C2
        h1 ^= k1
        h1 = _rotl(h1, 27)
        h1 += h2
        h1 = h1 * np.uint64(5) + np.uint64(0x52DCE729)
        k2 *= _C2
        k2 = _rotl(k2, 33)
        k2 *= _C1
        h2 ^= k2
        h2 = _rotl(h2, 31)
        h2 += h1
        h2 = h2 * np.uint64(5) + np.uint64(0x38495AB5)

    tail = blocks * 16
    remaining = length & 15
    k1 = np.uint64(0)
    k2 = np.uint64(0)
    for byte in range(remaining):
        if byte < 8:
            k1 |= np.uint64(data[tail + byte]) << np.uint64(8 * byte)
        else:
            k2 |= np.uint64(data[tail + byte]) << np.uint64(8 * (byte - 8))
    if remaining > 8:
        k2 *= _C2
        k2 = _rotl(k2, 33)
        k2 *= _C1
        h2 ^= k2
    if remaining > 0:
        k1 *= _C1
        k1 = _rotl(k1, 31)
        k1 *= _C2
        h1 ^= k1

    h1 ^= np.uint64(length)
    h2 ^= np.uint64(length)
    h1 += h2
    h2 += h1
    h1 = _fmix(h1)
    h2 = _fmix(h2)
    h1 += h2
    return h1


@njit(nogil=True, cache=True)
def contiguous_keys(form_bytes, form_offsets, form_ids, size, scratch, out):
    """Keys for every contiguous n-gram of `size` forms, in order.

    The n-gram's bytes are assembled into `scratch` and hashed there; the
    caller sizes it for the longest n-gram the form table can produce.
    """
    total = form_ids.size - size + 1
    if total < 0:
        total = 0
    for start in range(total):
        length = 0
        for offset in range(size):
            if offset:
                scratch[length] = _UNDERSCORE
                length += 1
            form = form_ids[start + offset]
            lo = form_offsets[form]
            hi = form_offsets[form + 1]
            for byte in range(lo, hi):
                scratch[length] = form_bytes[byte]
                length += 1
        out[start] = np.int64(murmur3_low64(scratch, length))
    return total
