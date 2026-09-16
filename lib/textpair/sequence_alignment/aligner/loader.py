"""Ngram loading for the sequence aligner.

A document is either JSON, {"<int32 hash>": [[index, start_byte, end_byte], ...], ...},
or the binary columnar format of ../ngram_binary.py; the format is picked per file by
extension and confirmed by the magic bytes, so a half-converted corpus loads. JSON is
parsed and sorted here, tolerating insignificant whitespace so files from older,
non-compact writers still parse; binary is already in the loader's own order and is
mmapped straight into place.

`load_corpus` builds a whole corpus's shared columnar arrays: per-document sorted keys
plus CSR position blocks, with positions inside a key left in file order as Go's append
leaves them (main.go:320-326).
"""
import mmap
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

from .. import ngram_binary


@njit(nogil=True, cache=True)
def count_keys_pos(buf):
    """(n_keys, n_positions) without building anything: ':' counts keys and '[' counts
    keys + positions, the same counting rule as parse_ngram_json."""
    nk = 0
    nb = 0
    for i in range(buf.shape[0]):
        b = buf[i]
        if b == 58:
            nk += 1
        elif b == 91:
            nb += 1
    return nk, nb - nk


@njit(nogil=True, cache=True)
def _skip_ws(buf, i):
    while True:
        b = buf[i]
        if b == 32 or b == 10 or b == 13 or b == 9:
            i += 1
        else:
            return i


@njit(nogil=True, cache=True)
def _parse_int(buf, i):
    neg = False
    if buf[i] == 45:  # '-'
        neg = True
        i += 1
    v = 0
    while True:
        b = buf[i]
        if b >= 48 and b <= 57:
            v = v * 10 + (b - 48)
            i += 1
        else:
            break
    if neg:
        v = -v
    return v, i


@njit(nogil=True, cache=True)
def parse_ngram_json(buf):
    """Returns (ok, keys, counts, idx, sb, eb) in file order. ok=0 on malformed input."""
    n = buf.shape[0]
    n_keys = 0
    n_br = 0
    for i in range(n):
        b = buf[i]
        if b == 58:      # ':'
            n_keys += 1
        elif b == 91:    # '['
            n_br += 1
    n_pos = n_br - n_keys
    keys = np.empty(n_keys, np.int32)
    counts = np.empty(n_keys, np.int32)
    idx = np.empty(n_pos, np.int32)
    sb = np.empty(n_pos, np.int32)
    eb = np.empty(n_pos, np.int32)
    i = _skip_ws(buf, 0)
    if buf[i] != 123:  # '{'
        return 0, keys, counts, idx, sb, eb
    i += 1
    k = 0
    p = 0
    while True:
        i = _skip_ws(buf, i)
        if buf[i] == 125:  # '}'
            break
        if buf[i] != 34:   # '"'
            return 0, keys, counts, idx, sb, eb
        i += 1
        v, i = _parse_int(buf, i)
        if buf[i] != 34:
            return 0, keys, counts, idx, sb, eb
        i += 1
        keys[k] = v
        i = _skip_ws(buf, i)
        if buf[i] != 58:
            return 0, keys, counts, idx, sb, eb
        i = _skip_ws(buf, i + 1)
        if buf[i] != 91:
            return 0, keys, counts, idx, sb, eb
        i += 1
        cnt = 0
        while True:
            i = _skip_ws(buf, i)
            b = buf[i]
            if b == 93:      # ']'
                i += 1
                break
            if b == 44:      # ','
                i += 1
                continue
            if b != 91:
                return 0, keys, counts, idx, sb, eb
            i = _skip_ws(buf, i + 1)
            v0, i = _parse_int(buf, i)
            i = _skip_ws(buf, i)
            if buf[i] != 44:
                return 0, keys, counts, idx, sb, eb
            i = _skip_ws(buf, i + 1)
            v1, i = _parse_int(buf, i)
            i = _skip_ws(buf, i)
            if buf[i] != 44:
                return 0, keys, counts, idx, sb, eb
            i = _skip_ws(buf, i + 1)
            v2, i = _parse_int(buf, i)
            i = _skip_ws(buf, i)
            if buf[i] != 93:
                return 0, keys, counts, idx, sb, eb
            i += 1
            idx[p] = v0; sb[p] = v1; eb[p] = v2
            p += 1
            cnt += 1
        counts[k] = cnt
        k += 1
        i = _skip_ws(buf, i)
        if buf[i] == 44:
            i += 1
    if k != n_keys or p != n_pos:
        return 0, keys, counts, idx, sb, eb
    return 1, keys, counts, idx, sb, eb


@njit(nogil=True, cache=True)
def build_csr_sorted(keys, counts, idx, sb, eb):
    """Sort keys ascending; reorder CSR blocks accordingly, keeping within-key file order."""
    n_keys = keys.shape[0]
    n_pos = idx.shape[0]
    off_in = np.empty(n_keys + 1, np.int32)
    off_in[0] = 0
    for k in range(n_keys):
        off_in[k + 1] = off_in[k] + counts[k]
    order = np.argsort(keys)
    skeys = np.empty(n_keys, np.int32)
    off = np.empty(n_keys + 1, np.int32)
    sidx = np.empty(n_pos, np.int32)
    ssb = np.empty(n_pos, np.int32)
    seb = np.empty(n_pos, np.int32)
    off[0] = 0
    p = 0
    for k in range(n_keys):
        o = order[k]
        skeys[k] = keys[o]
        for q in range(off_in[o], off_in[o + 1]):
            sidx[p] = idx[q]; ssb[p] = sb[q]; seb[p] = eb[q]
            p += 1
        off[k + 1] = p
    return skeys, off, sidx, ssb, seb


def is_binary(path):
    return path.endswith(".bin")


def load_corpus(paths, threads):
    """Parse `paths` (one ngram file per document, in SortID order) into global arrays.

    Two passes so the arrays are allocated once and filled in place: a sizing pass,
    then the parse. Sizing scans a JSON file's bytes but only reads a binary file's
    header. Returns (key_off, keys_all, off_all, idx_all, sb_all, eb_all):

      keys_all  int32[T]    per-document sorted keys, concatenated
      key_off   int64[N+1]  document -> its first slot in keys_all
      off_all   int32[T+N]  document d's CSR offsets at [key_off[d]+d .. key_off[d+1]+d],
                            already shifted to global positions in idx_all
      idx_all / sb_all / eb_all  int32[P]  ngram index, start byte, end byte
    """
    n = len(paths)
    nk = np.zeros(n, np.int64)
    npos = np.zeros(n, np.int64)

    def count(i):
        if is_binary(paths[i]):
            nk[i], npos[i] = ngram_binary.read_header(paths[i])
        else:
            nk[i], npos[i] = count_keys_pos(np.fromfile(paths[i], dtype=np.uint8))

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(count, range(n)))

    key_off = np.zeros(n + 1, np.int64)
    np.cumsum(nk, out=key_off[1:])
    pos_base = np.zeros(n + 1, np.int64)
    np.cumsum(npos, out=pos_base[1:])
    n_keys = int(key_off[-1])
    n_pos = int(pos_base[-1])
    if n_pos >= 2 ** 31:
        raise RuntimeError(f"{n_pos} ngram positions exceeds the int32 CSR offsets")
    keys_all = np.empty(n_keys, np.int32)
    off_all = np.empty(n_keys + n, np.int32)
    idx_all = np.empty(n_pos, np.int32)
    sb_all = np.empty(n_pos, np.int32)
    eb_all = np.empty(n_pos, np.int32)

    def store(i, skeys, off, sidx, ssb, seb):
        a, b = key_off[i], key_off[i + 1]
        keys_all[a:b] = skeys
        off_all[a + i: b + i + 1] = off.astype(np.int32) + np.int32(pos_base[i])
        p, q = pos_base[i], pos_base[i + 1]
        idx_all[p:q] = sidx
        sb_all[p:q] = ssb
        eb_all[p:q] = seb

    def parse(i):
        if is_binary(paths[i]):
            with open(paths[i], "rb") as ngram_file:
                with mmap.mmap(ngram_file.fileno(), 0, access=mmap.ACCESS_READ) as view:
                    store(i, *ngram_binary.columns(view, paths[i]))
            return
        buf = np.fromfile(paths[i], dtype=np.uint8)
        ok, k, c, ix, sb, eb = parse_ngram_json(buf)
        if not ok:
            raise ValueError(f"malformed ngram json: {paths[i]}")
        store(i, *build_csr_sorted(k, c, ix, sb, eb))

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(parse, range(n)))
    return key_off, keys_all, off_all, idx_all, sb_all, eb_all


def warmup():
    """Compile the loader kernels on a one-key document."""
    buf = np.frombuffer(b'{"1":[[0,1,2]]}', dtype=np.uint8)
    count_keys_pos(buf)
    ok, k, c, ix, sb, eb = parse_ngram_json(buf)
    build_csr_sorted(k, c, ix, sb, eb)
