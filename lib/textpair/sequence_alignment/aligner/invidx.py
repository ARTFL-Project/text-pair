"""Inverted-index intersection kernels for the sequence aligner.

Replaces compareNgrams' all-pairs merge join with:
  postings -> index the key groups -> per source document, sweep its own keys, emit one
  record per (key, comparable target), group by target and match.
Everything from the cross-product expansion onward follows main.go:507-528.

Key biasing: the int32 ngram hash is mapped to [0, 2^32) with `u = int64(key) + 2^31`, the
same monotone map as `uint32(key) ^ 0x80000000` but in int64 arithmetic.

Document slots are the combination's SortID order: sources first, then targets when source
and target corpora differ (`same_doc` non-empty). Array layout, with T = total (doc, key)
entries, P = total ngram positions:
  keys_all / key_off / off_all / idx_all / sb_all / eb_all  from loader.load_corpus
  post_u    uint32[T]  biased key, sorted
  post_slot int32[T]   slot in keys_all
  post_doc  int32[T]   document of that slot
  sweep_at  int32[source slots]  where a source's sweep of its key's group starts

The postings sort is stable and keys_all is document-major, so a key group holds its
documents in SortID order: the documents a source is compared against are the suffix of
the group that follows the source's own entry. Emissions are therefore built one source
at a time, which bounds them by the biggest source instead of by sum(df^2 / 2).
"""
import numpy as np
from numba import njit

from .kernels import NCOL, match_passage, merge_with_previous

MSD_BITS = 12                      # 4096 MSD buckets: 16 KB histogram per thread (L1)
MSD_BUCKETS = 1 << MSD_BITS
LOW_BITS = 32 - MSD_BITS           # 20 bits handled by two 10-bit LSD passes per bucket
LSD_RADIX = 1 << 10
BIAS = np.int64(1) << np.int64(31)


# --------------------------------------------------------------------------- postings

@njit(nogil=True, cache=True)
def hist_msd(keys, lo, hi, cnt):
    """Histogram of the top MSD_BITS of the biased key over keys[lo:hi]."""
    for i in range(lo, hi):
        cnt[(np.int64(keys[i]) + BIAS) >> LOW_BITS] += 1


@njit(nogil=True, cache=True)
def scatter_msd(keys, lo, hi, woff, post_u, post_slot):
    """Stable scatter of keys[lo:hi] into MSD buckets. woff is this chunk's write cursor."""
    for i in range(lo, hi):
        u = np.int64(keys[i]) + BIAS
        b = u >> LOW_BITS
        p = woff[b]
        woff[b] = p + 1
        post_u[p] = np.uint32(u)
        post_slot[p] = i


@njit(nogil=True, cache=True)
def sort_buckets(bstart, b_lo, b_hi, post_u, post_slot, buf_u, buf_s, cnt):
    """Sort each MSD bucket in [b_lo, b_hi) by the low LOW_BITS with two 10-bit LSD passes.
    Buckets are visited in ascending order, so the whole array ends up sorted by biased key.
    Both passes are stable, so equal keys keep their ascending slot order."""
    for b in range(b_lo, b_hi):
        s = bstart[b]
        n = bstart[b + 1] - s
        if n < 2:
            continue
        if n < 32:                                  # insertion sort on the biased key
            for i in range(s + 1, s + n):
                ku = post_u[i]
                ks = post_slot[i]
                j = i - 1
                while j >= s and post_u[j] > ku:
                    post_u[j + 1] = post_u[j]
                    post_slot[j + 1] = post_slot[j]
                    j -= 1
                post_u[j + 1] = ku
                post_slot[j + 1] = ks
            continue
        for shift in (0, 10):
            for i in range(LSD_RADIX):
                cnt[i] = 0
            for i in range(n):
                cnt[(post_u[s + i] >> shift) & (LSD_RADIX - 1)] += 1
            acc = 0
            for i in range(LSD_RADIX):
                c = cnt[i]
                cnt[i] = acc
                acc += c
            for i in range(n):
                d = (post_u[s + i] >> shift) & (LSD_RADIX - 1)
                j = cnt[d]
                cnt[d] = j + 1
                buf_u[j] = post_u[s + i]
                buf_s[j] = post_slot[s + i]
            for i in range(n):
                post_u[s + i] = buf_u[i]
                post_slot[s + i] = buf_s[i]


@njit(nogil=True, cache=True)
def _doc_of(key_off, slot):
    """Largest d with key_off[d] <= slot."""
    lo = 0
    hi = key_off.shape[0] - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if key_off[mid] <= slot:
            lo = mid
        else:
            hi = mid
    return lo


# --------------------------------------------------------------------------- key groups

@njit(nogil=True, cache=True)
def index_postings(bstart, b_lo, b_hi, post_u, post_slot, key_off, n_src, separate,
                   post_doc, sweep_at, cnt):
    """One pass over the key groups in [b_lo, b_hi): each posting's document, where every
    source document's sweep of its group starts, and cnt[source] += the emissions that
    sweep will yield. Returns the number of repeats -- the same key twice in one document,
    which the caller rejects.

    Within one corpus the sweep starts just past the source's own entry (main.go:487).
    With separate corpora it starts at the group's first target document (main.go:489
    drops only the target that has the source's own document ID, during the sweep).
    """
    repeats = 0
    for b in range(b_lo, b_hi):
        gs = bstart[b]
        e = bstart[b + 1]
        while gs < e:
            ge = gs + 1
            u = post_u[gs]
            while ge < e and post_u[ge] == u:
                ge += 1
            prev = -1
            for p in range(gs, ge):
                d = _doc_of(key_off, post_slot[p])
                post_doc[p] = d
                if d == prev:
                    repeats += 1
                prev = d
            if separate:
                ts = gs
                while ts < ge and post_doc[ts] < n_src:
                    ts += 1
                for p in range(gs, ts):
                    sweep_at[post_slot[p]] = ts
                    cnt[post_doc[p]] += ge - ts
            else:
                for p in range(gs, ge):
                    sweep_at[post_slot[p]] = p + 1
                    cnt[post_doc[p]] += ge - p - 1
            gs = ge
    return repeats


# --------------------------------------------------------------------------- emit + match

@njit(nogil=True, cache=True)
def align_source(s, excl, keys_all, key_off, sweep_at, post_u, post_slot, post_doc,
                 tcnt, toff, tcur, ssl, tsl,
                 off_all, idx_all, sb_all, eb_all,
                 min_in_docs, dup_threshold, window_size, max_gap, flex_gap,
                 min_matching, min_in_window, merge_byte, merge_ngram, multiplier):
    """Own one source document end to end: sweep its keys for the documents that follow
    it in each group, apply the two filters, then run the prototype's unchanged
    cross-product / sort / matchPassage pipeline.

    The sweep runs twice, first to size every target's block and then to fill it, so only
    the grouped (sourceSlot, targetSlot) pair is ever stored. The caller sizes ssl/tsl for
    this source's emissions and passes tcnt zeroed; tcnt is left zeroed again on return.

    Returns (rows, n_rows, stats, dup_st, dup_pct) with rows = int32[:, 2 + NCOL] of
    (source, target, alignment) and dup_st/dup_pct the duplicate pairs (main.go:499-505)
    that duplicate_files.csv needs and that still get a chunk file.
    stats = [pairs_with_common, pairs_below_min, pairs_duplicate, pairs_compared,
             sum_intersection_sizes, pairs_ge_min_in_docs]
    """
    out = np.empty((1024, NCOL + 2), np.int32)
    fill = 0
    dup_st = np.empty((64, 2), np.int32)
    dup_pct = np.empty(64, np.float64)
    ndup = 0
    stats = np.zeros(6, np.int64)
    n_post = post_u.shape[0]
    n_docs = key_off.shape[0] - 1

    first_key = key_off[s]
    last_key = key_off[s + 1]
    for i in range(first_key, last_key):                               # size the blocks
        u = np.uint32(np.int64(keys_all[i]) + BIAS)
        q = sweep_at[i]
        while q < n_post and post_u[q] == u:
            t = post_doc[q]
            if t != excl:
                tcnt[t] += 1
            q += 1
    acc = 0
    for t in range(n_docs):
        toff[t] = acc
        tcur[t] = acc
        acc += tcnt[t]
        tcnt[t] = 0
    toff[n_docs] = acc
    if acc == 0:
        return out[:0], 0, stats, dup_st[:0], dup_pct[:0]
    for i in range(first_key, last_key):                               # fill them
        u = np.uint32(np.int64(keys_all[i]) + BIAS)
        q = sweep_at[i]
        while q < n_post and post_u[q] == u:
            t = post_doc[q]
            if t != excl:
                p = tcur[t]
                tcur[t] = p + 1
                ssl[p] = i
                tsl[p] = post_slot[q]
            q += 1

    ns = last_key - first_key                 # sourceFile.NgramLength (distinct keys)
    for t in range(n_docs):
        q = toff[t]
        r = toff[t + 1]
        count = r - q
        if count == 0:
            continue
        stats[0] += 1
        stats[4] += count
        if count < min_in_docs:                                        # main.go:497
            stats[1] += 1
            continue
        stats[5] += 1
        pct = count / ns * 100
        if pct > dup_threshold:                                        # main.go:499
            stats[2] += 1
            if ndup == dup_st.shape[0]:
                new_st = np.empty((ndup * 2, 2), np.int32)
                new_pct = np.empty(ndup * 2, np.float64)
                new_st[:ndup] = dup_st
                new_pct[:ndup] = dup_pct
                dup_st = new_st
                dup_pct = new_pct
            dup_st[ndup, 0] = s
            dup_st[ndup, 1] = t
            dup_pct[ndup] = pct
            ndup += 1
            continue
        stats[3] += 1
        n_matches = 0
        for k in range(q, r):
            a = ssl[k] + s
            b = tsl[k] + t
            n_matches += (off_all[a + 1] - off_all[a]) * (off_all[b + 1] - off_all[b])
        packed = np.empty(n_matches, np.int64)                         # main.go:507-516
        srow = np.empty(n_matches, np.int32)
        trow = np.empty(n_matches, np.int32)
        k2 = 0
        for k in range(q, r):
            a = ssl[k] + s
            b = tsl[k] + t
            for p in range(off_all[a], off_all[a + 1]):
                for rr in range(off_all[b], off_all[b + 1]):
                    packed[k2] = (np.int64(idx_all[p]) << 32) | np.int64(idx_all[rr])
                    srow[k2] = p
                    trow[k2] = rr
                    k2 += 1
        order = np.argsort(packed)                                     # main.go:517-524
        m_sidx = np.empty(n_matches, np.int32)
        m_ssb = np.empty(n_matches, np.int32)
        m_seb = np.empty(n_matches, np.int32)
        m_tidx = np.empty(n_matches, np.int32)
        m_tsb = np.empty(n_matches, np.int32)
        m_teb = np.empty(n_matches, np.int32)
        for k in range(n_matches):
            p = srow[order[k]]
            rr = trow[order[k]]
            m_sidx[k] = idx_all[p]
            m_ssb[k] = sb_all[p]
            m_seb[k] = eb_all[p]
            m_tidx[k] = idx_all[rr]
            m_tsb[k] = sb_all[rr]
            m_teb[k] = eb_all[rr]
        al, n_al = match_passage(m_sidx, m_ssb, m_seb, m_tidx, m_tsb, m_teb, n_matches,
                                 window_size, max_gap, flex_gap, min_matching, min_in_window)
        if merge_byte or merge_ngram:                                  # main.go:526
            al, n_al = merge_with_previous(al, n_al, merge_byte, merge_ngram,
                                           window_size, multiplier)
        if n_al > 0:
            while fill + n_al > out.shape[0]:
                new = np.empty((out.shape[0] * 2, NCOL + 2), np.int32)
                new[:fill] = out[:fill]
                out = new
            for z in range(n_al):
                out[fill + z, 0] = s
                out[fill + z, 1] = t
                for cc in range(NCOL):
                    out[fill + z, cc + 2] = al[z, cc]
            fill += n_al
    return out[:fill], fill, stats, dup_st[:ndup], dup_pct[:ndup]
