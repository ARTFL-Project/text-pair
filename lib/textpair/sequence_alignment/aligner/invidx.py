"""Inverted-index intersection kernels for the sequence aligner.

Replaces compareNgrams' all-pairs merge join with:
  postings -> emit one record per (key, document pair) -> bucket by source -> group by target.
Everything from the cross-product expansion onward follows main.go:507-528.

Key biasing: the int32 ngram hash is mapped to [0, 2^32) with `u = int64(key) + 2^31`, the
same monotone map as `uint32(key) ^ 0x80000000` but in int64 arithmetic.

Document slots are the combination's SortID order: sources first, then targets when source
and target corpora differ (`same_doc` non-empty). Array layout, with T = total (doc, key)
entries, P = total ngram positions, M = total emissions:
  keys_all / key_off / off_all / idx_all / sb_all / eb_all  from loader.load_corpus
  post_u    uint32[T]  biased key, sorted                   (freed after emit)
  post_slot int32[T]   slot in keys_all                     (freed after emit)
  em_tgt/em_sslot/em_tslot int32[M]  source-major emissions, slots index keys_all
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
    Buckets are visited in ascending order, so the whole array ends up sorted by biased key."""
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


# --------------------------------------------------------------------------- emit

@njit(nogil=True, cache=True)
def _pair(da, dc, n_src, same_doc):
    """(source, target) document slots for a postings pair, or (-1, -1) when the pair is
    not compared: within one corpus (main.go:487) or a target with the source's own
    document ID (main.go:489). One corpus is signalled by an empty `same_doc`."""
    if same_doc.shape[0] == 0:
        if da < dc:
            return da, dc
        return dc, da
    if da < n_src:
        if dc < n_src:
            return -1, -1
        s = da
        t = dc
    else:
        if dc >= n_src:
            return -1, -1
        s = dc
        t = da
    if same_doc[s] == t:
        return -1, -1
    return s, t


@njit(nogil=True, cache=True)
def emit_count(bstart, b_lo, b_hi, post_u, post_slot, key_off, n_src, same_doc, cnt, docbuf):
    """Pass A: number of emissions this worker will contribute per source document.
    Returns (n_emissions, n_same_doc_pairs) -- the latter must be 0 (a key twice in one doc)."""
    total = 0
    same = 0
    for b in range(b_lo, b_hi):
        s = bstart[b]
        e = bstart[b + 1]
        i = s
        while i < e:
            j = i + 1
            u = post_u[i]
            while j < e and post_u[j] == u:
                j += 1
            n = j - i
            if n > 1:
                for a in range(n):
                    docbuf[a] = _doc_of(key_off, post_slot[i + a])
                for a in range(n - 1):
                    da = docbuf[a]
                    for c in range(a + 1, n):
                        dc = docbuf[c]
                        if da == dc:
                            same += 1
                            continue
                        s_doc, _ = _pair(da, dc, n_src, same_doc)
                        if s_doc < 0:
                            continue
                        cnt[s_doc] += 1
                        total += 1
            i = j
    return total, same


@njit(nogil=True, cache=True)
def emit_scatter(bstart, b_lo, b_hi, post_u, post_slot, key_off, n_src, same_doc, woff,
                 em_tgt, em_sslot, em_tslot, docbuf, slotbuf):
    """Pass B: write (target, sourceSlot, targetSlot) into this worker's slice of each
    source document's region. woff[d] is this worker's private cursor for source d."""
    for b in range(b_lo, b_hi):
        s = bstart[b]
        e = bstart[b + 1]
        i = s
        while i < e:
            j = i + 1
            u = post_u[i]
            while j < e and post_u[j] == u:
                j += 1
            n = j - i
            if n > 1:
                for a in range(n):
                    sl = post_slot[i + a]
                    slotbuf[a] = sl
                    docbuf[a] = _doc_of(key_off, sl)
                for a in range(n - 1):
                    da = docbuf[a]
                    for c in range(a + 1, n):
                        dc = docbuf[c]
                        if da == dc:
                            continue
                        src, tgt = _pair(da, dc, n_src, same_doc)
                        if src < 0:
                            continue
                        if src == da:
                            ss = slotbuf[a]
                            ts = slotbuf[c]
                        else:
                            ss = slotbuf[c]
                            ts = slotbuf[a]
                        p = woff[src]
                        woff[src] = p + 1
                        em_tgt[p] = tgt
                        em_sslot[p] = ss
                        em_tslot[p] = ts
            i = j


# --------------------------------------------------------------------------- group + match

@njit(nogil=True, cache=True)
def _radix_by_tgt(t0, s0, x0, n, npass, t1, s1, x1, cnt):
    """Stable LSD radix sort of (tgt, sslot, tslot) by tgt, npass passes of 8 bits."""
    for p in range(npass):
        shift = 8 * p
        for i in range(256):
            cnt[i] = 0
        for i in range(n):
            cnt[(t0[i] >> shift) & 255] += 1
        acc = 0
        for i in range(256):
            c = cnt[i]
            cnt[i] = acc
            acc += c
        for i in range(n):
            d = (t0[i] >> shift) & 255
            j = cnt[d]
            cnt[d] = j + 1
            t1[j] = t0[i]
            s1[j] = s0[i]
            x1[j] = x0[i]
        t0, t1 = t1, t0
        s0, s1 = s1, s0
        x0, x1 = x1, x0
    return t0, s0, x0


@njit(nogil=True, cache=True)
def match_sources(sources, src_off, em_tgt, em_sslot, em_tslot,
                  key_off, off_all, idx_all, sb_all, eb_all, npass,
                  min_in_docs, dup_threshold, window_size, max_gap, flex_gap,
                  min_matching, min_in_window, merge_byte, merge_ngram, multiplier):
    """Own a set of source documents end to end: group each region by target, apply the two
    filters, then run the prototype's unchanged cross-product / sort / matchPassage pipeline.

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
    cnt256 = np.empty(256, np.int32)
    for si in range(sources.shape[0]):
        s = sources[si]
        lo = src_off[s]
        hi = src_off[s + 1]
        n = hi - lo
        if n == 0:
            continue
        ns = key_off[s + 1] - key_off[s]          # sourceFile.NgramLength (distinct keys)
        tg, ssl, tsl = _radix_by_tgt(em_tgt[lo:hi], em_sslot[lo:hi], em_tslot[lo:hi], n, npass,
                                     np.empty(n, np.int32), np.empty(n, np.int32),
                                     np.empty(n, np.int32), cnt256)
        q = 0
        while q < n:
            r = q + 1
            t = tg[q]
            while r < n and tg[r] == t:
                r += 1
            count = r - q
            stats[0] += 1
            stats[4] += count
            if count < min_in_docs:                                        # main.go:497
                stats[1] += 1
                q = r
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
                q = r
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
            q = r
    return out[:fill], fill, stats, dup_st[:ndup], dup_pct[:ndup]
