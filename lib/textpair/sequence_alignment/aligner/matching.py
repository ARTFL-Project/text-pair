"""Passage matching and passage merging.

Kernels take int32/int64 arrays and scalars only, nogil and cached.
An alignment row is int32[9]: source start byte, source end byte, source first index,
source last index, target start byte, target end byte, target first index,
target last index, total matching ngrams.

Each passage's ends are then re-seeded: beside them, within max_gap words, ngrams are
looked for with a word skipped in either document, which an inserted, dropped or changed
word ("Le suffrage par le sort, dit Montesquieu, est de la nature...") leaves the two
documents no ngram to share otherwise. Chained onto the passage, as many of them as a
passage needs ngrams, less one, move its end.

Every kernel is symmetric: comparing a pair of documents the other way round gives the
mirrored passages. `MATCHER_SYMMETRY.md` measures the asymmetric matcher the chains
replaced, which agreed with itself on 52% of frantext passages when the document order
was reversed, and records what it cost to fix. The anchored scan is that matcher's walk,
run from both documents so that it is symmetric too.
"""
import math
import numpy as np
from numba import njit

NCOL = 9
# The blocked predecessor scan pays only where a source position expands to many
# matches; with near-unique matches its per-block bookkeeping costs more than the flat
# scan it replaces. eebo_ecco has pairs at a fan-out of 8,000.
BLOCK_FANOUT = 8
BLOCK_MIN_MATCHES = 256
# A packed int64 holds the source value in the high half and the target in the low.
TARGET_HALF = np.int64(0xFFFFFFFF)
# _order_by_target's digit: two passes cover any document under 4 million ngrams.
RADIX_BITS = 11
RADIX = 1 << RADIX_BITS


@njit(nogil=True, cache=True)
def _grow(out):
    new = np.empty((out.shape[0] * 2, NCOL), np.int32)
    new[: out.shape[0]] = out
    return new


@njit(nogil=True, cache=True)
def _pair_key(packed, source_first):
    """The two ngram indices as an unordered pair, so it is the same either way round,
    then the index in the first document, which separates (i, j) from (j, i).

    31 bits a side: an ngram index is a non-negative int32, so no document is long
    enough to make two different matches share a key by overflow.
    """
    source = packed >> 32
    target = packed & TARGET_HALF
    low = source if source < target else target
    high = source if source > target else target
    return (low << 31) | high, source if source_first else target



@njit(nogil=True, cache=True)
def _merge_pass(src, src_off, dst, dst_off, m, width, packed_indices, source_first):
    """One bottom-up merge pass. Ties take the left run, so the pass is stable."""
    i = 0
    while i < m:
        mid = i + width
        if mid > m:
            mid = m
        end = i + 2 * width
        if end > m:
            end = m
        a = i
        b = mid
        o = i
        if b < end:
            ka = _pair_key(packed_indices[src[src_off + a]], source_first)
            kb = _pair_key(packed_indices[src[src_off + b]], source_first)
            while True:
                if kb < ka:
                    dst[dst_off + o] = src[src_off + b]
                    o += 1
                    b += 1
                    if b >= end:
                        break
                    kb = _pair_key(packed_indices[src[src_off + b]], source_first)
                else:
                    dst[dst_off + o] = src[src_off + a]
                    o += 1
                    a += 1
                    if a >= mid:
                        break
                    ka = _pair_key(packed_indices[src[src_off + a]], source_first)
        while a < mid:
            dst[dst_off + o] = src[src_off + a]
            a += 1
            o += 1
        while b < end:
            dst[dst_off + o] = src[src_off + b]
            b += 1
            o += 1
        i = end


@njit(nogil=True, cache=True)
def _sort_block(order, lo, hi, packed_indices, buf, source_first):
    """Stable ascending sort of order[lo:hi] by _pair_key. buf needs hi - lo slots."""
    m = hi - lo
    if m < 2:
        return
    run = 32
    start = lo
    while start < hi:
        stop = start + run
        if stop > hi:
            stop = hi
        for i in range(start + 1, stop):
            entry = order[i]
            entry_key = _pair_key(packed_indices[entry], source_first)
            j = i - 1
            while (j >= start
                   and _pair_key(packed_indices[order[j]], source_first) > entry_key):
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = entry
        start = stop
    width = run
    in_order = True
    while width < m:
        if in_order:
            _merge_pass(order, lo, buf, 0, m, width, packed_indices, source_first)
        else:
            _merge_pass(buf, 0, order, lo, m, width, packed_indices, source_first)
        in_order = not in_order
        width *= 2
    if not in_order:
        for i in range(m):
            order[lo + i] = buf[i]


@njit(nogil=True, cache=True)
def _sort_rows(out, n):
    """By source index then target index: what merge_passages and the record order want.
    Chains are emitted longest-first, which is not that order."""
    for i in range(1, n):
        row = out[i].copy()
        j = i - 1
        while j >= 0 and (out[j, 2] > row[2]
                          or (out[j, 2] == row[2] and out[j, 6] > row[6])):
            out[j + 1] = out[j]
            j -= 1
        out[j + 1] = row
    return out, n


@njit(nogil=True, cache=True)
def _overlaps_kept(lo, hi, spans, n_spans, side):
    for k in range(n_spans):
        if lo <= spans[k, side * 2 + 1] and spans[k, side * 2] <= hi:
            return True
    return False


@njit(nogil=True, cache=True)
def _keep(out, n_alignments, spans, packed_indices, packed_positions, chain,
          first, last, count, start_bytes, end_bytes):
    """Emit chain[first:last] as a passage unless both of its stretches are already
    covered by a kept passage."""
    source_lo = packed_indices[chain[first]] >> 32
    source_hi = packed_indices[chain[last]] >> 32
    target_lo = packed_indices[chain[first]] & TARGET_HALF
    target_hi = packed_indices[chain[last]] & TARGET_HALF
    if (_overlaps_kept(source_lo, source_hi, spans, n_alignments, 0)
            and _overlaps_kept(target_lo, target_hi, spans, n_alignments, 1)):
        return out, n_alignments, spans
    if n_alignments == spans.shape[0]:
        bigger = np.empty((spans.shape[0] * 2, 4), np.int32)
        bigger[:n_alignments] = spans[:n_alignments]
        spans = bigger
    spans[n_alignments, 0] = source_lo
    spans[n_alignments, 1] = source_hi
    spans[n_alignments, 2] = target_lo
    spans[n_alignments, 3] = target_hi
    out, n_alignments = _emit(out, n_alignments, packed_indices, packed_positions,
                              chain[first], chain[last], count, start_bytes, end_bytes)
    return out, n_alignments, spans


@njit(nogil=True, cache=True)
def _emit(out, n_alignments, packed_indices, packed_positions, first, last, count,
          start_bytes, end_bytes):
    """Append the passage running from match `first` to match `last`."""
    if n_alignments == out.shape[0]:
        out = _grow(out)
    head = packed_positions[first]
    tail = packed_positions[last]
    out[n_alignments, 0] = start_bytes[head >> 32]
    out[n_alignments, 1] = end_bytes[tail >> 32]
    out[n_alignments, 2] = packed_indices[first] >> 32
    out[n_alignments, 3] = packed_indices[last] >> 32
    out[n_alignments, 4] = start_bytes[head & TARGET_HALF]
    out[n_alignments, 5] = end_bytes[tail & TARGET_HALF]
    out[n_alignments, 6] = packed_indices[first] & TARGET_HALF
    out[n_alignments, 7] = packed_indices[last] & TARGET_HALF
    out[n_alignments, 8] = count
    return out, n_alignments + 1


@njit(nogil=True, cache=True)
def link_bound(window_size, max_gap, flex_gap, min_matching):
    """The largest step `match_chains` must consider linking across.

    Without flex_gap that is max_gap. With it, the run's allowance starts at max_gap,
    jumps by min_matching once the run reaches min_matching matches, then climbs by one
    per match while it is below window_size -- so it cannot exceed the larger of
    max_gap + min_matching and window_size. Linking to that ceiling and enforcing the
    real allowance while walking the chain keeps every run the old kernel could build
    reachable.
    """
    if not flex_gap:
        return max_gap
    flexed = max_gap + min_matching
    return flexed if flexed > window_size else window_size


@njit(nogil=True, cache=True)
def _chain_flat(packed_indices, n, max_link, near_gap, source_first, best, parent, used):
    """Longest chain ending at each match, scanning every match in the source window.

    Also sets bit 2 of used[a] when a match follows a within near_gap in both documents:
    the anchored scan skips anchors without one."""
    window_start = 0
    longest = 1
    for b in range(n):
        source_b = packed_indices[b] >> 32
        target_b = packed_indices[b] & TARGET_HALF
        while (packed_indices[window_start] >> 32) < source_b - max_link:
            window_start += 1
        best_b = np.int32(1)
        parent_b = np.int32(-1)
        best_step = np.int64(0)
        best_near = np.int64(0)
        best_first = np.int64(0)
        for a in range(window_start, b):
            source_a = packed_indices[a] >> 32
            if source_a == source_b:
                break                       # and so are all the matches after it
            target_a = packed_indices[a] & TARGET_HALF
            target_step = target_b - target_a
            if target_step <= 0 or target_step > max_link:
                continue
            candidate = best[a] + np.int32(1)
            source_step = source_b - source_a
            if source_step <= near_gap and target_step <= near_gap:
                used[a] |= 2
            step = source_step + target_step
            # Among equally long chains, the nearest predecessor, measured by the two
            # steps as an unordered pair: their total first, then the smaller of them,
            # which together fix both. Both are symmetric in the two documents, so the
            # choice does not depend on which one is the source. What that leaves tied
            # is a predecessor at (di, dj) against one at (dj, di), exact mirror images:
            # those go to the smaller step in whichever document comes first by
            # identity, the same one either way round.
            near = source_step if source_step < target_step else target_step
            first_step = source_step if source_first else target_step
            if candidate > best_b or (candidate == best_b
                                      and (step < best_step
                                           or (step == best_step
                                               and (near < best_near
                                                    or (near == best_near
                                                        and first_step
                                                        < best_first))))):
                best_b = candidate
                parent_b = a
                best_step = step
                best_near = near
                best_first = first_step
        best[b] = best_b
        parent[b] = parent_b
        used[b] = 0
        if best_b > longest:
            longest = best_b
    return longest


@njit(nogil=True, cache=True)
def _chain_blocked(packed_indices, n, max_link, near_gap, source_first, best, parent,
                   used):
    """The same scan, reaching each source block's candidates by binary search.

    Matches sharing a source index are contiguous and their target indices ascend, so
    the candidates a block contributes are a contiguous slice of it: exactly the ones
    _chain_flat does not skip, visited in the order it visits them. Where a key repeats
    thousands of times in the target, a block holds thousands of matches of which
    almost none are within max_link, and the flat scan walks all of them -- 75 billion
    iterations on one eebo_ecco pair, against 20 million here.

    At most max_link blocks can precede a match inside the window, since their source
    indices are distinct, so the ring holds every block that can still be reached.
    """
    ring = max_link + 2
    block_at = np.empty(ring, np.int64)
    block_source = np.empty(ring, np.int64)
    n_seen = 0
    current_source = np.int64(-1)
    longest = 1
    for b in range(n):
        source_b = packed_indices[b] >> 32
        target_b = packed_indices[b] & TARGET_HALF
        if source_b != current_source:
            current_source = source_b
            block_at[n_seen % ring] = b
            block_source[n_seen % ring] = source_b
            n_seen += 1
        best_b = np.int32(1)
        parent_b = np.int32(-1)
        best_step = np.int64(0)
        best_near = np.int64(0)
        best_first = np.int64(0)
        target_floor = target_b - max_link
        first = n_seen - 1 - max_link
        if first < 0:
            first = 0
        for k in range(first, n_seen - 1):
            source_a = block_source[k % ring]
            if source_a < source_b - max_link:
                continue
            lo = block_at[k % ring]
            hi = block_at[(k + 1) % ring]
            left = lo
            right = hi
            while left < right:              # first target at or past target_floor
                mid = (left + right) >> 1
                if (packed_indices[mid] & TARGET_HALF) < target_floor:
                    left = mid + 1
                else:
                    right = mid
            source_step = source_b - source_a
            for a in range(left, hi):
                target_a = packed_indices[a] & TARGET_HALF
                if target_a >= target_b:
                    break
                target_step = target_b - target_a
                if source_step <= near_gap and target_step <= near_gap:
                    used[a] |= 2
                candidate = best[a] + np.int32(1)
                step = source_step + target_step
                near = source_step if source_step < target_step else target_step
                first_step = source_step if source_first else target_step
                if candidate > best_b or (candidate == best_b
                                          and (step < best_step
                                               or (step == best_step
                                                   and (near < best_near
                                                        or (near == best_near
                                                            and first_step
                                                            < best_first))))):
                    best_b = candidate
                    parent_b = a
                    best_step = step
                    best_near = near
                    best_first = first_step
        best[b] = best_b
        parent[b] = parent_b
        used[b] = 0
        if best_b > longest:
            longest = best_b
    return longest


@njit(nogil=True, cache=True)
def _order_by_target(packed_indices, n, order, spare, keys):
    """The matches by (target index, source index), as indices into them: a stable LSD
    radix sort on the target index of matches already ordered by source index. `keys`
    is scratch for n. Returns whichever of `order` and `spare` holds the result."""
    low = packed_indices[0] & TARGET_HALF
    high = low
    for i in range(n):
        target = packed_indices[i] & TARGET_HALF
        if target < low:
            low = target
        if target > high:
            high = target
    for i in range(n):
        keys[i] = (packed_indices[i] & TARGET_HALF) - low
    if n < 64:
        for i in range(n):
            order[i] = i
        for i in range(1, n):
            entry = order[i]
            key = keys[entry]
            j = i - 1
            while j >= 0 and keys[order[j]] > key:
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = entry
        return order
    span = high - low
    passes = 0
    while (span >> (passes * RADIX_BITS)) > 0:
        passes += 1
    if passes == 0:
        for i in range(n):
            order[i] = i
        return order
    counts = np.zeros((passes, RADIX), np.int64)
    for i in range(n):                          # every digit's histogram in one pass
        key = keys[i]
        for p in range(passes):
            counts[p, (key >> (p * RADIX_BITS)) & (RADIX - 1)] += 1
    for p in range(passes):
        running = 0
        for digit in range(RADIX):
            count = counts[p, digit]
            counts[p, digit] = running
            running += count
    for i in range(n):                          # the first pass reads in match order
        digit = keys[i] & (RADIX - 1)
        order[counts[0, digit]] = i
        counts[0, digit] += 1
    source, destination = order, spare
    for p in range(1, passes):
        shift = p * RADIX_BITS
        for i in range(n):
            entry = source[i]
            digit = (keys[entry] >> shift) & (RADIX - 1)
            destination[counts[p, digit]] = entry
            counts[p, digit] += 1
        source, destination = destination, source
    return source


@njit(nogil=True, cache=True)
def _anchored_scan(packed_indices, packed_positions, n, order, mirrored, best, used,
                   skip, start_bytes, end_bytes, window_size, max_gap, flex_gap,
                   min_matching, min_in_window, out, n_alignments):
    """The Go aligner's walk: through one document in order, each run anchored at a
    match's first partner in the other. `mirrored` walks the target, through `order`,
    the matches by (target, source). It pairs repeated material differently from the
    chains, and a step past the gap allowance in the walked document is accepted while
    the window already holds min_in_window matches.

    `best` and `used` are match_passage's; `skip` is scratch for n ranks.
    """
    if n == 0:
        return out, n_alignments
    # A run's first `lead` steps cannot overshoot the gap, so its lead-th match ends a
    # chain of at least lead. No step goes further than the window can grow, so stretches
    # between wider gaps are independent, and one with no such match has no run: skip[r]
    # is where rank r's stretch ends if so, else -1.
    lead = min_matching if min_matching < min_in_window else min_in_window
    widest = window_size
    if flex_gap:
        widest += min_matching
        if max_gap + min_matching < window_size:
            widest += window_size - max_gap - min_matching
    first = 0
    strong = False
    previous = np.int64(-1)
    for rank in range(n + 1):
        entry = np.int64(0)
        w = np.int64(0)
        if rank < n:
            entry = order[rank] if mirrored else rank
            if mirrored:
                w = packed_indices[entry] & TARGET_HALF
            else:
                w = packed_indices[entry] >> 32
        if rank == n or (rank > first and w - previous > widest):
            for r in range(first, rank):
                skip[r] = -1 if strong else rank
            first = rank
            strong = False
        if rank < n:
            if best[entry] >= lead:
                strong = True
            previous = w
    final = order[n - 1] if mirrored else n - 1
    if mirrored:
        last_of_all = packed_indices[final] & TARGET_HALF
    else:
        last_of_all = packed_indices[final] >> 32
    resume = 0
    anchor_rank = -1
    while anchor_rank + 1 < n:
        anchor_rank += 1
        if skip[anchor_rank] >= 0:
            anchor_rank = skip[anchor_rank] - 1
            continue
        anchor = order[anchor_rank] if mirrored else anchor_rank
        if mirrored:
            walked = packed_indices[anchor] & TARGET_HALF
            other = packed_indices[anchor] >> 32
        else:
            walked = packed_indices[anchor] >> 32
            other = packed_indices[anchor] & TARGET_HALF
        if walked < resume:
            continue
        # With nothing inside max_gap after it, the run takes no match and breaks, which
        # resumes past the anchor -- unless no match is left to break it.
        if (not (used[anchor] & 2) and max_gap <= window_size
                and last_of_all > walked + max_gap):
            resume = walked + 1
            continue
        walked_boundary = walked + window_size
        other_boundary = other + window_size
        last_walked = walked
        last_other = other
        walked_limit = walked + max_gap
        other_limit = other + max_gap
        previous_walked = walked
        last = anchor
        in_run = True
        in_alignment = 1
        in_window = 1
        gap = max_gap
        window = window_size
        for rank in range(anchor_rank + 1, n):
            entry = order[rank] if mirrored else rank
            if mirrored:
                w = packed_indices[entry] & TARGET_HALF
                o = packed_indices[entry] >> 32
            else:
                w = packed_indices[entry] >> 32
                o = packed_indices[entry] & TARGET_HALF
            if w == previous_walked:
                continue
            if o > other_limit or o <= last_other:
                if w <= walked_limit:
                    continue
                in_run = False
            if w > walked_limit and in_window < min_in_window:
                in_run = False
            if w > walked_boundary or o > other_boundary:
                if in_window < min_in_window:
                    in_run = False
                elif w > walked_limit or o > other_limit:
                    in_run = False
                else:
                    walked_boundary = w + window
                    other_boundary = o + window
                    in_window = 0
            if not in_run:
                break
            last_walked = w
            walked_limit = w + gap
            last_other = o
            other_limit = o + gap
            previous_walked = w
            in_window += 1
            in_alignment += 1
            if flex_gap:
                if in_alignment == min_matching:
                    gap += min_matching
                    window += min_matching
                elif in_alignment > min_matching and gap < window_size:
                    gap += 1
                    window += 1
            last = entry
        if in_alignment >= min_matching:
            out, n_alignments = _emit(out, n_alignments, packed_indices, packed_positions,
                                      anchor, last, in_alignment, start_bytes, end_bytes)
        resume = last_walked + 1 if not in_run else last_walked
    return out, n_alignments


@njit(nogil=True, cache=True)
def _skip(c):
    """The other two words of the c-th way of keeping three of four words, the first kept."""
    if c == 0:
        return 1, 2
    if c == 1:
        return 1, 3
    return 2, 3


@njit(nogil=True, cache=True)
def _local_matches(source_words, s_lo, s_hi, s_end, target_words, t_lo, t_hi, t_end, out):
    """The ngrams the two documents share with a word skipped in either or neither, by the
    (first, last) word of each: first words in [s_lo, s_hi) and [t_lo, t_hi), last words
    before s_end and t_end. Returns (out, count), out grown as needed."""
    n = 0
    for p in range(s_lo, s_hi):
        for q in range(t_lo, t_hi):
            if source_words[p] != target_words[q]:
                continue
            for c in range(3):
                i, j = _skip(c)
                if p + j >= s_end:
                    continue
                for d in range(3):
                    k, l = _skip(d)
                    if q + l >= t_end:
                        continue
                    if source_words[p + i] == target_words[q + k] and \
                            source_words[p + j] == target_words[q + l]:
                        if n == out.shape[0]:
                            grown = np.empty((2 * n, 4), np.int64)
                            grown[:n] = out
                            out = grown
                        out[n, 0], out[n, 1], out[n, 2], out[n, 3] = p, p + j, q, q + l
                        n += 1
    return out, n


@njit(nogil=True, cache=True)
def _farthest(found, n, a, b, gap, before, source_first):
    """Of the matches a chain reaches from word a and word b, in steps of 1 to gap words in
    both documents, the one at the end of the longest chain, then the farthest, then the
    farthest in the document first by identity: (its first and last words in each, the
    chain's length). Matches at the same first words are one, as far as either reaches."""
    keys = (found[:n, 0] << 32) | found[:n, 2]
    order = np.argsort(keys)
    m = np.empty((n, 4), np.int64)
    k = 0
    for o in order:
        if k and m[k - 1, 0] == found[o, 0] and m[k - 1, 2] == found[o, 2]:
            m[k - 1, 1] = max(m[k - 1, 1], found[o, 1])
            m[k - 1, 3] = max(m[k - 1, 3], found[o, 3])
        else:
            m[k] = found[o]
            k += 1
    length = np.zeros(k, np.int64)
    best = -1
    for step in range(k):
        # nearest the passage first, so a match's chain onward is known when it is reached
        i = k - 1 - step if before else step
        p, q = m[i, 0], m[i, 2]
        dp, dq = (a - p, b - q) if before else (p - a, q - b)
        here = 1 if 0 < dp <= gap and 0 < dq <= gap else 0
        for back in range(step):
            j = k - 1 - back if before else back
            if length[j] == 0:
                continue
            dp, dq = (m[j, 0] - p, m[j, 2] - q) if before else (p - m[j, 0], q - m[j, 2])
            if 0 < dp <= gap and 0 < dq <= gap and length[j] + 1 > here:
                here = length[j] + 1
        length[i] = here
        if here == 0:
            continue
        if best < 0 or here > length[best]:
            best = i
            continue
        if here < length[best]:
            continue
        far, best_far = p + q, m[best, 0] + m[best, 2]
        first, best_first = (p, m[best, 0]) if source_first else (q, m[best, 2])
        if before and (far < best_far or far == best_far and first < best_first) or \
                not before and (far > best_far or far == best_far and first > best_first):
            best = i
    if best < 0:
        return 0, 0, 0, 0, 0
    return m[best, 0], m[best, 1], m[best, 2], m[best, 3], length[best]


@njit(nogil=True, cache=True)
def reseed(rows, n, neighbours, n_neighbours, source_words, target_words, source_at,
           target_at, start_bytes, end_bytes, max_gap, min_matching, source_first):
    """Each passage's ends moved to the farthest of the ngrams beside them, within max_gap
    words, that the two documents share once a word is skipped in either, when a chain of
    min_matching - 1 of them reaches it; in place. An end one of `neighbours` (the pair's
    passages, rows too) lies within twice max_gap words of, in either document, is left
    alone, so no two moved ends meet: whether two passages so near are one is merging's to say.

    `*_words` are each document's word keys, word i being the first of ngram i; `*_at` give
    each ngram index's position in start_bytes and end_bytes. A document without words is
    not re-seeded.
    """
    ns, nt = source_words.shape[0], target_words.shape[0]
    need = max(min_matching - 1, 1)
    if ns == 0 or nt == 0 or n == 0:
        return
    # every passage's first and last words, before any is moved
    others = neighbours[:n_neighbours]
    source_firsts, target_firsts = np.sort(others[:, 2]), np.sort(others[:, 6])
    source_lasts, target_lasts = np.sort(others[:, 3] + 2), np.sort(others[:, 7] + 2)
    found = np.empty((64, 4), np.int64)
    reach = 2 * max_gap
    for r in range(n):
        a, b = rows[r, 2], rows[r, 6]
        a_end, b_end = rows[r, 3] + 2, rows[r, 7] + 2
        # the nearest passage ending before this one, and starting after it, in each document
        k, j = np.searchsorted(source_lasts, a) - 1, np.searchsorted(target_lasts, b) - 1
        near = k >= 0 and source_lasts[k] >= a - reach or j >= 0 and target_lasts[j] >= b - reach
        k = np.searchsorted(source_firsts, a_end, side="right")
        j = np.searchsorted(target_firsts, b_end, side="right")
        near_end = k < n_neighbours and source_firsts[k] <= a_end + reach or \
            j < n_neighbours and target_firsts[j] <= b_end + reach
        found, m = (found, 0) if near else _local_matches(
            source_words, max(a - max_gap, 0), a, min(a + 3, ns),
            target_words, max(b - max_gap, 0), b, min(b + 3, nt), found)
        if m:
            p, _, q, _, chained = _farthest(found, m, a, b, max_gap, True, source_first)
            if chained >= need and source_at[p] >= 0 and target_at[q] >= 0:
                rows[r, 0], rows[r, 2] = start_bytes[source_at[p]], p
                rows[r, 4], rows[r, 6] = start_bytes[target_at[q]], q
                rows[r, 8] += chained
        a, b = rows[r, 3], rows[r, 7]
        found, m = (found, 0) if near_end else _local_matches(
            source_words, a + 1, min(a + max_gap + 1, ns), min(a_end + max_gap + 1, ns),
            target_words, b + 1, min(b + max_gap + 1, nt), min(b_end + max_gap + 1, nt), found)
        if m:
            _, p, _, q, chained = _farthest(found, m, a, b, max_gap, False, source_first)
            if chained >= need and source_at[p - 2] >= 0 and target_at[q - 2] >= 0:
                rows[r, 1], rows[r, 3] = end_bytes[source_at[p - 2]], p - 2
                rows[r, 5], rows[r, 7] = end_bytes[target_at[q - 2]], q - 2
                rows[r, 8] += chained


@njit(nogil=True, cache=True)
def _coalesce(out, n):
    """Passages overlapping in both documents become one passage spanning them all."""
    out, n = _sort_rows(out, n)
    parent = np.empty(n, np.int64)
    for i in range(n):
        parent[i] = i
    for i in range(n):
        for j in range(i + 1, n):
            if out[j, 2] > out[i, 3]:
                break                       # sorted by source start: none later overlaps
            if out[j, 6] <= out[i, 7] and out[i, 6] <= out[j, 7]:
                root_i = _root(parent, i)
                root_j = _root(parent, j)
                if root_i < root_j:
                    parent[root_j] = root_i
                elif root_j < root_i:
                    parent[root_i] = root_j
    n_out = 0
    for i in range(n):
        root = _root(parent, i)
        if root == i:
            continue
        # A root precedes its members and is never itself a member, so it accumulates
        # in place.
        if out[i, 2] < out[root, 2]:
            out[root, 0] = out[i, 0]
            out[root, 2] = out[i, 2]
        if out[i, 3] > out[root, 3]:
            out[root, 1] = out[i, 1]
            out[root, 3] = out[i, 3]
        if out[i, 6] < out[root, 6]:
            out[root, 4] = out[i, 4]
            out[root, 6] = out[i, 6]
        if out[i, 7] > out[root, 7]:
            out[root, 5] = out[i, 5]
            out[root, 7] = out[i, 7]
        if out[i, 8] > out[root, 8]:        # the matches overlap, so not their sum
            out[root, 8] = out[i, 8]
    for i in range(n):
        if parent[i] == i:
            if n_out != i:
                out[n_out] = out[i]
            n_out += 1
    return _sort_rows(out, n_out)


@njit(nogil=True, cache=True)
def match_passage(packed_indices, packed_positions, n, n_blocks, start_bytes, end_bytes,
                  window_size, max_gap, flex_gap, min_matching, min_in_window,
                  source_first, best, parent, used, chain, key, order, spans, out):
    """Matches must be sorted by (source index, target index). `n_blocks` is how many
    distinct source indices they hold, which picks the predecessor scan. `source_first`
    says whether the source is the first of the pair by document identity, which is what
    settles ties between mirror images.

    A passage is a chain of matches strictly increasing in both ngram indices, with
    consecutive steps inside the run's gap allowance in either document, dense enough
    that every window_size stretch of either document holds min_in_window of them. Every
    test is a symmetric function of the two indices, so comparing a pair the other way
    round gives the mirrored passages. The matcher this replaced walked the source once
    and reserved the source range of each passage it emitted, which reports a phrase
    occurring once in the source and n times in the target as one passage, and as n when
    the same pair is compared the other way.

    Chains are found longest-first by dynamic programming, so a passage is never cut
    short by a nearer but unextendable match, and a chain is kept only when it covers a
    stretch of either document that no kept chain covers yet. Reserving neither side
    instead reports n*m passages for a phrase occurring n and m times; reserving both is
    what makes the count about n+m, every occurrence on both sides reported once.

    Every buffer is the caller's: best, parent, used, chain, order sized for n matches,
    key for n + 1, and spans and out grown here and handed back so the next pair reuses
    them. Returns (out, n_alignments, spans, longest), longest being the longest chain.
    """
    n_alignments = 0
    if n == 0:
        return out, 0, spans, 0
    max_link = link_bound(window_size, max_gap, flex_gap, min_matching)
    # Past the window, a first step inside max_gap can still end a run, so no skip then.
    near_gap = max_gap if max_gap <= window_size else -1

    if n_blocks > 0 and n >= BLOCK_MIN_MATCHES and n >= BLOCK_FANOUT * n_blocks:
        longest = _chain_blocked(packed_indices, n, max_link, near_gap, source_first,
                                 best, parent, used)
    else:
        longest = _chain_flat(packed_indices, n, max_link, near_gap, source_first, best,
                              parent, used)
    if longest < min_matching:
        return out, 0, spans, longest       # no chain here can reach the threshold

    # Chain ends, longest first, over only the matches that could end one: a counting
    # sort on the length, then each length's block by its coordinate pair as an
    # unordered pair, so the order does not depend on which document is the source.
    n_ends = 0
    n_buckets = longest - min_matching + 2
    for bucket in range(n_buckets):
        key[bucket] = 0
    for b in range(n):
        if best[b] >= min_matching:
            key[longest - best[b]] += 1
            n_ends += 1
    running = 0
    for bucket in range(n_buckets):
        count = key[bucket]
        key[bucket] = running
        running += count
    for b in range(n):
        if best[b] >= min_matching:
            bucket = longest - best[b]
            order[key[bucket]] = b
            key[bucket] += 1
    block_start = 0
    for bucket in range(n_buckets):
        block_end = key[bucket]
        _sort_block(order, block_start, block_end, packed_indices, chain, source_first)
        block_start = block_end

    for rank in range(n_ends):
        end = order[rank]
        if used[end] & 1:
            continue
        length = 0
        at = np.int32(end)
        while at >= 0 and not (used[at] & 1):
            chain[length] = at
            length += 1
            at = parent[at]
        if length < min_matching:
            used[end] |= 1
            continue
        for c in range(length // 2):        # the walk collected the chain backwards
            chain[c], chain[length - 1 - c] = chain[length - 1 - c], chain[c]
        for c in range(length):
            used[chain[c]] |= 1
        # Walk it applying the run's real gap allowance and the window test, cutting
        # where either fails and keeping both sides rather than dropping everything past
        # the first failure.
        segment_start = 0
        anchor = 0
        in_window = 1
        in_segment = 1
        gap_allowance = max_gap
        window = window_size
        for t in range(1, length + 1):
            cut = t == length
            if not cut:
                source_t = packed_indices[chain[t]] >> 32
                target_t = packed_indices[chain[t]] & TARGET_HALF
                previous = packed_indices[chain[t - 1]]
                if (source_t - (previous >> 32) > gap_allowance
                        or target_t - (previous & TARGET_HALF) > gap_allowance):
                    cut = True
                else:
                    source_anchor = packed_indices[chain[anchor]] >> 32
                    target_anchor = packed_indices[chain[anchor]] & TARGET_HALF
                    if (source_t > source_anchor + window
                            or target_t > target_anchor + window):
                        if in_window < min_in_window:
                            cut = True
                        else:
                            anchor = t
                            in_window = 0
            if cut:
                count = t - segment_start
                if count >= min_matching:
                    out, n_alignments, spans = _keep(
                        out, n_alignments, spans, packed_indices, packed_positions,
                        chain, segment_start, t - 1, count, start_bytes, end_bytes)
                if t == length:
                    break
                segment_start = t
                anchor = t
                in_window = 1
                in_segment = 1
                gap_allowance = max_gap
                window = window_size
                continue
            in_window += 1
            in_segment += 1
            if flex_gap:
                if in_segment == min_matching:
                    gap_allowance += min_matching
                    window += min_matching
                elif in_segment > min_matching and gap_allowance < window_size:
                    gap_allowance += 1
                    window += 1
    out, n_alignments = _sort_rows(out, n_alignments)
    return out, n_alignments, spans, longest


@njit(nogil=True, cache=True)
def _root(parent, x):
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


@njit(nogil=True, cache=True)
def merge_passages(al, n, merge_byte, merge_ngram, window_size, multiplier):
    """Merge passages that continue one another, independently of which document is the
    source. Rows must be in source order.

    Three things differ from the merger this replaced. Each document's byte allowance
    comes from that document's own passage length, where one allowance derived from the
    source length was applied to both. The candidate must not overlap the passage it
    merges into in either document, where only the target was checked -- enough while
    passages could not overlap in the source, which they can now that the matcher does
    not reserve source ranges. And merging is over the connected components of that
    relation rather than a single left-to-right pass, so a chain of merges does not
    depend on the order the passages arrive in.
    """
    if n == 0:
        return al, 0
    parent = np.empty(n, np.int64)
    for i in range(n):
        parent[i] = i
    max_ngram_distance = window_size if merge_ngram else 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # i strictly before j in both documents, or the relation says nothing here.
            if al[i, 1] >= al[j, 1] or al[i, 5] >= al[j, 5]:
                continue
            if al[j, 0] <= al[i, 1] or al[j, 4] <= al[i, 5]:
                continue
            merged = False
            if merge_byte:
                source_room = int(math.floor(float(al[i, 1] - al[i, 0]) * multiplier))
                target_room = int(math.floor(float(al[i, 5] - al[i, 4]) * multiplier))
                if (al[j, 0] <= al[i, 1] + source_room
                        and al[j, 4] <= al[i, 5] + target_room):
                    merged = True
            if not merged and merge_ngram:
                if (al[j, 2] <= al[i, 3] + max_ngram_distance
                        and al[j, 6] <= al[i, 7] + max_ngram_distance):
                    merged = True
            if merged:
                root_i = _root(parent, i)
                root_j = _root(parent, j)
                if root_i != root_j:
                    if root_i < root_j:     # the lowest row represents the component, so
                        parent[root_j] = root_i     # the result does not depend on the
                    else:                           # order the merges were found in
                        parent[root_i] = root_j
    out = np.empty((n, NCOL), np.int32)
    n_out = 0
    for i in range(n):
        if _root(parent, i) != i:
            continue
        for c in range(NCOL):
            out[n_out, c] = al[i, c]
        for j in range(n):
            if j == i or _root(parent, j) != i:
                continue
            if al[j, 0] < out[n_out, 0]:
                out[n_out, 0] = al[j, 0]
                out[n_out, 2] = al[j, 2]
            if al[j, 1] > out[n_out, 1]:
                out[n_out, 1] = al[j, 1]
                out[n_out, 3] = al[j, 3]
            if al[j, 4] < out[n_out, 4]:
                out[n_out, 4] = al[j, 4]
                out[n_out, 6] = al[j, 6]
            if al[j, 5] > out[n_out, 5]:
                out[n_out, 5] = al[j, 5]
                out[n_out, 7] = al[j, 7]
            out[n_out, 8] += al[j, 8]
        n_out += 1
    return _sort_rows(out, n_out)


@njit(nogil=True, cache=True)
def _merged(rows, n, merging, merge_byte, merge_ngram, window_size, multiplier):
    rows, n = _sort_rows(rows, n)
    if not merging:
        return rows, n
    return merge_passages(rows, n, merge_byte, merge_ngram, window_size, multiplier)


@njit(nogil=True, cache=True)
def align_pair(packed_indices, packed_positions, n, n_blocks, start_bytes, end_bytes,
               window_size, max_gap, flex_gap, min_matching, min_in_window,
               merge_byte, merge_ngram, multiplier, source_first,
               source_words, target_words, source_at, target_at, at_start_bytes,
               at_end_bytes, best, parent, used, chain, key, order, spans, out):
    """One pair's passages: the chains, and the anchored scan walking each document,
    each set re-seeded and merged on its own and then coalesced.

    The chains alone lose some of what the Go aligner found: a chain longest under the
    loose link can be cut into pieces too short to keep, and on repeated material the
    cover pairs occurrences differently. Merging each set before coalescing keeps every
    passage the chains give inside one of the results.

    `source_words` to `at_end_bytes` are reseed's. Buffers are match_passage's. Returns
    (rows, n_rows, spans, out), spans and out being the grown buffers to hand the next
    pair.
    """
    out, n_chains, spans, longest = match_passage(
        packed_indices, packed_positions, n, n_blocks, start_bytes, end_bytes,
        window_size, max_gap, flex_gap, min_matching, min_in_window, source_first,
        best, parent, used, chain, key, order, spans, out)
    # A scan's run either is a chain or reaches its first step past the gap with
    # min_in_window matches linked normally, so below both nothing here can pass.
    if longest < min_matching and longest < min_in_window:
        return out[:0], 0, spans, out
    forward = np.empty((64, NCOL), np.int32)
    forward, n_forward = _anchored_scan(
        packed_indices, packed_positions, n, order, False, best, used, key, start_bytes,
        end_bytes,
        window_size, max_gap, flex_gap, min_matching, min_in_window, forward, 0)
    by_target = _order_by_target(packed_indices, n, order, chain, parent)
    backward = np.empty((64, NCOL), np.int32)
    backward, n_backward = _anchored_scan(
        packed_indices, packed_positions, n, by_target, True, best, used, key,
        start_bytes, end_bytes,
        window_size, max_gap, flex_gap, min_matching, min_in_window, backward, 0)
    if source_words.shape[0] and target_words.shape[0]:
        # every passage of the pair, whichever set found it, is one an end may come near
        found = n_chains + n_forward + n_backward
        every = np.empty((found, NCOL), np.int32)
        every[:n_chains] = out[:n_chains]
        every[n_chains:n_chains + n_forward] = forward[:n_forward]
        every[n_chains + n_forward:found] = backward[:n_backward]
        for rows, count in ((out, n_chains), (forward, n_forward), (backward, n_backward)):
            reseed(rows, count, every, found, source_words, target_words, source_at,
                   target_at, at_start_bytes, at_end_bytes, max_gap, min_matching,
                   source_first)
    merging = merge_byte or merge_ngram
    chains, n_chains = _merged(out, n_chains, merging, merge_byte, merge_ngram,
                               window_size, multiplier)
    forward, n_forward = _merged(forward, n_forward, merging, merge_byte, merge_ngram,
                                 window_size, multiplier)
    backward, n_backward = _merged(backward, n_backward, merging, merge_byte, merge_ngram,
                                   window_size, multiplier)
    total = n_chains + n_forward + n_backward
    rows = np.empty((total, NCOL), np.int32)
    rows[:n_chains] = chains[:n_chains]
    rows[n_chains:n_chains + n_forward] = forward[:n_forward]
    rows[n_chains + n_forward:total] = backward[:n_backward]
    rows, total = _coalesce(rows, total)
    return rows, total, spans, out
