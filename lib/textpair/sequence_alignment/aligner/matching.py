"""Passage matching and passage merging.

Kernels take int32/int64 arrays and scalars only, nogil and cached.
An alignment row is int32[9]: source start byte, source end byte, source first index,
source last index, target start byte, target end byte, target first index,
target last index, total matching ngrams.

Both kernels are symmetric: comparing a pair of documents the other way round gives the
mirrored passages. `MATCHER_SYMMETRY.md` measures the asymmetric matcher they replaced,
which agreed with itself on 52% of frantext passages when the document order was
reversed, and records what it cost to fix.
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


@njit(nogil=True, cache=True)
def _grow(out):
    new = np.empty((out.shape[0] * 2, NCOL), np.int32)
    new[: out.shape[0]] = out
    return new


@njit(nogil=True, cache=True)
def _pair_key(packed):
    """The two ngram indices as an unordered pair, so it is the same either way round.

    31 bits a side: an ngram index is a non-negative int32, so no document is long
    enough to make two different matches share a key by overflow.
    """
    source = packed >> 32
    target = packed & TARGET_HALF
    low = source if source < target else target
    high = source if source > target else target
    return (low << 31) | high


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
    covered by a kept passage. Grows out and spans together."""
    source_lo = packed_indices[chain[first]] >> 32
    source_hi = packed_indices[chain[last]] >> 32
    target_lo = packed_indices[chain[first]] & TARGET_HALF
    target_hi = packed_indices[chain[last]] & TARGET_HALF
    if (_overlaps_kept(source_lo, source_hi, spans, n_alignments, 0)
            and _overlaps_kept(target_lo, target_hi, spans, n_alignments, 1)):
        return out, n_alignments, spans
    if n_alignments == out.shape[0]:
        out = _grow(out)
        bigger = np.empty((out.shape[0], 4), np.int32)
        bigger[:n_alignments] = spans[:n_alignments]
        spans = bigger
    spans[n_alignments, 0] = source_lo
    spans[n_alignments, 1] = source_hi
    spans[n_alignments, 2] = target_lo
    spans[n_alignments, 3] = target_hi
    head = packed_positions[chain[first]]
    tail = packed_positions[chain[last]]
    out[n_alignments, 0] = start_bytes[head >> 32]
    out[n_alignments, 1] = end_bytes[tail >> 32]
    out[n_alignments, 2] = source_lo
    out[n_alignments, 3] = source_hi
    out[n_alignments, 4] = start_bytes[head & TARGET_HALF]
    out[n_alignments, 5] = end_bytes[tail & TARGET_HALF]
    out[n_alignments, 6] = target_lo
    out[n_alignments, 7] = target_hi
    out[n_alignments, 8] = count
    return out, n_alignments + 1, spans


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
def _chain_flat(packed_indices, n, max_link, best, parent, used):
    """Longest chain ending at each match, scanning every match in the source window."""
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
            step = source_step + target_step
            # Among equally long chains, the nearest predecessor, measured by the two
            # steps as an unordered pair: their total first, then the smaller of them,
            # which together fix both. Both are symmetric in the two documents, so the
            # choice does not depend on which one is the source. What that leaves tied
            # is a predecessor at (di, dj) against one at (dj, di) -- exact mirror
            # images, which nothing symmetric can separate.
            near = source_step if source_step < target_step else target_step
            if candidate > best_b or (candidate == best_b
                                      and (step < best_step
                                           or (step == best_step
                                               and near < best_near))):
                best_b = candidate
                parent_b = a
                best_step = step
                best_near = near
        best[b] = best_b
        parent[b] = parent_b
        used[b] = 0
        if best_b > longest:
            longest = best_b
    return longest


@njit(nogil=True, cache=True)
def _chain_blocked(packed_indices, n, max_link, best, parent, used):
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
                candidate = best[a] + np.int32(1)
                step = source_step + target_step
                near = source_step if source_step < target_step else target_step
                if candidate > best_b or (candidate == best_b
                                          and (step < best_step
                                               or (step == best_step
                                                   and near < best_near))):
                    best_b = candidate
                    parent_b = a
                    best_step = step
                    best_near = near
        best[b] = best_b
        parent[b] = parent_b
        used[b] = 0
        if best_b > longest:
            longest = best_b
    return longest


@njit(nogil=True, cache=True)
def match_passage(packed_indices, packed_positions, n, n_blocks, start_bytes, end_bytes,
                  window_size, max_gap, flex_gap, min_matching, min_in_window,
                  best, parent, used, chain, key, order, spans, out):
    """Matches must be sorted by (source index, target index). `n_blocks` is how many
    distinct source indices they hold, which picks the predecessor scan.

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
    them. Returns (out, n_alignments, spans).
    """
    n_alignments = 0
    if n == 0:
        return out, 0, spans
    max_link = link_bound(window_size, max_gap, flex_gap, min_matching)

    if n_blocks > 0 and n >= BLOCK_MIN_MATCHES and n >= BLOCK_FANOUT * n_blocks:
        longest = _chain_blocked(packed_indices, n, max_link, best, parent, used)
    else:
        longest = _chain_flat(packed_indices, n, max_link, best, parent, used)
    if longest < min_matching:
        return out, 0, spans                # no chain here can reach the threshold

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
    for rank in range(1, n_ends + 1):
        if rank < n_ends and best[order[rank]] == best[order[block_start]]:
            continue
        for i in range(block_start + 1, rank):
            entry = order[i]
            entry_key = _pair_key(packed_indices[entry])
            j = i - 1
            while j >= block_start and _pair_key(packed_indices[order[j]]) > entry_key:
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = entry
        block_start = rank

    for rank in range(n_ends):
        end = order[rank]
        if used[end]:
            continue
        length = 0
        at = np.int32(end)
        while at >= 0 and not used[at]:
            chain[length] = at
            length += 1
            at = parent[at]
        if length < min_matching:
            used[end] = 1
            continue
        for c in range(length // 2):        # the walk collected the chain backwards
            chain[c], chain[length - 1 - c] = chain[length - 1 - c], chain[c]
        for c in range(length):
            used[chain[c]] = 1
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
    return out, n_alignments, spans


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
