"""Inverted-index intersection kernels for the sequence aligner.

Replaces an all-pairs merge join with:
  postings -> index the key groups -> per source document, sweep its own keys, emit one
  record per (key, comparable target), group by target and match.

Key biasing: an int32 ngram hash is mapped onto [0, 2^32) by adding 2^31 in int64
arithmetic, the same monotone map as flipping the sign bit, so that sorting the biased
keys as unsigned sorts the signed originals.

Document slots are the combination's SortID order: sources first, then targets when source
and target corpora differ (`same_doc` non-empty). Array layout, with T = total (doc, key)
entries, P = total ngram positions:
  ngram_keys, key_offsets, position_offsets, ngram_indices, start_bytes, end_bytes
            from loader.load_corpus
  posting_keys   uint32[T]  biased key, sorted
  posting_slots  int32[T]   slot in ngram_keys
  posting_docs   int32[T]   document of that slot
  sweep_starts   int32[source slots]  where a source's sweep of its group starts

The postings sort is stable and ngram_keys is document-major, so a key group holds its
documents in SortID order: the documents a source is compared against are the suffix of
the group that follows the source's own entry. Emissions are therefore built one source
at a time, which bounds them by the biggest source instead of by sum(df^2 / 2).
"""
import numpy as np
from numba import njit

from .kernels import NCOL, match_passage, merge_passages

MSD_BITS = 12                      # 4096 MSD buckets: 16 KB histogram per thread (L1)
MSD_BUCKETS = 1 << MSD_BITS
LOW_BITS = 32 - MSD_BITS           # 20 bits handled by two 10-bit LSD passes per bucket
LSD_RADIX = 1 << 10
BIAS = np.int64(1) << np.int64(31)


# --------------------------------------------------------------------------- postings

@njit(nogil=True, cache=True)
def hist_msd(keys, lo, hi, bucket_counts):
    """Histogram of the top MSD_BITS of the biased key over keys[lo:hi]."""
    for i in range(lo, hi):
        bucket_counts[(np.int64(keys[i]) + BIAS) >> LOW_BITS] += 1


@njit(nogil=True, cache=True)
def scatter_msd(keys, lo, hi, write_cursors, posting_keys, posting_slots):
    """Stable scatter of keys[lo:hi] into MSD buckets, at this chunk's write cursors."""
    for i in range(lo, hi):
        biased_key = np.int64(keys[i]) + BIAS
        bucket = biased_key >> LOW_BITS
        at = write_cursors[bucket]
        write_cursors[bucket] = at + 1
        posting_keys[at] = np.uint32(biased_key)
        posting_slots[at] = i


@njit(nogil=True, cache=True)
def sort_buckets(bucket_starts, first_bucket, last_bucket, posting_keys, posting_slots,
                 key_buffer, slot_buffer, digit_counts):
    """Sort each MSD bucket in [first_bucket, last_bucket) by the low LOW_BITS, with
    two 10-bit LSD passes.
    Buckets are visited in ascending order, so the whole array ends up sorted by biased key.
    Both passes are stable, so equal keys keep their ascending slot order."""
    for bucket in range(first_bucket, last_bucket):
        start = bucket_starts[bucket]
        count = bucket_starts[bucket + 1] - start
        if count < 2:
            continue
        if count < 32:                              # insertion sort on the biased key
            for i in range(start + 1, start + count):
                key = posting_keys[i]
                slot = posting_slots[i]
                j = i - 1
                while j >= start and posting_keys[j] > key:
                    posting_keys[j + 1] = posting_keys[j]
                    posting_slots[j + 1] = posting_slots[j]
                    j -= 1
                posting_keys[j + 1] = key
                posting_slots[j + 1] = slot
            continue
        for shift in (0, 10):
            for digit in range(LSD_RADIX):
                digit_counts[digit] = 0
            for i in range(count):
                digit_counts[(posting_keys[start + i] >> shift) & (LSD_RADIX - 1)] += 1
            running = 0
            for digit in range(LSD_RADIX):
                in_digit = digit_counts[digit]
                digit_counts[digit] = running
                running += in_digit
            for i in range(count):
                digit = (posting_keys[start + i] >> shift) & (LSD_RADIX - 1)
                at = digit_counts[digit]
                digit_counts[digit] = at + 1
                key_buffer[at] = posting_keys[start + i]
                slot_buffer[at] = posting_slots[start + i]
            for i in range(count):
                posting_keys[start + i] = key_buffer[i]
                posting_slots[start + i] = slot_buffer[i]


@njit(nogil=True, cache=True)
def _doc_of(key_offsets, slot):
    """The document owning `slot`: the largest d with key_offsets[d] <= slot."""
    lo = 0
    hi = key_offsets.shape[0] - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if key_offsets[mid] <= slot:
            lo = mid
        else:
            hi = mid
    return lo


# --------------------------------------------------------------------------- key groups

@njit(nogil=True, cache=True)
def index_postings(bucket_starts, first_bucket, last_bucket, posting_keys,
                   posting_slots, key_offsets, n_sources, separate, posting_docs,
                   sweep_starts, emission_counts):
    """One pass over the key groups in [first_bucket, last_bucket): each posting's
    document, where every source document's sweep of its group starts, and
    emission_counts[source] += the emissions that sweep will yield. Returns the number of
    repeats -- the same key twice in one document, which the caller rejects.

    Within one corpus the sweep starts just past the source's own entry. With separate
    corpora it starts at the group's first target document, and the target that carries
    the source's own document ID is dropped later, during the sweep itself.
    """
    repeats = 0
    for bucket in range(first_bucket, last_bucket):
        group_start = bucket_starts[bucket]
        bucket_end = bucket_starts[bucket + 1]
        while group_start < bucket_end:
            group_end = group_start + 1
            biased_key = posting_keys[group_start]
            while group_end < bucket_end and posting_keys[group_end] == biased_key:
                group_end += 1
            previous_doc = -1
            for at in range(group_start, group_end):
                doc = _doc_of(key_offsets, posting_slots[at])
                posting_docs[at] = doc
                if doc == previous_doc:
                    repeats += 1
                previous_doc = doc
            if separate:
                first_target = group_start
                while first_target < group_end and posting_docs[first_target] < n_sources:
                    first_target += 1
                for at in range(group_start, first_target):
                    sweep_starts[posting_slots[at]] = first_target
                    emission_counts[posting_docs[at]] += group_end - first_target
            else:
                for at in range(group_start, group_end):
                    sweep_starts[posting_slots[at]] = at + 1
                    emission_counts[posting_docs[at]] += group_end - at - 1
            group_start = group_end
    return repeats


# --------------------------------------------------------------------------- emit + match

@njit(nogil=True, cache=True)
def sort_insertion(keys, n, order):
    """Order 0..n by `keys`. Cheapest for the many pairs with few source positions."""
    for i in range(n):
        order[i] = i
    for i in range(1, n):
        o = order[i]
        k = keys[o]
        j = i - 1
        while j >= 0 and keys[order[j]] > k:
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = o
    return order


@njit(nogil=True, cache=True)
def sort_radix(keys, n, max_key, order, spare, digit_counts):
    """Order 0..n by `keys`, LSD radix over only the bytes `max_key` needs.

    Ping-pongs between `order` and `spare` and returns whichever holds the result, so
    neither this nor the caller allocates. `keys` are ngram indices within a document:
    non-negative and usually inside 16 bits, so this is two passes.
    """
    for i in range(n):
        order[i] = i
    source = order
    target = spare
    shift = 0
    while (max_key >> shift) > 0:
        for digit in range(256):
            digit_counts[digit] = 0
        for i in range(n):
            digit_counts[(keys[source[i]] >> shift) & 255] += 1
        running = 0
        for digit in range(256):
            in_digit = digit_counts[digit]
            digit_counts[digit] = running
            running += in_digit
        for i in range(n):
            entry = source[i]
            digit = (keys[entry] >> shift) & 255
            target[digit_counts[digit]] = entry
            digit_counts[digit] += 1
        source, target = target, source
        shift += 8
    return source


@njit(nogil=True, cache=True)
def align_source(s, exclude_slot, ngram_keys, key_offsets, sweep_starts, posting_keys,
                 posting_slots, posting_docs,
                 target_counts, target_offsets, target_cursors, source_slots, target_slots,
                 position_offsets, ngram_indices, start_bytes, end_bytes,
                 min_in_docs, dup_threshold, window_size, max_gap, flex_gap,
                 min_matching, min_in_window, merge_byte, merge_ngram, multiplier):
    """Own one source document end to end: sweep its keys for the documents that follow
    it in each group, apply the two filters, then expand each surviving pair's matches
    and hand them to kernels.match_passage.

    The sweep runs twice, first to size every target's block and then to fill it, so only
    the grouped (source slot, target slot) pair is ever stored. The caller sizes
    source_slots and target_slots for
    this source's emissions and passes target_counts zeroed; target_counts is left
    zeroed again on return.

    Returns (rows, n_rows, stats, duplicate_slots, duplicate_percents), with
    rows = int32[:, 2 + NCOL] of
    (source, target, alignment) and duplicate_slots/duplicate_percents the duplicate pairs
    that duplicate_files.csv needs and that still get a chunk file.
    stats = [pairs_with_common, pairs_below_min, pairs_duplicate, pairs_compared,
             sum_intersection_sizes, pairs_ge_min_in_docs]
    """
    out = np.empty((1024, NCOL + 2), np.int32)
    n_rows = 0
    # Scratch for one target at a time, reused across every target of this source. Held
    # here rather than allocated per pair because 3,253 matches is 76 KiB of match
    # arrays: reused it stays in L2 between pairs, freshly allocated every write is a
    # cold first touch, and eccotcp compares 4.3 million pairs.
    position_capacity = 64
    source_positions = np.empty(position_capacity, np.int32)
    source_indices = np.empty(position_capacity, np.int32)
    target_block_starts = np.empty(position_capacity, np.int32)
    target_block_ends = np.empty(position_capacity, np.int32)
    order_buffer = np.empty(position_capacity, np.int32)
    order_spare = np.empty(position_capacity, np.int32)
    digit_counts = np.empty(256, np.int64)
    match_capacity = 256
    packed_indices = np.empty(match_capacity, np.int64)
    packed_positions = np.empty(match_capacity, np.int64)
    # match_passage's chaining buffers, kept across this source's targets for the same
    # reason the match arrays are. `chain_spans` and `chain_out` grow inside the kernel
    # and come back out, so a later pair inherits whatever size an earlier one needed.
    chain_best = np.empty(match_capacity, np.int32)
    chain_parent = np.empty(match_capacity, np.int32)
    chain_used = np.empty(match_capacity, np.uint8)
    chain_members = np.empty(match_capacity, np.int32)
    chain_key = np.empty(match_capacity + 1, np.int32)
    chain_order = np.empty(match_capacity, np.int32)
    chain_spans = np.empty((64, 4), np.int32)
    chain_out = np.empty((64, NCOL), np.int32)
    duplicate_slots = np.empty((64, 2), np.int32)
    duplicate_percents = np.empty(64, np.float64)
    n_duplicates = 0
    stats = np.zeros(6, np.int64)
    n_postings = posting_keys.shape[0]
    n_docs = key_offsets.shape[0] - 1

    first_key = key_offsets[s]
    last_key = key_offsets[s + 1]
    for i in range(first_key, last_key):                               # size the blocks
        u = np.uint32(np.int64(ngram_keys[i]) + BIAS)
        q = sweep_starts[i]
        while q < n_postings and posting_keys[q] == u:
            t = posting_docs[q]
            if t != exclude_slot:
                target_counts[t] += 1
            q += 1
    running_total = 0
    for t in range(n_docs):
        target_offsets[t] = running_total
        target_cursors[t] = running_total
        running_total += target_counts[t]
        target_counts[t] = 0
    target_offsets[n_docs] = running_total
    if running_total == 0:
        return out[:0], 0, stats, duplicate_slots[:0], duplicate_percents[:0]
    for i in range(first_key, last_key):                               # n_rows them
        u = np.uint32(np.int64(ngram_keys[i]) + BIAS)
        q = sweep_starts[i]
        while q < n_postings and posting_keys[q] == u:
            t = posting_docs[q]
            if t != exclude_slot:
                p = target_cursors[t]
                target_cursors[t] = p + 1
                source_slots[p] = i
                target_slots[p] = posting_slots[q]
            q += 1

    ns = last_key - first_key                 # sourceFile.NgramLength (distinct keys)
    for t in range(n_docs):
        q = target_offsets[t]
        r = target_offsets[t + 1]
        count = r - q
        if count == 0:
            continue
        stats[0] += 1
        stats[4] += count
        if count < min_in_docs:
            stats[1] += 1
            continue
        stats[5] += 1
        # The share of the smaller document's ngrams the two have in common: the same
        # reading of "one of these is a reprint of the other" whichever way round the
        # pair is compared. Dividing by the source's own count made a short document
        # inside a long one a duplicate one way only.
        nt = key_offsets[t + 1] - key_offsets[t]
        pct = count / (ns if ns < nt else nt) * 100
        if pct > dup_threshold:
            stats[2] += 1
            if n_duplicates == duplicate_slots.shape[0]:
                new_st = np.empty((n_duplicates * 2, 2), np.int32)
                new_pct = np.empty(n_duplicates * 2, np.float64)
                new_st[:n_duplicates] = duplicate_slots
                new_pct[:n_duplicates] = duplicate_percents
                duplicate_slots = new_st
                duplicate_percents = new_pct
            duplicate_slots[n_duplicates, 0] = s
            duplicate_slots[n_duplicates, 1] = t
            duplicate_percents[n_duplicates] = pct
            n_duplicates += 1
            continue
        stats[3] += 1
        # match_passage needs the matches ordered by (source index, target index).
        # Sorting the cross-product to get there is wasteful: a source index is unique
        # within its document and a key's positions are ascending by index -- the
        # writers sort keys stably, keeping each key's positions in collection order --
        # so ordering the pair's source positions alone and expanding each one's target
        # block in place produces exactly that order, over three times fewer elements.
        # tests/test_match_order.py checks both properties on a corpus.
        n_positions = 0
        n_matches = 0
        for k in range(q, r):
            a = source_slots[k] + s
            b = target_slots[k] + t
            in_source = position_offsets[a + 1] - position_offsets[a]
            n_positions += in_source
            n_matches += in_source * (position_offsets[b + 1] - position_offsets[b])
        if n_positions > position_capacity:
            position_capacity = n_positions * 2
            source_positions = np.empty(position_capacity, np.int32)
            source_indices = np.empty(position_capacity, np.int32)
            target_block_starts = np.empty(position_capacity, np.int32)
            target_block_ends = np.empty(position_capacity, np.int32)
            order_buffer = np.empty(position_capacity, np.int32)
            order_spare = np.empty(position_capacity, np.int32)
        j = 0
        max_source_index = 0
        for k in range(q, r):
            a = source_slots[k] + s
            b = target_slots[k] + t
            block_start = position_offsets[b]
            block_end = position_offsets[b + 1]
            for p in range(position_offsets[a], position_offsets[a + 1]):
                index = ngram_indices[p]
                # where it sits in ngram_indices, start_bytes and end_bytes
                source_positions[j] = p
                source_indices[j] = index               # its ngram index, the sort key
                if index > max_source_index:
                    max_source_index = index
                target_block_starts[j] = block_start   # the shared key's target positions
                target_block_ends[j] = block_end
                j += 1
        if n_positions < 64:
            order = sort_insertion(source_indices, n_positions, order_buffer)
        else:
            order = sort_radix(source_indices, n_positions, max_source_index,
                               order_buffer, order_spare, digit_counts)
        if n_matches > match_capacity:
            match_capacity = n_matches * 2
            packed_indices = np.empty(match_capacity, np.int64)
            packed_positions = np.empty(match_capacity, np.int64)
            chain_best = np.empty(match_capacity, np.int32)
            chain_parent = np.empty(match_capacity, np.int32)
            chain_used = np.empty(match_capacity, np.uint8)
            chain_members = np.empty(match_capacity, np.int32)
            chain_key = np.empty(match_capacity + 1, np.int32)
            chain_order = np.empty(match_capacity, np.int32)
        written = 0
        for rank in range(n_positions):
            entry = order[rank]
            at = source_positions[entry]
            # The source half is constant down a source position's run, so it is shifted
            # once and only the target half varies. Byte offsets are not copied at all;
            # match_passage reaches them through packed_positions.
            source_index_half = np.int64(source_indices[entry]) << 32
            source_position_half = np.int64(at) << 32
            for target_at in range(target_block_starts[entry], target_block_ends[entry]):
                packed_indices[written] = (source_index_half
                                           | np.int64(ngram_indices[target_at]))
                packed_positions[written] = source_position_half | np.int64(target_at)
                written += 1
        chain_out, n_alignments, chain_spans = match_passage(
            packed_indices, packed_positions, n_matches, start_bytes, end_bytes,
            window_size, max_gap, flex_gap, min_matching, min_in_window,
            chain_best, chain_parent, chain_used, chain_members, chain_key,
            chain_order, chain_spans, chain_out)
        alignments = chain_out
        if merge_byte or merge_ngram:
            alignments, n_alignments = merge_passages(
                alignments, n_alignments, merge_byte, merge_ngram, window_size,
                multiplier)
        if n_alignments > 0:
            while n_rows + n_alignments > out.shape[0]:
                new = np.empty((out.shape[0] * 2, NCOL + 2), np.int32)
                new[:n_rows] = out[:n_rows]
                out = new
            for z in range(n_alignments):
                out[n_rows + z, 0] = s
                out[n_rows + z, 1] = t
                for cc in range(NCOL):
                    out[n_rows + z, cc + 2] = alignments[z, cc]
            n_rows += n_alignments
    return (out[:n_rows], n_rows, stats, duplicate_slots[:n_duplicates],
            duplicate_percents[:n_duplicates])
