"""Passage matching and passage merging.

Kernels take int32/int64 arrays and scalars only, nogil and cached.
An alignment row is int32[9]: source start byte, source end byte, source first index,
source last index, target start byte, target end byte, target first index,
target last index, total matching ngrams.
"""
import math
import numpy as np
from numba import njit

NCOL = 9
# A packed int64 holds the source value in the high half and the target in the low.
TARGET_HALF = np.int64(0xFFFFFFFF)


@njit(nogil=True, cache=True)
def _grow(out):
    new = np.empty((out.shape[0] * 2, NCOL), np.int32)
    new[: out.shape[0]] = out
    return new


@njit(nogil=True, cache=True)
def match_passage(packed_indices, packed_positions, n, start_bytes, end_bytes,
                  window_size, max_gap_cfg, flex_gap, min_matching, min_in_window):
    """Matches must be sorted by (source index, target index).

    A match is two packed int64s. `packed_indices` holds its source and target ngram
    indices, which drive every comparison, so they are packed together and read as one
    stream. `packed_positions` holds its source and target positions in start_bytes and
    end_bytes: those offsets are wanted only at a run's first match and on the ones it
    accepts, so they stay in the corpus arrays and are fetched when needed rather than
    copied for every match.
    """
    out = np.empty((64, NCOL), np.int32)
    n_alignments = 0
    last_source_position = 0
    in_alignment = False
    for match_index in range(n):
        anchor_indices = packed_indices[match_index]
        anchor_source_index = anchor_indices >> 32
        if anchor_source_index < last_source_position:
            continue
        anchor_target_index = anchor_indices & TARGET_HALF
        anchor_positions = packed_positions[match_index]
        anchor_source_position = anchor_positions >> 32
        anchor_target_position = anchor_positions & TARGET_HALF
        source_anchor = anchor_source_index
        source_window_boundary = source_anchor + window_size
        last_source_position = source_anchor
        max_source_gap = last_source_position + max_gap_cfg
        target_anchor = anchor_target_index
        target_window_boundary = target_anchor + window_size
        last_target_position = target_anchor
        max_target_gap = last_target_position + max_gap_cfg
        in_alignment = True
        previous_source_index = source_anchor
        first_source_start_byte = start_bytes[anchor_source_position]   # the run's first
        first_source_index = anchor_source_index
        first_target_start_byte = start_bytes[anchor_target_position]
        first_target_index = anchor_target_index
        matches_in_current_alignment = 1
        matches_in_current_window = 1
        last_source_end_byte = end_bytes[anchor_source_position]        # and its last
        last_source_index = anchor_source_index
        last_target_end_byte = end_bytes[anchor_target_position]
        last_target_index = anchor_target_index
        max_gap = max_gap_cfg
        matching_window_size = window_size
        for j in range(match_index + 1, n):
            match_indices = packed_indices[j]
            source_index = match_indices >> 32
            target_index = match_indices & TARGET_HALF
            if source_index == previous_source_index:
                continue
            if target_index > max_target_gap or target_index <= last_target_position:
                # The bound here is on the current match, so it always holds and the
                # run continues whenever the source index is still within the gap.
                if source_index <= max_source_gap:
                    continue
                else:
                    in_alignment = False
            if source_index > max_source_gap and matches_in_current_window < min_in_window:
                in_alignment = False
            if source_index > source_window_boundary or target_index > target_window_boundary:
                if matches_in_current_window < min_in_window:
                    in_alignment = False
                else:
                    if source_index > max_source_gap or target_index > max_target_gap:
                        in_alignment = False
                    else:
                        source_anchor = source_index
                        source_window_boundary = source_anchor + matching_window_size
                        target_anchor = target_index
                        target_window_boundary = target_anchor + matching_window_size
                        matches_in_current_window = 0
            if not in_alignment:
                if matches_in_current_alignment >= min_matching:
                    if n_alignments == out.shape[0]:
                        out = _grow(out)
                    out[n_alignments, 0] = first_source_start_byte
                    out[n_alignments, 1] = last_source_end_byte
                    out[n_alignments, 2] = first_source_index
                    out[n_alignments, 3] = last_source_index
                    out[n_alignments, 4] = first_target_start_byte
                    out[n_alignments, 5] = last_target_end_byte
                    out[n_alignments, 6] = first_target_index
                    out[n_alignments, 7] = last_target_index
                    out[n_alignments, 8] = matches_in_current_alignment
                    n_alignments += 1
                last_source_position = last_source_index + 1
                break
            last_source_position = source_index
            max_source_gap = last_source_position + max_gap
            last_target_position = target_index
            max_target_gap = last_target_position + max_gap
            previous_source_index = source_index
            matches_in_current_window += 1
            matches_in_current_alignment += 1
            if flex_gap:
                if matches_in_current_alignment == min_matching:
                    max_gap += min_matching
                    matching_window_size += min_matching
                elif matches_in_current_alignment > min_matching:
                    if max_gap < window_size:
                        max_gap += 1
                        matching_window_size += 1
            match_positions = packed_positions[j]
            last_source_end_byte = end_bytes[match_positions >> 32]
            last_source_index = source_index
            last_target_end_byte = end_bytes[match_positions & TARGET_HALF]
            last_target_index = target_index
        if in_alignment and matches_in_current_alignment >= min_matching:
            if n_alignments == out.shape[0]:
                out = _grow(out)
            out[n_alignments, 0] = first_source_start_byte
            out[n_alignments, 1] = last_source_end_byte
            out[n_alignments, 2] = first_source_index
            out[n_alignments, 3] = last_source_index
            out[n_alignments, 4] = first_target_start_byte
            out[n_alignments, 5] = last_target_end_byte
            out[n_alignments, 6] = first_target_index
            out[n_alignments, 7] = last_target_index
            out[n_alignments, 8] = matches_in_current_alignment
            n_alignments += 1
    return out, n_alignments


@njit(nogil=True, cache=True)
def merge_with_previous(al, n, merge_byte, merge_ngram, window_size, multiplier):
    out = np.empty((n, NCOL), np.int32)
    n_alignments = 0
    if n == 0:
        return out, 0
    max_ngram_distance = window_size if merge_ngram else 0
    max_source_distance = 0
    max_target_distance = 0
    prev = np.empty(NCOL, np.int64)
    for c in range(NCOL):
        prev[c] = al[0, c]
    last_index = n - 1
    for index in range(1, n):
        merged = False
        if merge_byte:
            distance_value = int(math.floor(float(prev[1] - prev[0]) * multiplier))
            max_source_distance = prev[1] + distance_value
            max_target_distance = prev[5] + distance_value
        source_ngram_distance = prev[3] + max_ngram_distance
        target_ngram_distance = prev[7] + max_ngram_distance
        candidate_source_start_byte = al[index, 0]
        candidate_target_start_byte = al[index, 4]
        candidate_source_index = al[index, 2]
        candidate_target_index = al[index, 6]
        if (candidate_source_start_byte <= max_source_distance
                and candidate_target_start_byte <= max_target_distance
                and candidate_target_start_byte > prev[5]):
            merged = True
        elif (candidate_source_index <= source_ngram_distance
                and candidate_target_index <= target_ngram_distance
                and candidate_target_index > prev[7]):
            merged = True
        if merged:
            prev[1] = al[index, 1]; prev[3] = al[index, 3]
            prev[5] = al[index, 5]; prev[7] = al[index, 7]
            prev[8] = prev[8] + al[index, 8]
        else:
            for c in range(NCOL):
                out[n_alignments, c] = prev[c]
            n_alignments += 1
            for c in range(NCOL):
                prev[c] = al[index, c]
        if index == last_index:
            if merged:
                for c in range(NCOL):
                    out[n_alignments, c] = prev[c]
            else:
                for c in range(NCOL):
                    out[n_alignments, c] = al[index, c]
            n_alignments += 1
    if n == 1:                       # a lone alignment is emitted unchanged
        for c in range(NCOL):
            out[0, c] = prev[c]
        n_alignments = 1
    return out, n_alignments
