"""numba ports of compareNgrams' matchPassage (main.go:660-765) and
mergeWithPrevious (main.go:768-829).

Kernels take int32/int64 arrays and scalars only, nogil and cached.
An alignment row is int32[9]: s_sb, s_eb, s_sidx, s_eidx, t_sb, t_eb, t_sidx,
t_eidx, total_matching_ngrams.
"""
import math
import numpy as np
from numba import njit

NCOL = 9


@njit(nogil=True, cache=True)
def _grow(out):
    new = np.empty((out.shape[0] * 2, NCOL), np.int32)
    new[: out.shape[0]] = out
    return new


@njit(nogil=True, cache=True)
def match_passage(sidx, ssb, seb, tidx, tsb, teb, n,
                  window_size, max_gap_cfg, flex_gap, min_matching, min_in_window):
    # main.go:660-765. Matches must be sorted by (source.index, target.index).
    out = np.empty((64, NCOL), np.int32)
    cnt = 0
    last_source_position = 0            # main.go:663
    in_alignment = False
    for match_index in range(n):
        if sidx[match_index] < last_source_position:   # 667
            continue
        source_anchor = sidx[match_index]
        source_window_boundary = source_anchor + window_size
        last_source_position = source_anchor
        max_source_gap = last_source_position + max_gap_cfg
        target_anchor = tidx[match_index]
        target_window_boundary = target_anchor + window_size
        last_target_position = target_anchor
        max_target_gap = last_target_position + max_gap_cfg
        in_alignment = True
        previous_source_index = source_anchor
        first_s_sb = ssb[match_index]; first_s_idx = sidx[match_index]     # firstMatch (680)
        first_t_sb = tsb[match_index]; first_t_idx = tidx[match_index]
        matches_in_current_alignment = 1
        matches_in_current_window = 1
        last_s_eb = seb[match_index]; last_s_idx = sidx[match_index]       # lastMatch (683)
        last_t_eb = teb[match_index]; last_t_idx = tidx[match_index]
        max_gap = max_gap_cfg
        matching_window_size = window_size
        for j in range(match_index + 1, n):                                # 692
            s_index = sidx[j]
            t_index = tidx[j]
            if s_index == previous_source_index:                           # 695
                continue
            if t_index > max_target_gap or t_index <= last_target_position:   # 698
                # Go: nextIndex = pos + matchIndex + 1 == j, i.e. the CURRENT match,
                # so matches[nextIndex].source.index == s_index and the bound is always true (701)
                if s_index <= max_source_gap:
                    continue
                else:
                    in_alignment = False
            if s_index > max_source_gap and matches_in_current_window < min_in_window:   # 707
                in_alignment = False
            if s_index > source_window_boundary or t_index > target_window_boundary:    # 710
                if matches_in_current_window < min_in_window:
                    in_alignment = False
                else:
                    if s_index > max_source_gap or t_index > max_target_gap:
                        in_alignment = False
                    else:
                        source_anchor = s_index
                        source_window_boundary = source_anchor + matching_window_size
                        target_anchor = t_index
                        target_window_boundary = target_anchor + matching_window_size
                        matches_in_current_window = 0
            if not in_alignment:                                           # 725
                if matches_in_current_alignment >= min_matching:
                    if cnt == out.shape[0]:
                        out = _grow(out)
                    out[cnt, 0] = first_s_sb; out[cnt, 1] = last_s_eb
                    out[cnt, 2] = first_s_idx; out[cnt, 3] = last_s_idx
                    out[cnt, 4] = first_t_sb; out[cnt, 5] = last_t_eb
                    out[cnt, 6] = first_t_idx; out[cnt, 7] = last_t_idx
                    out[cnt, 8] = matches_in_current_alignment
                    cnt += 1
                last_source_position = last_s_idx + 1                      # 734
                break
            last_source_position = s_index                                 # 737
            max_source_gap = last_source_position + max_gap
            last_target_position = t_index
            max_target_gap = last_target_position + max_gap
            previous_source_index = s_index
            matches_in_current_window += 1
            matches_in_current_alignment += 1
            if flex_gap:                                                   # 744
                if matches_in_current_alignment == min_matching:
                    max_gap += min_matching
                    matching_window_size += min_matching
                elif matches_in_current_alignment > min_matching:
                    if max_gap < window_size:
                        max_gap += 1
                        matching_window_size += 1
            last_s_eb = seb[j]; last_s_idx = s_index                       # 755
            last_t_eb = teb[j]; last_t_idx = t_index
        if in_alignment and matches_in_current_alignment >= min_matching:  # 760
            if cnt == out.shape[0]:
                out = _grow(out)
            out[cnt, 0] = first_s_sb; out[cnt, 1] = last_s_eb
            out[cnt, 2] = first_s_idx; out[cnt, 3] = last_s_idx
            out[cnt, 4] = first_t_sb; out[cnt, 5] = last_t_eb
            out[cnt, 6] = first_t_idx; out[cnt, 7] = last_t_idx
            out[cnt, 8] = matches_in_current_alignment
            cnt += 1
    return out, cnt


@njit(nogil=True, cache=True)
def merge_with_previous(al, n, merge_byte, merge_ngram, window_size, multiplier):
    # main.go:768-829
    out = np.empty((n, NCOL), np.int32)
    cnt = 0
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
        if merge_byte:                                                     # 787
            distance_value = int(math.floor(float(prev[1] - prev[0]) * multiplier))
            max_source_distance = prev[1] + distance_value
            max_target_distance = prev[5] + distance_value
        source_ngram_distance = prev[3] + max_ngram_distance
        target_ngram_distance = prev[7] + max_ngram_distance
        c_s_sb = al[index, 0]; c_t_sb = al[index, 4]
        c_s_sidx = al[index, 2]; c_t_sidx = al[index, 6]
        if c_s_sb <= max_source_distance and c_t_sb <= max_target_distance and c_t_sb > prev[5]:   # 795
            merged = True
        elif c_s_sidx <= source_ngram_distance and c_t_sidx <= target_ngram_distance and c_t_sidx > prev[7]:  # 802
            merged = True
        if merged:
            prev[1] = al[index, 1]; prev[3] = al[index, 3]
            prev[5] = al[index, 5]; prev[7] = al[index, 7]
            prev[8] = prev[8] + al[index, 8]
        else:                                                              # 809
            for c in range(NCOL):
                out[cnt, c] = prev[c]
            cnt += 1
            for c in range(NCOL):
                prev[c] = al[index, c]
        if index == last_index:                                            # 813
            if merged:
                for c in range(NCOL):
                    out[cnt, c] = prev[c]
            else:
                for c in range(NCOL):
                    out[cnt, c] = al[index, c]
            cnt += 1
    if n == 1:                                                             # 821: previousAlignment != zero && len(merged)==0
        for c in range(NCOL):
            out[0, c] = prev[c]
        cnt = 1
    return out, cnt
