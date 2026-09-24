"""Per-pair matching traces, produced after matching rather than during it.

`write_traces` re-derives each compared pair's matches from the loaded corpus and walks
them again, so nothing here is on the matching path: `matching.align_pair` and
`inverted_index.align_source` do not know this module exists. The cost is that `_walk` and
`pair_rows` must behave exactly like `matching.match_passage` and `matching.align_pair`,
which `tests/check_tracing.py` checks by comparing the alignments they produce.

One file per traced pair lands in `<output_path>/debug_output`, named
`<source_doc_id>_<target_doc_id>`, holding a block per kept or rejected passage and
closing lines when merging or coalescing removed passages. `ngram_index` resolves an
ngram's int32 key to its text; without it the keys cannot be named and the ngram lines
come out empty.

Rejected passages shorter than `debug_minimum_ngrams` ngrams are replaced by one counted
summary line per pair, since they outnumber the near misses by about a hundred to one.
The default keeps runs one ngram short of `minimum_matching_ngrams`; 1 keeps everything.
`debug_pairs` narrows the trace to named pairs, which also skips re-deriving the rest.

The matcher builds chains longest-first and splits each where it thins out, so a block is
one segment of one chain. Its `Run ended:` line names where the segment stopped -- a gap,
a sparse window, the end of the chain -- and, when the segment was rejected, why.
"""
import os

import numpy as np

from . import matching

# Why a passage ends where it does, and why a rejected one was rejected, as a bitmask:
# several can hold at once.
END_GAP = 1             # the step to the next match is beyond the gap allowance
END_WINDOW_SPARSE = 2   # past a window boundary with too few matches in the window
END_OF_CHAIN = 4        # no further match could extend the chain
TRUNCATED = 8           # the chain stopped at a match an earlier passage had taken
DROP_SHORT = 16         # fewer than minimum_matching_ngrams matches
DROP_COVERED = 32       # both stretches already covered by a passage that was kept
ANCHORED_SOURCE = 64    # found by the anchored scan walking the source
ANCHORED_TARGET = 128   # found by the anchored scan walking the target
_END_NAMES = (
    (END_GAP, "the next match is beyond max_gap in one of the documents"),
    (END_WINDOW_SPARSE, "past the window boundary with too few matches in window"),
    (END_OF_CHAIN, "the chain could not be extended"),
    (TRUNCATED, "the chain ran into matches an earlier passage had taken"),
    (DROP_SHORT, "fewer than minimum_matching_ngrams matches"),
    (DROP_COVERED, "both stretches are already covered by a passage that was kept"),
    (ANCHORED_SOURCE, "found by the anchored scan walking the source"),
    (ANCHORED_TARGET, "found by the anchored scan walking the target"),
)


def load_ngram_index(path):
    """Read an `ngram<TAB>key` index, as written to `index/index.tab` by generation."""
    index = {}
    with open(path, "rb") as handle:
        for raw in handle:
            fields = raw.decode("utf8", "replace").strip().split("\t")
            if len(fields) != 2:
                continue
            try:
                index[int(fields[1])] = fields[0]
            except ValueError:
                continue
    return index


def parse_pairs(spec):
    """`source[:target]` entries, comma separated. Empty means trace every pair.

    A bare `source` selects all of that document's pairs. Returns None for no filter,
    else {source_doc: set(target_docs) or None for all}.
    """
    if not spec.strip():
        return None
    selected = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        source, _, target = entry.partition(":")
        source, target = source.strip(), target.strip()
        if not target or target == "*":
            selected[source] = None
        elif selected.get(source, ()) is not None:
            selected.setdefault(source, set()).add(target)
    return selected or None


class Corpus:
    """The loaded ngram arrays, with the lookups the trace needs off the hot path."""

    def __init__(self, key_offsets, ngram_keys, position_offsets, ngram_indices,
                 start_bytes, end_bytes):
        self.key_offsets = key_offsets
        self.ngram_keys = ngram_keys
        self.position_offsets = position_offsets
        self.ngram_indices = ngram_indices
        self.start_bytes = start_bytes
        self.end_bytes = end_bytes

    def keys(self, doc):
        return self.ngram_keys[self.key_offsets[doc]:self.key_offsets[doc + 1]]

    def positions(self, doc, key):
        """(index, start_byte, end_byte) columns for one key of one document."""
        base = int(self.key_offsets[doc])
        slot = base + int(np.searchsorted(self.keys(doc), key))
        lo, hi = int(self.position_offsets[slot + doc]), int(self.position_offsets[slot + doc + 1])
        return self.ngram_indices[lo:hi], self.start_bytes[lo:hi], self.end_bytes[lo:hi]

    def matches(self, source, target, limit=0):
        """The pair's matches, ordered as the matcher requires, and each one's key.

        The cross product of every shared key's positions, sorted by (source index,
        target index). That pair is unique per match, so the order is too.

        Returns (columns, count, stopped_early). `limit` stops once another shared key
        would take the count past it, keeping whole keys -- so the count usually lands
        below the limit rather than on it, which is why whether it stopped is reported
        rather than inferred. The result is then a subset rather than all of the pair's
        matches, which is still valid input to either implementation -- neither requires
        the match list to be complete -- so a caller comparing the two can bound the work
        without weakening the comparison. `tests/check_tracing.py` is why it exists;
        write_traces leaves it at 0, since a trace has to describe the whole pair.
        """
        shared = np.intersect1d(self.keys(source), self.keys(target),
                                assume_unique=True)
        cols = []
        stopped_early = False
        for key in shared:
            source_idx, source_start, source_end = self.positions(source, key)
            target_idx, target_start, target_end = self.positions(target, key)
            if limit and cols and len(cols) + source_idx.shape[0] * target_idx.shape[0] > limit:
                stopped_early = True
                break
            for a in range(source_idx.shape[0]):
                for b in range(target_idx.shape[0]):
                    cols.append((source_idx[a], source_start[a], source_end[a],
                                 target_idx[b], target_start[b], target_end[b], key))
        if not cols:
            return None, 0, False
        cols.sort(key=lambda row: (int(row[0]), int(row[3])))
        return list(zip(*cols)), len(cols), stopped_early


def _walk(match, n, params, floor, source_first=True):
    """Mirror of matching.match_passage, recording a block per kept or rejected passage.

    Returns (rows, blocks, hidden, longest): the alignments as that kernel would emit
    them, the trace of each passage, a histogram of the rejected ones the floor hid, and
    the longest chain.
    """
    (source_indices, source_start_bytes, source_end_bytes,
     target_indices, target_start_bytes, target_end_bytes, match_keys) = match
    window_size = params["matching_window_size"]
    max_gap = params["max_gap"]
    flex_gap = params["flex_gap"]
    min_matching = params["minimum_matching_ngrams"]
    min_in_window = params["minimum_matching_ngrams_in_window"]
    max_link = matching.link_bound(window_size, max_gap, flex_gap, min_matching)

    rows, blocks, hidden = [], [], {}
    best = [1] * n
    parent = [-1] * n
    used = [False] * n
    window_start = 0
    longest = 1
    for b in range(n):
        source_b = int(source_indices[b])
        target_b = int(target_indices[b])
        while int(source_indices[window_start]) < source_b - max_link:
            window_start += 1
        best_b, parent_b, best_step, best_near, best_first = 1, -1, 0, 0, 0
        for a in range(window_start, b):
            source_a = int(source_indices[a])
            if source_a == source_b:
                break
            target_step = target_b - int(target_indices[a])
            if target_step <= 0 or target_step > max_link:
                continue
            candidate = best[a] + 1
            source_step = source_b - source_a
            step = source_step + target_step
            near = min(source_step, target_step)
            first_step = source_step if source_first else target_step
            if candidate > best_b or (candidate == best_b and (step, near, first_step)
                                      < (best_step, best_near, best_first)):
                best_b, parent_b, best_step, best_near, best_first = (
                    candidate, a, step, near, first_step)
        best[b], parent[b] = best_b, parent_b
        longest = max(longest, best_b)
    if longest < min_matching:
        return rows, blocks, hidden, longest

    def pair_key(i):
        source, target = int(source_indices[i]), int(target_indices[i])
        low, high = min(source, target), max(source, target)
        return (low << 31) | high, source if source_first else target

    # Ends longest first, then by coordinate pair: the kernel's counting sort followed by
    # its per-length insertion sort, both stable, come to the same order.
    ends = sorted((b for b in range(n) if best[b] >= min_matching),
                  key=lambda b: (-best[b], pair_key(b)))
    spans = []

    def record(first, last, keys, reason):
        """Offer chain[first:last] to the coverage test and trace the outcome."""
        source_lo, source_hi = int(source_indices[first]), int(source_indices[last])
        target_lo, target_hi = int(target_indices[first]), int(target_indices[last])
        covered_source = any(source_lo <= hi and lo <= source_hi
                             for lo, hi, _, _ in spans)
        covered_target = any(target_lo <= hi and lo <= target_hi
                             for _, _, lo, hi in spans)
        emit = not (covered_source and covered_target)
        if emit:
            spans.append((source_lo, source_hi, target_lo, target_hi))
            rows.append((int(source_start_bytes[first]), int(source_end_bytes[last]),
                         source_lo, source_hi,
                         int(target_start_bytes[first]), int(target_end_bytes[last]),
                         target_lo, target_hi, len(keys)))
        else:
            reason |= DROP_COVERED
        if emit or len(keys) >= floor:
            blocks.append((emit, int(source_start_bytes[first]),
                           int(source_end_bytes[last]), source_lo, source_hi,
                           int(target_start_bytes[first]), int(target_end_bytes[last]),
                           target_lo, target_hi, keys, reason))
        else:
            hidden[len(keys)] = hidden.get(len(keys), 0) + 1

    for end in ends:
        if used[end]:
            continue
        chain = []
        at = end
        while at >= 0 and not used[at]:
            chain.append(at)
            at = parent[at]
        truncated = TRUNCATED if at >= 0 else 0
        if len(chain) < min_matching:
            used[end] = True
            keys = [match_keys[i] for i in reversed(chain)]
            if len(keys) >= floor:
                first, last = chain[-1], chain[0]
                blocks.append((False, int(source_start_bytes[first]),
                               int(source_end_bytes[last]),
                               int(source_indices[first]), int(source_indices[last]),
                               int(target_start_bytes[first]),
                               int(target_end_bytes[last]),
                               int(target_indices[first]), int(target_indices[last]),
                               keys, DROP_SHORT | truncated))
            else:
                hidden[len(keys)] = hidden.get(len(keys), 0) + 1
            continue
        chain.reverse()
        for i in chain:
            used[i] = True
        segment_start = 0
        anchor = 0
        in_window = 1
        in_segment = 1
        gap_allowance = max_gap
        window = window_size
        for t in range(1, len(chain) + 1):
            reason = 0
            cut = t == len(chain)
            if cut:
                reason = END_OF_CHAIN | truncated
            else:
                source_t = int(source_indices[chain[t]])
                target_t = int(target_indices[chain[t]])
                if (source_t - int(source_indices[chain[t - 1]]) > gap_allowance
                        or target_t - int(target_indices[chain[t - 1]]) > gap_allowance):
                    cut, reason = True, END_GAP
                elif (source_t > int(source_indices[chain[anchor]]) + window
                      or target_t > int(target_indices[chain[anchor]]) + window):
                    if in_window < min_in_window:
                        cut, reason = True, END_WINDOW_SPARSE
                    else:
                        anchor = t
                        in_window = 0
            if cut:
                count = t - segment_start
                keys = [match_keys[i] for i in chain[segment_start:t]]
                if count >= min_matching:
                    record(chain[segment_start], chain[t - 1], keys, reason)
                elif len(keys) >= floor:
                    first, last = chain[segment_start], chain[t - 1]
                    blocks.append((False, int(source_start_bytes[first]),
                                   int(source_end_bytes[last]),
                                   int(source_indices[first]), int(source_indices[last]),
                                   int(target_start_bytes[first]),
                                   int(target_end_bytes[last]),
                                   int(target_indices[first]), int(target_indices[last]),
                                   keys, reason | DROP_SHORT))
                else:
                    hidden[len(keys)] = hidden.get(len(keys), 0) + 1
                if t == len(chain):
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
    # The kernel hands its rows back in source order; passages are found longest-first,
    # which is not that order. `blocks` stays in the order the matcher produced them,
    # since that is what the trace is for.
    rows.sort(key=lambda row: (row[2], row[6]))
    return rows, blocks, hidden, longest


def pair_rows(match, n, params, floor, source_first=True):
    """Mirror of matching.align_pair: the chains and both anchored scans, each merged,
    then coalesced.

    Returns (rows, blocks, hidden, merged_away, coalesced), the last two counting the
    passages merging and coalescing folded into others.
    """
    (source_indices, source_start_bytes, source_end_bytes,
     target_indices, target_start_bytes, target_end_bytes, match_keys) = match
    rows, blocks, hidden, longest = _walk(match, n, params, floor, source_first)
    if (longest < params["minimum_matching_ngrams"]
            and longest < params["minimum_matching_ngrams_in_window"]):
        return [], blocks, hidden, 0, 0
    sets = [rows]
    for mirrored in (False, True):
        runs = []
        for members in _anchored_scan(source_indices, target_indices, n, params,
                                      mirrored):
            first, last = members[0], members[-1]
            runs.append((int(source_start_bytes[first]), int(source_end_bytes[last]),
                         int(source_indices[first]), int(source_indices[last]),
                         int(target_start_bytes[first]), int(target_end_bytes[last]),
                         int(target_indices[first]), int(target_indices[last]),
                         len(members)))
            blocks.append((True,) + runs[-1][:8] + ([match_keys[i] for i in members],
                          ANCHORED_TARGET if mirrored else ANCHORED_SOURCE))
        sets.append(sorted(runs, key=lambda row: (row[2], row[6])))
    merged_away = 0
    if (params["merge_passages_on_byte_distance"]
            or params["merge_passages_on_ngram_distance"]):
        for k, rows in enumerate(sets):
            if not rows:
                continue
            merged, after = matching.merge_passages(
                np.array(rows, np.int32), len(rows),
                params["merge_passages_on_byte_distance"],
                params["merge_passages_on_ngram_distance"],
                params["matching_window_size"], params["passage_distance_multiplier"])
            merged_away += len(rows) - after
            sets[k] = [tuple(int(v) for v in merged[i]) for i in range(after)]
    combined = [row for rows in sets for row in rows]
    rows = _coalesce(combined)
    return rows, blocks, hidden, merged_away, len(combined) - len(rows)


def _anchored_scan(source_indices, target_indices, n, params, mirrored):
    """Mirror of matching._anchored_scan. Yields each run's matches, in order."""
    window_size = params["matching_window_size"]
    max_gap = params["max_gap"]
    flex_gap = params["flex_gap"]
    min_matching = params["minimum_matching_ngrams"]
    min_in_window = params["minimum_matching_ngrams_in_window"]
    if mirrored:
        walked_of, other_of = target_indices, source_indices
        order = sorted(range(n), key=lambda i: (int(target_indices[i]),
                                                int(source_indices[i])))
    else:
        walked_of, other_of = source_indices, target_indices
        order = range(n)
    walked_at = [int(walked_of[i]) for i in order]
    other_at = [int(other_of[i]) for i in order]
    resume = 0
    for anchor_rank in range(n):
        walked, other = walked_at[anchor_rank], other_at[anchor_rank]
        if walked < resume:
            continue
        walked_boundary, other_boundary = walked + window_size, other + window_size
        last_walked, last_other = walked, other
        walked_limit, other_limit = walked + max_gap, other + max_gap
        previous_walked = walked
        members = [anchor_rank]
        in_run = True
        in_alignment = in_window = 1
        gap, window = max_gap, window_size
        for rank in range(anchor_rank + 1, n):
            w, o = walked_at[rank], other_at[rank]
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
                    walked_boundary, other_boundary = w + window, o + window
                    in_window = 0
            if not in_run:
                break
            last_walked, walked_limit = w, w + gap
            last_other, other_limit = o, o + gap
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
            members.append(rank)
        if in_alignment >= min_matching:
            yield [order[r] for r in members]
        resume = last_walked + 1 if not in_run else last_walked


def _coalesce(rows):
    """Mirror of matching._coalesce."""
    rows = sorted(rows, key=lambda row: (row[2], row[6]))
    parent = list(range(len(rows)))

    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if rows[j][2] > rows[i][3]:
                break
            if rows[j][6] <= rows[i][7] and rows[i][6] <= rows[j][7]:
                a, b = root(i), root(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    merged = {}
    for i, row in enumerate(rows):
        r = root(i)
        if r not in merged:
            merged[r] = list(row)
            continue
        m = merged[r]
        if row[2] < m[2]:
            m[0], m[2] = row[0], row[2]
        if row[3] > m[3]:
            m[1], m[3] = row[1], row[3]
        if row[6] < m[6]:
            m[4], m[6] = row[4], row[6]
        if row[7] > m[7]:
            m[5], m[7] = row[5], row[7]
        m[8] = max(m[8], row[8])
    return sorted((tuple(m) for m in merged.values()), key=lambda row: (row[2], row[6]))


def _why(reason):
    if not reason:
        return "ran out of matches"
    return ", ".join(name for bit, name in _END_NAMES if reason & bit)


def _render(block, ngram_index):
    (emit, source_start_byte, source_end_byte, source_first, source_last,
     target_start_byte, target_end_byte, target_first, target_last,
     keys, reason) = block
    names = [ngram_index.get(int(k), "") for k in keys]
    return (f"\n\n## {'MATCH' if emit else 'FAILED MATCH'} ##\n"
            f"Source byte range: {source_start_byte}-{source_end_byte}\n"
            f"Source matching index range: {source_first}-{source_last}\n"
            f"Target byte range: {target_start_byte}-{target_end_byte}\n"
            f"Target matching index range: {target_first}-{target_last}\n"
            f"Matching ngrams: {' '.join(names)}\n"
            f"Number of matching ngrams: {len(names)}\n"
            f"Run ended: {_why(reason)}")


def _pairs(docs, n_sources, same_doc, pair_filter):
    """The (source slot, target slot) pairs align_source would have compared."""
    separate = same_doc.shape[0] != 0
    for source in range(n_sources):
        if pair_filter is not None and docs[source] not in pair_filter:
            continue
        wanted = pair_filter.get(docs[source]) if pair_filter is not None else None
        targets = (range(n_sources, len(docs)) if separate
                   else range(source + 1, len(docs)))
        for target in targets:
            if separate and target == int(same_doc[source]):
                continue
            if wanted is not None and docs[target] not in wanted:
                continue
            yield source, target


def write_traces(output_path, docs, corpus, params, same_doc, n_sources, ngram_index,
                 pair_filter, document_rank=None):
    """Trace every compared pair, or only those `pair_filter` names. Returns the count.
    `document_rank` is run_match's."""
    directory = os.path.join(output_path, "debug_output")
    os.makedirs(directory, exist_ok=True)
    floor = params["debug_minimum_ngrams"]
    min_in_docs = params["minimum_matching_ngrams_in_docs"]
    dup_threshold = params["duplicate_threshold"]
    written = 0
    for source, target in _pairs(docs, n_sources, same_doc, pair_filter):
        source_keys = corpus.keys(source)
        shared = np.intersect1d(source_keys, corpus.keys(target), assume_unique=True)
        if shared.shape[0] < min_in_docs:
            continue
        smaller = min(source_keys.shape[0], corpus.keys(target).shape[0])
        if shared.shape[0] / smaller * 100 > dup_threshold:
            continue                                   # a duplicate, never matched
        match, n, _stopped = corpus.matches(source, target)
        if not n:
            continue
        source_first = (document_rank is None
                        or document_rank[source] < document_rank[target])
        _rows, blocks, hidden, merged_away, coalesced = pair_rows(match, n, params, floor,
                                                                  source_first)
        parts = [_render(block, ngram_index) for block in blocks]
        if merged_away:
            parts.append(f"\n\n{merged_away} passage(s) merged with previous passage")
        if coalesced:
            parts.append(f"\n\n{coalesced} passage(s) coalesced with an overlapping "
                         "passage")
        if hidden:
            summary = ", ".join(f"{hidden[n_keys]} at {n_keys} ngram(s)"
                                for n_keys in sorted(hidden))
            parts.append(f"\n\n## {sum(hidden.values())} FAILED RUN(S) NOT SHOWN ##\n"
                         f"below the {floor}-ngram trace floor: {summary}")
        if not parts:
            continue
        name = f"{docs[source]}_{docs[target]}"
        with open(os.path.join(directory, name), "w", encoding="utf8") as handle:
            handle.write("".join(parts))
        written += 1
    return written
