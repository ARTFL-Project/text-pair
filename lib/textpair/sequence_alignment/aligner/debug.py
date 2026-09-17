"""Per-pair matching traces, produced after matching rather than during it.

`write_traces` re-derives each compared pair's matches from the loaded corpus and walks
them again, so nothing here is on the matching path: `kernels.match_passage` and
`invidx.align_source` do not know this module exists. The cost is that the walk below
must behave exactly like `kernels.match_passage`, which `tests/test_debug_trace.py`
checks by comparing the alignments the two produce.

One file per traced pair lands in `<output_path>/debug_output`, named
`<source_doc_id>_<target_doc_id>`, holding a block per kept or rejected passage and a
closing line when merging removed passages. `ngram_index` resolves an ngram's int32 key
to its text; without it the keys cannot be named and the ngram lines come out empty.

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

from . import kernels

# Why a passage ends where it does, and why a rejected one was rejected, as a bitmask:
# several can hold at once.
END_GAP = 1             # the step to the next match is beyond the gap allowance
END_WINDOW_SPARSE = 2   # past a window boundary with too few matches in the window
END_OF_CHAIN = 4        # no further match could extend the chain
TRUNCATED = 8           # the chain stopped at a match an earlier passage had taken
DROP_SHORT = 16         # fewer than minimum_matching_ngrams matches
DROP_COVERED = 32       # both stretches already covered by a passage that was kept
_END_NAMES = (
    (END_GAP, "the next match is beyond max_gap in one of the documents"),
    (END_WINDOW_SPARSE, "past the window boundary with too few matches in window"),
    (END_OF_CHAIN, "the chain could not be extended"),
    (TRUNCATED, "the chain ran into matches an earlier passage had taken"),
    (DROP_SHORT, "fewer than minimum_matching_ngrams matches"),
    (DROP_COVERED, "both stretches are already covered by a passage that was kept"),
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

    def matches(self, source, target):
        """The pair's matches, ordered as the matcher requires, and each one's key.

        The cross product of every shared key's positions, sorted by (source index,
        target index). That pair is unique per match, so the order is too.
        """
        shared = np.intersect1d(self.keys(source), self.keys(target),
                                assume_unique=True)
        cols = []
        for key in shared:
            source_idx, source_start, source_end = self.positions(source, key)
            target_idx, target_start, target_end = self.positions(target, key)
            for a in range(source_idx.shape[0]):
                for b in range(target_idx.shape[0]):
                    cols.append((source_idx[a], source_start[a], source_end[a],
                                 target_idx[b], target_start[b], target_end[b], key))
        if not cols:
            return None, 0
        cols.sort(key=lambda row: (int(row[0]), int(row[3])))
        return list(zip(*cols)), len(cols)


def _walk(match, n, params, floor):
    """Mirror of kernels.match_passage, recording a block per kept or rejected passage.

    Returns (rows, blocks, hidden): the alignments as that kernel would emit them, the
    trace of each passage, and a histogram of the rejected ones the floor hid.
    """
    (source_indices, source_start_bytes, source_end_bytes,
     target_indices, target_start_bytes, target_end_bytes, match_keys) = match
    window_size = params["matching_window_size"]
    max_gap = params["max_gap"]
    flex_gap = params["flex_gap"]
    min_matching = params["minimum_matching_ngrams"]
    min_in_window = params["minimum_matching_ngrams_in_window"]
    max_link = kernels.link_bound(window_size, max_gap, flex_gap, min_matching)

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
        best_b, parent_b, best_step, best_near = 1, -1, 0, 0
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
            if candidate > best_b or (candidate == best_b
                                      and (step, near) < (best_step, best_near)):
                best_b, parent_b, best_step, best_near = candidate, a, step, near
        best[b], parent[b] = best_b, parent_b
        longest = max(longest, best_b)
    if longest < min_matching:
        return rows, blocks, hidden

    def pair_key(i):
        source, target = int(source_indices[i]), int(target_indices[i])
        low, high = min(source, target), max(source, target)
        return (low << 31) | high

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
    return rows, blocks, hidden


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
                 pair_filter):
    """Trace every compared pair, or only those `pair_filter` names. Returns the count."""
    directory = os.path.join(output_path, "debug_output")
    os.makedirs(directory, exist_ok=True)
    floor = params["debug_minimum_ngrams"]
    min_in_docs = params["minimum_matching_ngrams_in_docs"]
    dup_threshold = params["duplicate_threshold"]
    merging = (params["merge_passages_on_byte_distance"]
               or params["merge_passages_on_ngram_distance"])
    written = 0
    for source, target in _pairs(docs, n_sources, same_doc, pair_filter):
        source_keys = corpus.keys(source)
        shared = np.intersect1d(source_keys, corpus.keys(target), assume_unique=True)
        if shared.shape[0] < min_in_docs:
            continue
        smaller = min(source_keys.shape[0], corpus.keys(target).shape[0])
        if shared.shape[0] / smaller * 100 > dup_threshold:
            continue                                   # a duplicate, never matched
        match, n = corpus.matches(source, target)
        if not n:
            continue
        rows, blocks, hidden = _walk(match, n, params, floor)
        parts = [_render(block, ngram_index) for block in blocks]
        if merging and rows:
            alignments = np.array(rows, np.int32)
            _merged, after = kernels.merge_passages(
                alignments, len(rows), params["merge_passages_on_byte_distance"],
                params["merge_passages_on_ngram_distance"],
                params["matching_window_size"], params["passage_distance_multiplier"])
            if len(rows) > after:
                parts.append(f"\n\n{len(rows) - after} passage(s) merged with "
                             "previous passage")
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
