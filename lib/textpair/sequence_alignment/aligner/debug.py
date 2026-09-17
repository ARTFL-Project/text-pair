"""Per-pair matching traces, produced after matching rather than during it.

`write_traces` re-derives each compared pair's matches from the loaded corpus and walks
them again, so nothing here is on the matching path: `kernels.match_passage` and
`invidx.align_source` do not know this module exists. The cost is that the walk below
must behave exactly like `kernels.match_passage`, which `tests/test_debug_trace.py`
checks by comparing the alignments the two produce.

One file per traced pair lands in `<output_path>/debug_output`, named
`<source_doc_id>_<target_doc_id>`, holding a block per emitted or rejected run and a
closing line when merging removed passages. `ngram_index` resolves an ngram's int32 key
to its text; without it the keys cannot be named and the ngram lines come out empty.

Failed runs shorter than `debug_minimum_ngrams` ngrams are replaced by one counted
summary line per pair, since they outnumber the near misses by about a hundred to one.
The default keeps runs one ngram short of `minimum_matching_ngrams`; 1 keeps everything.
`debug_pairs` narrows the trace to named pairs, which also skips re-deriving the rest.

Each block reports which tests ended the run, in evaluation order, so the first named is
the one that ended it and the rest also held.
"""
import os

import numpy as np

from . import kernels

# Which test ended a run, as a bitmask: several can hold at once.
END_TARGET_GAP = 1        # target index beyond the target gap
END_SOURCE_GAP = 2        # source index beyond the source gap, too few in window
END_WINDOW_SPARSE = 4     # past a window boundary, too few matches in the window
END_WINDOW_GAP = 8        # past a window boundary and beyond a gap
_END_NAMES = (
    (END_TARGET_GAP, "target index beyond max_gap"),
    (END_SOURCE_GAP, "source index beyond max_gap with too few matches in window"),
    (END_WINDOW_SPARSE, "past the window boundary with too few matches in window"),
    (END_WINDOW_GAP, "past the window boundary and beyond max_gap"),
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
    """Mirror of kernels.match_passage, recording a block per emitted or rejected run.

    Returns (rows, blocks, hidden): the alignments as that kernel would emit them, the
    trace of each run, and a histogram of the failed runs the floor hid.
    """
    (source_indices, source_start_bytes, source_end_bytes,
     target_indices, target_start_bytes, target_end_bytes, match_keys) = match
    window_size = params["matching_window_size"]
    max_gap_cfg = params["max_gap"]
    flex_gap = params["flex_gap"]
    min_matching = params["minimum_matching_ngrams"]
    min_in_window = params["minimum_matching_ngrams_in_window"]

    rows, blocks, hidden = [], [], {}
    last_source_position = 0
    in_alignment = False
    for match_index in range(n):
        if source_indices[match_index] < last_source_position:
            continue
        source_anchor = source_indices[match_index]
        source_window_boundary = source_anchor + window_size
        last_source_position = source_anchor
        max_source_gap = last_source_position + max_gap_cfg
        target_anchor = target_indices[match_index]
        target_window_boundary = target_anchor + window_size
        last_target_position = target_anchor
        max_target_gap = last_target_position + max_gap_cfg
        in_alignment = True
        previous_source_index = source_anchor
        first_source_start_byte = source_start_bytes[match_index]
        first_source_index = source_indices[match_index]
        first_target_start_byte = target_start_bytes[match_index]
        first_target_index = target_indices[match_index]
        matches_in_current_alignment = 1
        matches_in_current_window = 1
        last_source_end_byte = source_end_bytes[match_index]
        last_source_index = source_indices[match_index]
        last_target_end_byte = target_end_bytes[match_index]
        last_target_index = target_indices[match_index]
        max_gap = max_gap_cfg
        matching_window_size = window_size
        keys = [match_keys[match_index]]
        reason = 0
        for j in range(match_index + 1, n):
            source_index = source_indices[j]
            target_index = target_indices[j]
            if source_index == previous_source_index:
                continue
            if target_index > max_target_gap or target_index <= last_target_position:
                if source_index <= max_source_gap:
                    continue
                else:
                    in_alignment = False
                    reason |= END_TARGET_GAP
            if source_index > max_source_gap and matches_in_current_window < min_in_window:
                in_alignment = False
                reason |= END_SOURCE_GAP
            if source_index > source_window_boundary or target_index > target_window_boundary:
                if matches_in_current_window < min_in_window:
                    in_alignment = False
                    reason |= END_WINDOW_SPARSE
                else:
                    if source_index > max_source_gap or target_index > max_target_gap:
                        in_alignment = False
                        reason |= END_WINDOW_GAP
                    else:
                        source_anchor = source_index
                        source_window_boundary = source_anchor + matching_window_size
                        target_anchor = target_index
                        target_window_boundary = target_anchor + matching_window_size
                        matches_in_current_window = 0
            if not in_alignment:
                emit = matches_in_current_alignment >= min_matching
                if emit:
                    rows.append((first_source_start_byte, last_source_end_byte,
                                     first_source_index, last_source_index,
                                     first_target_start_byte, last_target_end_byte,
                                     first_target_index, last_target_index,
                                     matches_in_current_alignment))
                if emit or len(keys) >= floor:
                    blocks.append((emit, first_source_start_byte, last_source_end_byte,
                                       first_source_index, last_source_index,
                                       first_target_start_byte, last_target_end_byte,
                                       first_target_index, last_target_index, keys, reason))
                else:
                    hidden[len(keys)] = hidden.get(len(keys), 0) + 1
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
            last_source_end_byte = source_end_bytes[j]; last_source_index = source_index
            last_target_end_byte = target_end_bytes[j]; last_target_index = target_index
            keys.append(match_keys[j])
        if in_alignment and matches_in_current_alignment >= min_matching:
            rows.append((first_source_start_byte, last_source_end_byte,
                             first_source_index, last_source_index,
                             first_target_start_byte, last_target_end_byte,
                             first_target_index, last_target_index,
                             matches_in_current_alignment))
            # This run reached the end of the matches; record a block so every
            # alignment in the output appears in the trace.
            blocks.append((True, first_source_start_byte, last_source_end_byte,
                               first_source_index, last_source_index,
                               first_target_start_byte, last_target_end_byte,
                               first_target_index, last_target_index, keys, reason))
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
        if shared.shape[0] / source_keys.shape[0] * 100 > dup_threshold:
            continue                                   # a duplicate, never matched
        match, n = corpus.matches(source, target)
        if not n:
            continue
        rows, blocks, hidden = _walk(match, n, params, floor)
        parts = [_render(block, ngram_index) for block in blocks]
        if merging and rows:
            alignments = np.array(rows, np.int32)
            _merged, after = kernels.merge_with_previous(
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
