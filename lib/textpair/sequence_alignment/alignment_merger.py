"""Merge Overlapping Alignments"""

import contextlib
import os
import sys
from array import array
from bisect import bisect_left
from collections import defaultdict
from typing import Union

import lz4.frame
import msgspec
import orjson
from tqdm import tqdm

from textpair.utils import clean_passage

from .banality_finder import (
    _bump,
    _concatenate,
    _Framer,
    _lz4_window,
    _run_segments,
    _SHARED,
    _work_split,
)


_STEPS = ("finding groups", "refining groups", "assigning passages",
          "writing group sources", "rewriting results")


def _label(step: int, doing: str | None = None) -> str:
    return f"  {step}/{len(_STEPS)} {doing or _STEPS[step - 1]}"


def _status(text: str = "") -> None:
    """Put `text` on the progress line, for the stretches no bar covers."""
    if sys.stderr.isatty():  # in a log, each would be a stray line
        sys.stderr.write(f"\r\x1b[K{text}")
        sys.stderr.flush()


_FILL_CHUNK = 1 << 16


def _filled(typecode: str, size: int, value: int) -> array:
    """`size` copies of `value`, without building a full-size temporary to copy from."""
    values = array(typecode)
    chunk = array(typecode, (value,)) * _FILL_CHUNK
    while len(values) + _FILL_CHUNK <= size:
        values.extend(chunk)
    if len(values) < size:
        values.extend(chunk[: size - len(values)])
    return values


class _DenseMap:
    """Passage id -> int in a flat array, since passage ids are dense. -1 is absent."""

    __slots__ = ("_values",)

    def __init__(self, expected: int):
        self._values = _filled("i", max(expected, 1), -1)

    def __setitem__(self, index: int, value: int) -> None:
        values = self._values
        if index >= len(values):
            values.extend(_filled("i", index + 1 - len(values), -1))
        values[index] = value

    def __getitem__(self, index: int) -> int:
        return self._values[index] if index < len(self._values) else -1

    def __len__(self) -> int:
        return len(self._values)


class _Alignment(msgspec.Struct):
    """The fields grouping needs from a record; the rest is read back for representatives only."""

    source_doc_id: str
    source_filename: str
    source_start_byte: int
    source_end_byte: int
    target_doc_id: str
    target_start_byte: int
    target_end_byte: int
    passage_id: int = -1


_DECODE_ALIGNMENT = msgspec.json.Decoder(_Alignment).decode


_BUCKET_BITS = 10  # index granularity, 1 KiB
_LONG_SPAN = 8192  # longer spans are kept out of the buckets


class _DocumentSpans:
    """One document's target spans, searchable by containment.

    A covering span shorter than `_LONG_SPAN` starts within `_LONG_SPAN` of the
    passage's end, so only those buckets are searched; long spans always are.
    The earliest-added match wins, so search order does not matter. Buckets are
    flat (start, end, group, order) arrays.
    """

    __slots__ = ("buckets", "long_spans")

    def __init__(self):
        self.buckets: dict[int, array] = {}
        self.long_spans = array("q")

    def add(self, start_byte: int, end_byte: int, group_id: int, order: int) -> None:
        if end_byte - start_byte > _LONG_SPAN:
            target = self.long_spans
        else:
            key = start_byte >> _BUCKET_BITS
            target = self.buckets.get(key)
            if target is None:
                target = self.buckets[key] = array("q")
        target.extend((start_byte, end_byte, group_id, order))

    def containing(self, start_byte: int, end_byte: int) -> int | None:
        """Group of the first-added span covering [start_byte, end_byte], if any."""
        best_order = -1
        best_group = None
        spans = self.long_spans
        for position in range(0, len(spans), 4):
            if spans[position] <= start_byte and spans[position + 1] >= end_byte:
                order = spans[position + 3]
                if best_order < 0 or order < best_order:
                    best_order = order
                    best_group = spans[position + 2]
        buckets = self.buckets
        first = (end_byte - _LONG_SPAN) >> _BUCKET_BITS
        if first < 0:
            first = 0
        for bucket in range(first, (start_byte >> _BUCKET_BITS) + 1):
            spans = buckets.get(bucket)
            if spans is None:
                continue
            for position in range(0, len(spans), 4):
                if spans[position] <= start_byte and spans[position + 1] >= end_byte:
                    order = spans[position + 3]
                    if best_order < 0 or order < best_order:
                        best_order = order
                        best_group = spans[position + 2]
        return best_group


class AlignmentGroups:
    """Holding alignment group data"""

    __slots__ = ("group_id", "merged_target_passages", "group_map", "span_order", "found_groups")

    def __init__(self, expected: int = 0):
        self.group_id = -1
        self.merged_target_passages: dict[str, _DocumentSpans] = {}
        self.group_map = _DenseMap(expected)
        self.span_order = 0
        self.found_groups: dict[tuple[str, int, int], int] = {}

    def add_target_span(self, passage: _Alignment, group_id: int) -> None:
        """Record a passage's target span under its target document"""
        doc_id = passage.target_doc_id
        spans = self.merged_target_passages.get(doc_id)
        if spans is None:
            spans = self.merged_target_passages[doc_id] = _DocumentSpans()
        spans.add(passage.target_start_byte, passage.target_end_byte, group_id, self.span_order)
        self.span_order += 1

    def passage_group_init(self, passage: _Alignment) -> None:
        """Initialize new group"""
        self.group_id += 1
        self.add_target_span(passage, self.group_id)
        self.group_map[passage.passage_id] = self.group_id

    def passage_group_update(self, passage: _Alignment) -> None:
        """Update current group"""
        self.add_target_span(passage, self.group_id)
        self.group_map[passage.passage_id] = self.group_id

    def merge_passages(self, passages: list[_Alignment]) -> None:
        """Merge passages that are aligned to the same source passage"""
        passages.sort(
            key=lambda x: (
                x.source_start_byte,
                x.source_start_byte - x.source_end_byte,
            )
        )  # sort by smaller start byte and bigger end_byte
        current_end = None
        for passage in passages:
            if current_end is None or passage.source_start_byte >= current_end:
                self.passage_group_init(passage)
                current_end = passage.source_end_byte
            else:
                self.passage_group_update(passage)
                if passage.source_end_byte > current_end:
                    current_end = passage.source_end_byte

    def find_group(self, new_pair: _Alignment) -> bool:
        """Find group for new pair.

        Hits are cached, since spans are only added and the earliest match stays
        earliest. Misses are not: a later span can cover them.
        """
        start_byte = new_pair.source_start_byte
        end_byte = new_pair.source_end_byte
        key = (new_pair.source_doc_id, start_byte, end_byte)
        group_id = self.found_groups.get(key)
        if group_id is None:
            group_id = self.merged_target_passages[key[0]].containing(start_byte, end_byte)
            if group_id is None:
                return False
            self.found_groups[key] = group_id
        self.add_target_span(new_pair, group_id)
        self.group_map[new_pair.passage_id] = group_id
        return True


class _Spans:
    """Every passage's source file and span, by passage id, in flat arrays."""

    __slots__ = ("names", "file_ids", "starts", "ends", "_ids")

    def __init__(self):
        self.names: list[str] = []
        self.file_ids = array("i")
        self.starts = array("q")
        self.ends = array("q")
        self._ids: dict[str, int] = {}

    def append(self, filename: str, start_byte: int, end_byte: int) -> None:
        file_id = self._ids.get(filename)
        if file_id is None:
            file_id = self._ids[filename] = len(self.names)
            self.names.append(filename)
        self.file_ids.append(file_id)
        self.starts.append(start_byte)
        self.ends.append(end_byte)

    def __len__(self) -> int:
        return len(self.file_ids)


class _SourceText:
    """Passage text, holding open the file it last read from.

    Groups are written a file at a time -- four in five consecutive ones come
    from the same file -- and opening it again was over a third of what
    fetching a passage cost.
    """

    __slots__ = ("name", "handle")

    def __init__(self):
        self.name: str | None = None
        self.handle = None

    def read(self, start_byte: int, end_byte: int, filename: str) -> str:
        if filename != self.name:
            self.close()
            self.handle = open(filename, "rb")
            self.name = filename
        if start_byte < 0:
            start_byte = 0
        self.handle.seek(start_byte)
        return clean_passage(self.handle.read(end_byte - start_byte))

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
        self.handle = None
        self.name = None


class _GroupFields(msgspec.Struct):
    """Whether a record already carries the fields the rewrite sets.

    Raw so the values are never parsed -- only their presence matters, and a
    record that has them has to be rewritten rather than appended to.
    """

    group_id: Union[msgspec.Raw, msgspec.UnsetType] = msgspec.UNSET
    count: Union[msgspec.Raw, msgspec.UnsetType] = msgspec.UNSET


_DECODE_GROUP_FIELDS = msgspec.json.Decoder(_GroupFields).decode


def with_group_ids(line: bytes, groups: bytes) -> bytes:
    """`line` with its group list set, and any stale count dropped.

    `groups` comes already encoded as JSON. The rewrite adds one field to a
    record of about a hundred, so the list is spliced into the bytes rather
    than the record decoded and encoded back. orjson wrote these files and
    reproduces them byte for byte, so the result is what the round trip
    produced. A record that already carries either field goes through the
    round trip, since one has to keep its place and the other has to go.
    """
    present = _DECODE_GROUP_FIELDS(line)
    if present.group_id is msgspec.UNSET and present.count is msgspec.UNSET:
        body = line[:-1] if line.endswith(b"\n") else line
        if body.endswith(b"}") and not body.endswith(b"{}"):
            return body[:-1] + b',"group_id":' + groups + b"}\n"
    fields = orjson.loads(line)
    fields["group_id"] = orjson.loads(groups)
    # Remove old 'count' field if it exists, use final_group_counts later
    if "count" in fields:
        del fields["count"]
    return orjson.dumps(fields) + b"\n"


_READ_SIZE = 1 << 22  # 4 MiB


def read_line_batches(input_file, read_size: int = _READ_SIZE):
    """The file's lines, a block at a time: line-by-line reads through lz4 are slow."""
    remainder = b""
    while True:
        chunk = input_file.read(read_size)
        if not chunk:
            break
        lines = (remainder + chunk).split(b"\n")
        remainder = lines.pop()
        if lines:
            yield lines
    if remainder:
        yield [remainder]


def _join_parts(parts: list[str], target: str) -> None:
    """Concatenate the parts, raising if the join came out short.

    `_concatenate` ignores `cat`'s exit status, and the results file is replaced
    straight after.
    """
    expected = sum(os.path.getsize(part) for part in parts)
    _concatenate(parts, target)
    joined = os.path.getsize(target)
    if joined != expected:
        raise OSError(f"{target}: joined {joined:,} bytes where the parts held {expected:,}")


def _flush(progress, since: int) -> None:
    """Add what `_bump` left behind, which is under the stride by definition."""
    if progress is not None and since:
        with progress.get_lock():
            progress.value += since


def _count_segment(job: tuple[str, int, int, str]) -> int:
    """Records in one frame range, counted the way `read_line_batches` splits them."""
    path, offset, size, _part = job
    progress = _SHARED.get("progress")
    total = 0
    pending = 0
    last = b"\n"
    with _lz4_window(path, offset, size) as input_file:
        while True:
            chunk = input_file.read(_READ_SIZE)
            if not chunk:
                break
            found = chunk.count(b"\n")
            total += found
            pending = _bump(progress, pending + found)
            last = chunk[-1:]
    if last != b"\n":
        total += 1  # a last line without a newline is still a line
    _flush(progress, pending)
    return total


def _segment_bases(path: str, segments: list[tuple[int, int]], count: int) -> dict[int, int]:
    """The passage id each segment starts at, keyed by offset: frames don't record it."""
    counts = _run_segments(_count_segment, path, segments, count, _label(4, "locating records"))
    bases = {}
    running = 0
    for (offset, _size), found in zip(segments, counts):
        bases[offset] = running
        running += found
    return bases


def _segment_plan(results_file: str, count: int, workers: int):
    """Frame ranges for the two output passes, and where each starts; one range means run here."""
    segments = _work_split(results_file, workers)
    if len(segments) < 2:
        return segments, {}
    return segments, _segment_bases(results_file, segments, count)


def first_step_merge(results_file: str, count: int) -> tuple[_DenseMap, int, _Spans]:
    """Merge passages that are aligned to the same source passage.

    Also returns each passage's source file and span, by passage id. The two
    passes after this one need nothing else out of a record, and this one has
    already decoded every one of them.
    """
    passages: list[_Alignment] = []
    alignment_groups = AlignmentGroups(count)
    spans = _Spans()
    doc_id = None
    merged_target_passages = alignment_groups.merged_target_passages
    find_group = alignment_groups.find_group
    append_span = spans.append
    passage_id = 0
    with lz4.frame.open(results_file) as input_file:
        progress = tqdm(total=count, desc=_label(1), leave=False)
        for batch in read_line_batches(input_file):
            for line in batch:
                new_pair = _DECODE_ALIGNMENT(line)
                new_pair.passage_id = passage_id
                passage_id += 1
                append_span(new_pair.source_filename, new_pair.source_start_byte, new_pair.source_end_byte)
                source_doc_id: str = new_pair.source_doc_id
                if source_doc_id in merged_target_passages and find_group(new_pair):
                    continue

                current_doc_id = source_doc_id
                if doc_id != current_doc_id and doc_id is not None:
                    alignment_groups.merge_passages(passages)
                    passages = []
                doc_id = current_doc_id
                passages.append(new_pair)
            progress.update(len(batch))
        progress.close()

        if len(passages) > 0:
            alignment_groups.merge_passages(passages)
            passages = []

    return alignment_groups.group_map, alignment_groups.group_id + 1, spans


class _RefinedGroups:
    """Each refined group's file and intersection, by id. An empty intersection marks it gone.

    `representatives` holds the member whose record supplies the group's metadata: the last
    to join, which starts where the intersection does since members arrive in start order.
    """

    __slots__ = ("file_ids", "starts", "ends", "representatives")

    def __init__(self):
        self.file_ids = array("i")
        self.starts = array("q")
        self.ends = array("q")
        self.representatives = array("q")

    def add(self, file_id: int, start_byte: int, end_byte: int, passage_id: int) -> int:
        """Start a group over [start_byte, end_byte) and return its id"""
        self.file_ids.append(file_id)
        self.starts.append(start_byte)
        self.ends.append(end_byte)
        self.representatives.append(passage_id)
        return len(self.file_ids) - 1

    def live(self, group_id: int) -> bool:
        return self.starts[group_id] < self.ends[group_id]

    def __len__(self) -> int:
        return len(self.file_ids)


def refine_groups_strict_intersection(
    spans: _Spans, initial_group_map: _DenseMap
) -> tuple[_DenseMap, _RefinedGroups]:
    """
    Pass 2: Split initial groups based purely on direct source passage overlap
            using the shrinking intersection logic. Assigns each passage to ONE refined group.
    Returns:
        - refined_group_map: Mapping from original passage_id to the refined group_id.
        - refined_groups: Each refined group's final intersection span and source file.
    """
    # 1. Group alignments by the *initial* group_id and source file, as flat (start, end, id) arrays
    groups_data: dict[int, dict[int, array]] = defaultdict(dict)
    file_ids = spans.file_ids
    starts = spans.starts
    ends = spans.ends
    for passage_id in tqdm(range(len(spans)), desc=_label(2), leave=False):
        initial_group_id = initial_group_map[passage_id]
        if initial_group_id >= 0:
            files_in_group = groups_data[initial_group_id]
            file_id = file_ids[passage_id]
            alignments_in_file = files_in_group.get(file_id)
            if alignments_in_file is None:
                alignments_in_file = files_in_group[file_id] = array("q")
            alignments_in_file.extend((starts[passage_id], ends[passage_id], passage_id))

    # 2. Process each initial group's alignments per source file
    refined_group_map = _DenseMap(len(spans))
    refined_groups = _RefinedGroups()
    group_starts = refined_groups.starts
    group_ends = refined_groups.ends
    group_representatives = refined_groups.representatives

    for initial_group_id, files_in_group in tqdm(groups_data.items(), desc=_label(2), leave=False):
        for file_id, alignments_in_file in files_in_group.items():
            # Sort alignments within this file by start byte
            order = sorted(range(0, len(alignments_in_file), 3), key=alignments_in_file.__getitem__)

            # Track active refined groups *for this specific file* within the initial group
            active_refined_groups_for_file: list[int] = []

            for record in order:
                p_start = alignments_in_file[record]
                p_end = alignments_in_file[record + 1]
                p_id = alignments_in_file[record + 2]

                # Try to join an existing *refined* group within this file.
                # Starts only grow, so a group ending at or before this start is dropped for good.
                best_fit_group = -1
                still_live = []
                for position, current_refined_group in enumerate(active_refined_groups_for_file):
                    if group_ends[current_refined_group] <= p_start:
                        continue
                    still_live.append(current_refined_group)
                    if p_end > group_starts[current_refined_group]:
                        best_fit_group = current_refined_group  # First fit
                        still_live.extend(active_refined_groups_for_file[position + 1 :])
                        break
                active_refined_groups_for_file = still_live

                if best_fit_group >= 0:
                    # --- Join Existing Refined Group ---
                    refined_group_map[p_id] = best_fit_group
                    group_representatives[best_fit_group] = p_id
                    # Update intersection (shrinking)
                    if p_start > group_starts[best_fit_group]:
                        group_starts[best_fit_group] = p_start
                    if p_end < group_ends[best_fit_group]:
                        group_ends[best_fit_group] = p_end
                    touched = best_fit_group

                else:
                    # --- Start New Refined Group ---
                    touched = refined_groups.add(file_id, p_start, p_end, p_id)
                    active_refined_groups_for_file.append(touched)
                    refined_group_map[p_id] = touched

                # --- Cleanup: only the group just touched can have emptied ---
                if group_starts[touched] >= group_ends[touched]:
                    active_refined_groups_for_file.remove(touched)

    return refined_group_map, refined_groups


class _GroupSpans:
    """One file's refined groups, searchable by overlap.

    Held in start order with a running maximum end, so a query stops once no
    earlier group can reach it. Ties on start may sort either way: a query's
    result is a set.
    """

    __slots__ = ("starts", "ends", "ids", "max_ends")

    def __init__(self, group_ids: array, starts: array, ends: array):
        order = sorted(group_ids, key=starts.__getitem__)
        self.starts = array("q", [starts[group_id] for group_id in order])
        self.ends = array("q", [ends[group_id] for group_id in order])
        self.ids = array("i", order)
        self.max_ends = _filled("q", max(len(order), 1), 0)
        running = 0
        for position, end in enumerate(self.ends):
            if end > running:
                running = end
            self.max_ends[position] = running

    def overlapping(self, start_byte: int, end_byte: int) -> list[int]:
        """Groups whose intersection overlaps [start_byte, end_byte)"""
        ends = self.ends
        max_ends = self.max_ends
        found = []
        position = bisect_left(self.starts, end_byte) - 1
        while position >= 0 and max_ends[position] > start_byte:
            if ends[position] > start_byte:
                found.append(self.ids[position])
            position -= 1
        return found


class _GroupLists:
    """Each passage's group list, as an index into the distinct lists.

    A list is kept as JSON for the rewrite and as ids in `members` for the
    counts. The JSON goes in one buffer: each bytes object orjson returns keeps
    its whole encoding buffer alive. Index 0 is the empty list.
    """

    __slots__ = ("indexes", "blob", "bounds", "members", "offsets", "repeats")

    def __init__(self, expected: int):
        self.indexes = _filled("i", max(expected, 1), 0)
        self.blob = bytearray(b"[]")
        self.bounds = array("q", (0, 2))
        self.members = array("i")
        self.offsets = array("q", (0, 0))
        self.repeats = array("q", (0,))

    def add(self, group_ids: list[int]) -> int:
        """Record a list not seen before and return its index"""
        self.blob += orjson.dumps(group_ids)
        self.bounds.append(len(self.blob))
        self.members.extend(group_ids)
        self.offsets.append(len(self.members))
        self.repeats.append(0)
        return len(self.bounds) - 2

    def assign(self, passage_id: int, index: int) -> None:
        indexes = self.indexes
        if passage_id >= len(indexes):
            indexes.extend(_filled("i", passage_id + 1 - len(indexes), 0))
        indexes[passage_id] = index
        self.repeats[index] += 1

    def counts(self, groups: int) -> array:
        """Passages per refined group, over every list and what repeated it"""
        totals = _filled("q", max(groups, 1), 0)
        members = self.members
        offsets = self.offsets
        for index in range(1, len(self.repeats)):
            repeats = self.repeats[index]
            if repeats:
                for position in range(offsets[index], offsets[index + 1]):
                    totals[members[position]] += repeats
        return totals

    def __getitem__(self, passage_id: int) -> bytearray:
        index = self.indexes[passage_id] if passage_id < len(self.indexes) else 0
        return self.blob[self.bounds[index] : self.bounds[index + 1]]


# --- Pass 3: Assign Multiple Memberships ---
def assign_multiple_memberships(
    spans: _Spans, refined_groups: _RefinedGroups
) -> tuple[_GroupLists, array]:
    """
    Pass 3: Assign passages to potentially multiple refined groups if their
            original source span overlaps a group's final intersection span.
    Returns:
        - group_lists: Each passage's list of refined group_ids, as JSON.
        - final_group_counts: Each refined group_id's final passage count.
    """
    _status(_label(3))
    group_lists = _GroupLists(len(spans))

    # Pass 2 makes groups file-specific, so a passage can only overlap groups
    # from its own file. Bucketing them by filename is what makes this a pass
    # over the results rather than over the results times every group in the
    # corpus: 31,254 passages against 24,056 groups was 752M comparisons.
    per_file: dict[int, array] = {}
    group_files = refined_groups.file_ids
    for group_id in range(len(refined_groups)):
        if refined_groups.live(group_id):
            file_id = group_files[group_id]
            bucket = per_file.get(file_id)
            if bucket is None:
                bucket = per_file[file_id] = array("i")
            bucket.append(group_id)
    groups_by_file = {
        file_id: _GroupSpans(group_ids, refined_groups.starts, refined_groups.ends)
        for file_id, group_ids in per_file.items()
    }
    del per_file

    # Spans repeat a lot, so each distinct one is looked up once.
    seen: dict[tuple[int, int, int], int] = {}
    file_ids = spans.file_ids
    starts = spans.starts
    ends = spans.ends
    for passage_id in tqdm(range(len(spans)), desc=_label(3), leave=False):
        span = (file_ids[passage_id], starts[passage_id], ends[passage_id])
        index = seen.get(span)
        if index is None:
            found = groups_by_file[span[0]].overlapping(span[1], span[2]) if span[0] in groups_by_file else []
            # Store the unique, sorted list of groups for this passage
            index = group_lists.add(sorted(found)) if found else 0
            seen[span] = index
        if index:
            group_lists.assign(passage_id, index)

    _status(_label(3))
    return group_lists, group_lists.counts(len(refined_groups))


# --- Main Orchestration and File Writing ---
def merge_alignments(results_file: str, count: int, workers: int = 1):
    """Merge alignments using the 3-pass method"""
    # Step 1: Initial broad grouping
    initial_group_map, initial_groups, spans = first_step_merge(results_file, count)
    if not initial_groups:
        _status()
        print("  No passage groups found.")
        return None

    # Step 2: Refine groups by strict intersection
    refined_group_map, refined_groups = refine_groups_strict_intersection(spans, initial_group_map)
    if not any(refined_groups.live(group_id) for group_id in range(len(refined_groups))):
        _status()
        print("  No passage groups left after refinement.")
        return None

    # Step 3: Assign multiple memberships
    group_lists, final_group_counts = assign_multiple_memberships(spans, refined_groups)

    # Both output steps read the records back, so they share one split of the file
    _status(_label(4))
    segments, bases = _segment_plan(results_file, count, workers)

    # Step 4: Write the final source passage group file
    groups_file = os.path.join(os.path.dirname(results_file), "passage_group_source.jsonl")
    groups = write_group_sources(
        results_file, groups_file, refined_group_map, refined_groups, final_group_counts,
        spans, segments, bases,
    )

    # Step 5: Rewrite the results file with the final group lists
    temp_results_file = f"{results_file}.temp_final.lz4"
    rewrite_results(results_file, temp_results_file, group_lists, count, segments, bases)
    os.remove(results_file)
    os.rename(temp_results_file, results_file)

    _status()
    print(f"  {groups:,} passage groups, written to {os.path.basename(groups_file)}")
    return groups_file


def _rewrite_records(input_file, output_file, group_lists: _GroupLists, passage_id: int,
                     progress=None, bar=None) -> int:
    """Put each record's group list into it. Returns the id after the last record."""
    pending = 0
    for batch in read_line_batches(input_file):
        rewritten = []
        for line in batch:
            # Get list, default empty
            rewritten.append(with_group_ids(line, group_lists[passage_id]))
            passage_id += 1
        output_file.write(b"".join(rewritten))
        if bar is not None:
            bar.update(len(batch))
        else:
            pending = _bump(progress, pending + len(batch))
    _flush(progress, pending)
    return passage_id


def _rewrite_segment(job: tuple[str, int, int, str]) -> str:
    """One frame range rewritten into its own part file.

    The group lists come through the fork; being flat arrays, reading them copies no pages.
    """
    path, offset, size, part = job
    group_lists = _SHARED["group_lists"]
    with contextlib.ExitStack() as stack:
        input_file = stack.enter_context(_lz4_window(path, offset, size))
        output_file = stack.enter_context(_Framer(stack.enter_context(open(part, "wb"))))
        _rewrite_records(input_file, output_file, group_lists,
                         _SHARED["bases"][offset], _SHARED["progress"])
    return part


def rewrite_results(results_file: str, target: str, group_lists: _GroupLists,
                    count: int, segments: list[tuple[int, int]], bases: dict[int, int]) -> None:
    """Write the results file back with a group list on every record.

    Split over frames and joined back in order. Written as many frames, so the
    next pass can split it too.
    """
    if len(segments) < 2:
        with contextlib.ExitStack() as stack:
            input_file = stack.enter_context(lz4.frame.open(results_file))
            output_file = stack.enter_context(_Framer(stack.enter_context(open(target, "wb"))))
            bar = stack.enter_context(tqdm(total=count, desc=_label(5), leave=False))
            _rewrite_records(input_file, output_file, group_lists, 0, bar=bar)
        return

    _SHARED.update(bases=bases, group_lists=group_lists)
    try:
        parts = _run_segments(_rewrite_segment, results_file, segments, count, _label(5))
    finally:
        # A module global, so a failed pass would otherwise leave the lists
        # pinned for the rest of the run.
        for key in ("bases", "group_lists"):
            _SHARED.pop(key, None)
    _status(_label(5))
    _join_parts(parts, target)


def write_group_sources(
    results_file: str,
    groups_file: str,
    refined_group_map: _DenseMap,
    refined_groups: _RefinedGroups,
    final_group_counts: array,
    spans: _Spans,
    segments: list[tuple[int, int]],
    bases: dict[int, int],
) -> int:
    """Write one record a refined group, with its representative member's metadata.

    Representatives are read back off the results file, so groups come out in
    file order; the database load keys on group_id and does not mind. Returns
    the number of groups written.
    """
    _status(_label(4))
    total = sum(1 for group_id in range(len(refined_groups)) if refined_groups.live(group_id))
    if len(segments) < 2:
        with contextlib.ExitStack() as stack:
            input_file = stack.enter_context(lz4.frame.open(results_file))
            output_file = stack.enter_context(open(groups_file, "wb"))
            bar = stack.enter_context(tqdm(total=total, desc=_label(4), leave=False))
            _group_source_records(input_file, output_file, refined_group_map, refined_groups,
                                  final_group_counts, spans, 0, bar=bar)
        return total

    _SHARED.update(bases=bases, refined_group_map=refined_group_map, refined_groups=refined_groups,
                   final_group_counts=final_group_counts, spans=spans)
    try:
        parts = _run_segments(_group_source_segment, results_file, segments, total, _label(4))
    finally:
        for key in ("bases", "refined_group_map", "refined_groups", "final_group_counts", "spans"):
            _SHARED.pop(key, None)
    # Plain jsonl, not lz4, so the parts join as they are.
    _status(_label(4))
    _join_parts(parts, groups_file)
    return total


def _group_source_records(input_file, output_file, refined_group_map: _DenseMap,
                          refined_groups: _RefinedGroups, final_group_counts: array,
                          spans: _Spans, passage_id: int, progress=None, bar=None) -> None:
    """Write a record for every group whose representative is in this range."""
    representatives = refined_groups.representatives
    source_text = _SourceText()
    pending = 0
    for batch in read_line_batches(input_file):
        records = []
        for line in batch:
            refined_id = refined_group_map[passage_id]
            if (refined_id < 0 or representatives[refined_id] != passage_id
                    or not refined_groups.live(refined_id)):
                passage_id += 1
                continue
            fields = orjson.loads(line)
            fields["passage_id"] = passage_id
            passage_id += 1
            metadata = {k: v for k, v in fields.items() if not k.startswith("target_")}
            filename = spans.names[refined_groups.file_ids[refined_id]]
            start_byte = refined_groups.starts[refined_id]
            end_byte = refined_groups.ends[refined_id]
            records.append(
                orjson.dumps(
                    {
                        **metadata,  # from the representative, a member of this group
                        "source_filename": filename,
                        "source_passage": source_text.read(start_byte, end_byte, filename),
                        "group_id": refined_id,
                        "source_start_byte": start_byte,
                        "source_end_byte": end_byte,
                        "count": final_group_counts[refined_id],
                    }
                )
                + b"\n"
            )
        if records:
            output_file.write(b"".join(records))
            if bar is not None:
                bar.update(len(records))
            else:
                pending = _bump(progress, pending + len(records))
    _flush(progress, pending)
    source_text.close()


def _group_source_segment(job: tuple[str, int, int, str]) -> str:
    """One frame range's group records, into its own part file."""
    path, offset, size, part = job
    with contextlib.ExitStack() as stack:
        input_file = stack.enter_context(_lz4_window(path, offset, size))
        output_file = stack.enter_context(open(part, "wb"))
        _group_source_records(input_file, output_file, _SHARED["refined_group_map"],
                              _SHARED["refined_groups"], _SHARED["final_group_counts"],
                              _SHARED["spans"], _SHARED["bases"][offset], _SHARED["progress"])
    return part


if __name__ == "__main__":
    import sys

    output_path = sys.argv[1]
    with open(os.path.join(output_path, "results/count.txt"), encoding="utf8") as input_file:
        count = int(input_file.read().strip())
    merge_alignments(os.path.join(output_path, "results/alignments.jsonl.lz4"), count)
