"""Merge Overlapping Alignments"""

import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Union

import lz4.frame
import msgspec
import orjson
from tqdm import tqdm

from textpair.utils import clean_passage, get_text


@dataclass(slots=True)
class SourcePassageGroup:
    """Source passage for group"""

    filename: str
    start_byte: int
    end_byte: int
    metadata: dict[str, Any]

    def __pos_init__(self):
        self.metadata = {k: v for k, v in self.metadata.items() if k.startswith(("source_")) and k != "source_passage"}


@dataclass(slots=True)
class PassageGroup:
    "A passage group representation"

    filename: str
    start_byte: int
    end_byte: int
    group_id: int
    matches: int
    fields: dict[str, str]


@dataclass(slots=True)
class PassagePosition:
    """Passage position representation"""

    start_byte: int
    end_byte: int
    group_id: int


@dataclass(slots=True)
class AlignmentGroups:
    """Holding alignment group data"""

    group_id: int = -1
    merged_target_passages: dict[str, list[PassagePosition]] = field(default_factory=dict)
    group_map: dict[int, int] = field(default_factory=dict)
    group_to_bytes: dict[int, SourcePassageGroup] = field(default_factory=dict)

    def passage_group_init(
        self,
        passage: dict[str, Any],
    ) -> PassageGroup:
        """Initialize new group"""
        self.group_id += 1
        current_target = PassagePosition(passage["target_start_byte"], passage["target_end_byte"], self.group_id)
        if passage["target_doc_id"] not in self.merged_target_passages:
            self.merged_target_passages[passage["target_doc_id"]] = []
        self.merged_target_passages[passage["target_doc_id"]].append(current_target)
        self.group_map[passage["passage_id"]] = self.group_id
        self.group_to_bytes[self.group_id] = SourcePassageGroup(
            passage["source_filename"],
            passage["source_start_byte"],
            passage["source_end_byte"],
            {k: v for k, v in passage.items() if not k.startswith("target_")},
        )
        return PassageGroup(
            passage["source_filename"],
            passage["source_start_byte"],
            passage["source_end_byte"],
            self.group_id,
            2,
            passage,
        )

    def passage_group_update(
        self,
        current_group: PassageGroup,
        passage: dict[str, Any],
    ) -> PassageGroup:
        """Update current group"""
        if passage["source_end_byte"] > current_group.end_byte:
            current_group.end_byte = passage["source_end_byte"]
            self.group_to_bytes[self.group_id].end_byte = passage["source_end_byte"]
        current_target = PassagePosition(passage["target_start_byte"], passage["target_end_byte"], self.group_id)
        if passage["target_doc_id"] not in self.merged_target_passages:
            self.merged_target_passages[passage["target_doc_id"]] = []
        self.merged_target_passages[passage["target_doc_id"]].append(current_target)
        current_group.matches += 2
        self.group_map[passage["passage_id"]] = self.group_id
        return current_group

    def merge_passages(
        self,
        passages: list[dict[str, Any]],
    ):
        """Merge passages that are aligned to the same source passage"""
        passages.sort(
            key=lambda x: (
                x["source_start_byte"],
                x["source_start_byte"] - x["source_end_byte"],
            )
        )  # sort by smaller start byte and bigger end_byte
        current_group = None
        for passage in passages:
            if current_group is None:
                current_group = self.passage_group_init(passage)
                continue
            if passage["source_start_byte"] < current_group.end_byte:
                current_group = self.passage_group_update(current_group, passage)
            else:
                current_group = self.passage_group_init(passage)

    def find_group(self, new_pair: dict[str, Any]) -> bool:
        """Find group for new pair"""
        match = False
        for local_target in self.merged_target_passages[new_pair["source_doc_id"]]:
            if (  # new passage is within
                local_target.start_byte <= new_pair["source_start_byte"]
                and local_target.end_byte >= new_pair["source_end_byte"]
            ):
                if new_pair["target_doc_id"] not in self.merged_target_passages:
                    self.merged_target_passages[new_pair["target_doc_id"]] = []
                self.merged_target_passages[new_pair["target_doc_id"]].append(
                    PassagePosition(
                        new_pair["target_start_byte"],
                        new_pair["target_end_byte"],
                        local_target.group_id,
                    )
                )
                self.group_map[new_pair["passage_id"]] = local_target.group_id
                match = True
                break
        return match


@dataclass(slots=True)
class AlignmentRefineInfo:
    passage_id: int
    source_filename: str
    source_start_byte: int
    source_end_byte: int


@dataclass(slots=True)
class RefinedGroupInfo:
    """Stores info for a refined group during Pass 2"""

    id: int
    source_filename: str  # Groups are file-specific in Pass 2
    intersect_start: int
    intersect_end: int
    member_ids: set = field(default_factory=set)


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


def with_group_ids(line: bytes, groups: list[int]) -> bytes:
    """`line` with its group list set, and any stale count dropped.

    The rewrite adds one field to a record of about a hundred, so the list is
    spliced into the bytes rather than the record decoded and encoded back.
    orjson wrote these files and reproduces them byte for byte, so the result
    is what the round trip produced. A record that already carries either
    field goes through the round trip, since one has to keep its place and the
    other has to go.
    """
    present = _DECODE_GROUP_FIELDS(line)
    if present.group_id is msgspec.UNSET and present.count is msgspec.UNSET:
        body = line[:-1] if line.endswith(b"\n") else line
        if body.endswith(b"}") and not body.endswith(b"{}"):
            return body[:-1] + b',"group_id":' + orjson.dumps(groups) + b"}\n"
    fields = orjson.loads(line)
    fields["group_id"] = groups
    # Remove old 'count' field if it exists, use final_group_counts later
    if "count" in fields:
        del fields["count"]
    return orjson.dumps(fields) + b"\n"


def read_alignment(line: bytes, passage_id: int):
    """Read alignment from line"""
    alignment: dict[str, Any] = orjson.loads(line)
    alignment["passage_id"] = passage_id
    return alignment


def first_step_merge(
    results_file: str, count: int
) -> tuple[dict[int, int], dict[int, SourcePassageGroup], list[tuple[str, int, int]]]:
    """Merge passages that are aligned to the same source passage.

    Also returns each passage's source file and span, by passage id. The two
    passes after this one need nothing else out of a record, and this one has
    already decoded every one of them.
    """
    passages: list[dict[str, str]] = []
    alignment_groups = AlignmentGroups()
    spans: list[tuple[str, int, int]] = []
    filenames: dict[str, str] = {}
    doc_id = None
    with lz4.frame.open(results_file) as input_file:
        for passage_id, line in tqdm(
            enumerate(input_file),
            total=count,
            desc="Identifying passages groups...",
            leave=False,
        ):
            new_pair = read_alignment(line, passage_id)
            # One string per distinct file rather than one per record.
            filename = new_pair["source_filename"]
            spans.append(
                (
                    filenames.setdefault(filename, filename),
                    new_pair["source_start_byte"],
                    new_pair["source_end_byte"],
                )
            )
            source_doc_id: str = new_pair["source_doc_id"]
            match = False
            if source_doc_id in alignment_groups.merged_target_passages:
                match = alignment_groups.find_group(new_pair)
            if match is True:
                continue

            current_doc_id = source_doc_id
            if doc_id != current_doc_id and doc_id is not None:
                alignment_groups.merge_passages(passages)
                passages = []
            doc_id = current_doc_id
            passages.append(new_pair)

        if len(passages) > 0:
            alignment_groups.merge_passages(passages)
            passages = []

    print(f"Pass 1 finished. Found {len(alignment_groups.group_to_bytes)} initial groups.")
    return alignment_groups.group_map, alignment_groups.group_to_bytes, spans

    # group_id_count = Counter()
    # with (
    #     lz4.frame.open(f"{results_file}.groups.lz4", mode="wb") as output_file,
    #     lz4.frame.open(results_file) as input_file,
    # ):
    #     for passage_id, line in tqdm(
    #         enumerate(input_file), total=count, desc="Inserting group ids into alignments...", leave=False
    #     ):
    #         fields = orjson.loads(line)
    #         fields["group_id"] = alignment_groups.group_map[passage_id]
    #         fields["count"] = 1
    #         group_id_count[fields["group_id"]] += 1
    #         output_file.write(orjson.dumps(fields) + b"\n")  # type: ignore
    # os.remove(results_file)
    # os.rename(f"{results_file}.groups.lz4", f"{results_file}")
    # return alignment_groups, group_id_count


def refine_groups_strict_intersection(
    spans: list[tuple[str, int, int]], initial_group_map: dict[int, int]
) -> tuple[dict[int, int], dict[int, dict]]:
    """
    Pass 2: Split initial groups based purely on direct source passage overlap
            using the shrinking intersection logic. Assigns each passage to ONE refined group.
    Returns:
        - refined_group_map: Mapping from original passage_id to the refined group_id.
        - refined_group_definitions: Maps refined group ID to its final intersection span
                                      {'start': int, 'end': int, 'source_filename': str}.
    """
    # 1. Group alignments by the *initial* group_id and source_filename
    groups_data = defaultdict(lambda: defaultdict(list))  # initial_group_id -> source_filename -> [AlignmentRefineInfo]
    print("Starting Pass 2: Refining Groups by Strict Intersection...")
    for passage_id, (source_filename, start_byte, end_byte) in tqdm(
        enumerate(spans), total=len(spans), desc="Pass 2 Reading...", leave=False
    ):
        initial_group_id = initial_group_map.get(passage_id)
        if initial_group_id is not None:
            groups_data[initial_group_id][source_filename].append(
                AlignmentRefineInfo(
                    passage_id=passage_id,
                    source_filename=source_filename,
                    source_start_byte=start_byte,
                    source_end_byte=end_byte,
                )
            )

    # 2. Process each initial group's alignments per source file
    refined_group_map = {}
    refined_group_definitions_temp = {}  # Store temp group info {refined_id: RefinedGroupInfo}
    next_refined_group_id = 0

    print("Processing initial groups for refinement...")
    for initial_group_id, files_in_group in tqdm(groups_data.items(), desc="Pass 2 Processing Groups", leave=False):
        for source_filename, alignments_in_file in files_in_group.items():
            # Sort alignments within this file by start byte
            alignments_in_file.sort(key=lambda x: x.source_start_byte)

            # Track active refined groups *for this specific file* within the initial group
            active_refined_groups_for_file = []  # List[RefinedGroupInfo]

            for align_info in alignments_in_file:
                p_id = align_info.passage_id
                p_start = align_info.source_start_byte
                p_end = align_info.source_end_byte
                joined_group_info = None

                # Try to join an existing *refined* group within this file
                best_fit_group = None
                for current_refined_group in active_refined_groups_for_file:
                    # Check overlap with current intersection
                    if p_start < current_refined_group.intersect_end and p_end > current_refined_group.intersect_start:
                        best_fit_group = current_refined_group
                        break  # First fit

                if best_fit_group:
                    # --- Join Existing Refined Group ---
                    refined_group_map[p_id] = best_fit_group.id
                    best_fit_group.member_ids.add(p_id)
                    # Update intersection (shrinking)
                    best_fit_group.intersect_start = max(best_fit_group.intersect_start, p_start)
                    best_fit_group.intersect_end = min(best_fit_group.intersect_end, p_end)

                else:
                    # --- Start New Refined Group ---
                    new_id = next_refined_group_id
                    next_refined_group_id += 1
                    new_group_info = RefinedGroupInfo(
                        id=new_id,
                        source_filename=source_filename,
                        intersect_start=p_start,
                        intersect_end=p_end,
                        member_ids={p_id},
                    )
                    active_refined_groups_for_file.append(new_group_info)
                    refined_group_map[p_id] = new_id
                    refined_group_definitions_temp[new_id] = new_group_info  # Store initial info

                # --- Cleanup invalid groups within this file context ---
                groups_to_remove_indices = set()
                for i, group in enumerate(active_refined_groups_for_file):
                    if group.intersect_start >= group.intersect_end:
                        groups_to_remove_indices.add(i)
                        # Remove definition if it becomes invalid
                        if group.id in refined_group_definitions_temp:
                            del refined_group_definitions_temp[group.id]

                if groups_to_remove_indices:
                    active_refined_groups_for_file = [
                        g for i, g in enumerate(active_refined_groups_for_file) if i not in groups_to_remove_indices
                    ]

    # Finalize definitions for valid groups
    refined_group_definitions = {}
    for group_id, group_info in refined_group_definitions_temp.items():
        if group_info.intersect_start < group_info.intersect_end and group_info.member_ids:
            refined_group_definitions[group_id] = {
                "start": group_info.intersect_start,
                "end": group_info.intersect_end,
                "source_filename": group_info.source_filename,  # Include filename
            }

    print(f"Pass 2 finished. Refined into {len(refined_group_definitions)} groups.")
    return refined_group_map, refined_group_definitions


# --- Pass 3: Assign Multiple Memberships ---
def assign_multiple_memberships(
    spans: list[tuple[str, int, int]], refined_group_definitions: dict[int, dict]
) -> tuple[dict[int, list[int]], Counter]:
    """
    Pass 3: Assign passages to potentially multiple refined groups if their
            original source span overlaps a group's final intersection span.
    Returns:
        - final_passage_to_groups: Maps passage_id to a LIST of refined group_ids.
        - final_group_counts: Maps refined group_id to its final passage count.
    """
    final_passage_to_groups = defaultdict(list)
    final_group_counts = Counter()
    print("Starting Pass 3: Assigning Multiple Memberships...")

    # Pass 2 makes groups file-specific, so a passage can only overlap groups
    # from its own file. Bucketing them by filename is what makes this a pass
    # over the results rather than over the results times every group in the
    # corpus: 31,254 passages against 24,056 groups was 752M comparisons.
    groups_by_file = defaultdict(list)
    for group_id, definition in refined_group_definitions.items():
        groups_by_file[definition["source_filename"]].append(
            (definition["start"], definition["end"], group_id)
        )

    for passage_id, (source_filename, p_start, p_end) in tqdm(
        enumerate(spans), total=len(spans), desc="Pass 3 Checking...", leave=False
    ):
        assigned_groups = set()

        # Check against the final intersections from Pass 2 for this file
        for intersect_start, intersect_end, group_id in groups_by_file.get(source_filename, ()):
            # Check if passage's *original* span overlaps the group's *final* intersection
            if p_start < intersect_end and p_end > intersect_start:
                assigned_groups.add(group_id)

        # Store the unique, sorted list of groups for this passage
        if assigned_groups:
            final_groups_list = sorted(assigned_groups)
            final_passage_to_groups[passage_id] = final_groups_list
            # Increment count for each group this passage belongs to
            for group_id in final_groups_list:
                final_group_counts[group_id] += 1

    print(f"Pass 3 finished.")
    return dict(final_passage_to_groups), final_group_counts


# --- Main Orchestration and File Writing ---
def merge_alignments(results_file: str, count: int):
    """Merge alignments using the 3-pass method"""
    print("Starting alignment grouping...")

    # Step 1: Initial broad grouping
    initial_group_map, initial_group_sources, spans = first_step_merge(results_file, count)

    if not initial_group_map:
        print("Aborting: No initial groups found in Pass 1.")
        return None  # Indicate failure

    # Step 2: Refine groups by strict intersection
    refined_group_map, refined_group_definitions = refine_groups_strict_intersection(
        spans, initial_group_map
    )

    if not refined_group_definitions:
        print("Aborting: No valid refined groups found in Pass 2.")
        # Optionally, clean up intermediate files if needed
        return None  # Indicate failure

    # Step 3: Assign multiple memberships
    final_passage_to_groups, final_group_counts = assign_multiple_memberships(
        spans, refined_group_definitions
    )

    # Step 4: Rewrite the results file with the final GROUP LIST
    temp_results_file = f"{results_file}.temp_final.lz4"
    print(f"Rewriting results file with final group lists...")
    with (
        lz4.frame.open(temp_results_file, mode="wb") as output_file,
        lz4.frame.open(results_file) as input_file,
    ):
        for passage_id, line in tqdm(enumerate(input_file), total=count, desc="Rewriting results...", leave=False):
            # Get list, default empty
            final_groups = final_passage_to_groups.get(passage_id, [])
            output_file.write(with_group_ids(line, final_groups))

    # Replace original file with the new one
    os.remove(results_file)
    os.rename(temp_results_file, results_file)
    print("Results file rewritten successfully.")

    # Step 5: Write the final source passage group file
    groups_file = os.path.join(os.path.dirname(results_file), "passage_group_source.jsonl")
    print(f"Writing final source group definitions to {groups_file}...")
    source_text = _SourceText()
    with open(groups_file, "wb") as output_file:
        # Representative metadata is whatever was stored when the group was
        # created, so which of its passages came first never entered into it.
        representative_metadata = {}
        for init_group_id in initial_group_map.values():
            source_info = initial_group_sources.get(init_group_id)
            if source_info:
                representative_metadata[init_group_id] = source_info.metadata

        # Need to map refined group to initial group to get metadata
        # Find the initial group for each refined group (can be arbitrary if multiple)
        refined_to_initial_map = {}
        for passage_id, refined_id in refined_group_map.items():
            if refined_id not in refined_to_initial_map:
                initial_id = initial_group_map.get(passage_id)
                if initial_id is not None:
                    refined_to_initial_map[refined_id] = initial_id

        # Now write the final source file using refined definitions and counts
        for group_id, definition in tqdm(
            refined_group_definitions.items(),
            total=len(refined_group_definitions),
            desc="Saving final passage group sources...",
            leave=False,
        ):
            # Get metadata via the initial group mapping
            initial_group_id_for_meta = refined_to_initial_map.get(group_id)
            metadata_to_use = representative_metadata.get(initial_group_id_for_meta, {})

            # Get the actual source text using the final intersection span
            source_passage = source_text.read(
                definition["start"], definition["end"], definition["source_filename"]
            )

            output_file.write(
                orjson.dumps(
                    {
                        **metadata_to_use,  # Use metadata from representative initial group
                        "source_filename": definition["source_filename"],  # Use correct filename
                        "source_passage": source_passage,
                        "group_id": group_id,
                        "source_start_byte": definition["start"],
                        "source_end_byte": definition["end"],
                        "count": final_group_counts[group_id],
                    }
                )
                + b"\n"
            )
    source_text.close()

    print("Grouping alignments... done.")
    return groups_file


# def merge_alignments(results_file: str, count: int):
#     """Merge alignments"""
#     alignment_groups, group_id_count = first_step_merge(results_file, count)

#     groups_file = os.path.join(os.path.dirname(results_file), "passage_group_source.jsonl")
#     with open(groups_file, "wb") as output_file:
#         for group_id, source in tqdm(
#             alignment_groups.group_to_bytes.items(),
#             total=alignment_groups.group_id + 1,
#             desc="Saving passage group sources...",
#             leave=False,
#         ):
#             source_passage = get_text(source.start_byte, source.end_byte, source.filename)
#             output_file.write(
#                 orjson.dumps(
#                     {
#                         **source.metadata,
#                         "source_passage": source_passage,
#                         "group_id": group_id,
#                         "source_start_byte": source.start_byte,
#                         "source_end_byte": source.end_byte,
#                         "count": group_id_count[group_id],
#                     }
#                 )
#                 + b"\n"
#             )
#     print("Grouping alignments... done.")
#     return groups_file

if __name__ == "__main__":
    import sys

    output_path = sys.argv[1]
    with open(os.path.join(output_path, "results/count.txt"), encoding="utf8") as input_file:
        count = int(input_file.read().strip())
    merge_alignments(os.path.join(output_path, "results/alignments.jsonl.lz4"), count)
