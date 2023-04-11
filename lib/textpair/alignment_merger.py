"""Merge Overlapping Alignments"""

from dataclasses import dataclass, field
from typing import Any
import os
from collections import Counter

import lz4.frame
import orjson
from tqdm import tqdm
from textpair import get_text


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
    ):
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

    def merge_passages(
        self,
        passages: list[dict[str, Any]],
    ):
        """Merge passages that are aligned to the same source passage"""
        passages.sort(
            key=lambda x: (x["source_start_byte"], x["source_start_byte"] - x["source_end_byte"])
        )  # sort by smaller start byte and bigger end_byte
        current_group = None
        for passage in passages:
            if current_group is None:
                current_group = self.passage_group_init(passage)
                continue
            if passage["source_start_byte"] < current_group.end_byte:
                self.passage_group_update(current_group, passage)
            else:
                current_group = self.passage_group_init(passage)

    def find_group(self, new_pair: dict[str, Any]) -> bool:
        """Find group for new pair"""
        match = False
        index = 0
        for local_target in self.merged_target_passages[new_pair["source_doc_id"]]:
            index += 1
            if (  # new passage is within
                local_target.start_byte <= new_pair["source_start_byte"]
                and local_target.end_byte >= new_pair["source_end_byte"]
            ):
                if new_pair["target_doc_id"] not in self.merged_target_passages:
                    self.merged_target_passages[new_pair["target_doc_id"]] = []
                self.merged_target_passages[new_pair["target_doc_id"]].append(
                    PassagePosition(new_pair["target_start_byte"], new_pair["target_end_byte"], local_target.group_id)
                )
                self.group_map[new_pair["passage_id"]] = local_target.group_id
                match = True
                break
        return match


def read_alignment(line: str, passage_id: int):
    """Read alignment from line"""
    alignment: dict[str, Any] = orjson.loads(line)
    alignment["passage_id"] = passage_id
    return alignment


def merge_alignments(results_file: str, count: int) -> str:
    """Merge passages that are aligned to the same source passage"""
    passages: list[dict[str, str]] = []
    alignment_groups = AlignmentGroups()
    doc_id = None
    with lz4.frame.open(results_file) as input_file:
        for passage_id, line in tqdm(
            enumerate(input_file), total=count, desc="Identifying passages groups...", leave=False
        ):
            new_pair = read_alignment(line, passage_id)
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

    group_id_count = Counter()
    with (
        lz4.frame.open(f"{results_file}.groups.lz4", mode="wb") as output_file,
        lz4.frame.open(results_file) as input_file,
    ):
        for passage_id, line in tqdm(
            enumerate(input_file), total=count, desc="Inserting group ids into alignments...", leave=False
        ):
            fields = orjson.loads(line)
            fields["group_id"] = alignment_groups.group_map[passage_id]
            fields["count"] = 1
            group_id_count[fields["group_id"]] += 1
            output_file.write(orjson.dumps(fields) + b"\n")  # type: ignore
    os.remove(results_file)
    os.rename(f"{results_file}.groups.lz4", f"{results_file}")

    groups_file = os.path.join(os.path.dirname(results_file), "passage_group_source.jsonl")
    with open(groups_file, "wb") as output_file:
        for group_id, source in tqdm(
            alignment_groups.group_to_bytes.items(),
            total=alignment_groups.group_id + 1,
            desc="Saving passage group sources...",
            leave=False,
        ):
            source_passage = get_text(source.start_byte, source.end_byte, source.filename)
            output_file.write(
                orjson.dumps(
                    {
                        **source.metadata,
                        "source_passage": source_passage,
                        "group_id": group_id,
                        "source_start_byte": source.start_byte,
                        "source_end_byte": source.end_byte,
                        "count": group_id_count[group_id],
                    }
                )
                + b"\n"
            )
    print("Grouping alignments... done.")
    return groups_file


if __name__ == "__main__":
    import sys

    output_path = sys.argv[1]
    with open(os.path.join(output_path, "results/count.txt"), encoding="utf8") as input_file:
        count = int(input_file.read().strip())
    merge_alignments(os.path.join(output_path, "results/alignments.jsonl.lz4"), count)
