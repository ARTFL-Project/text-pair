"""Banality detection"""

import bisect
import os
from math import floor
from typing import Any, Dict, List, Optional
import subprocess

import lz4.frame
import orjson
from tqdm import tqdm


class NgramDoc:
    """Doc with various properties"""

    __slots__ = ["name", "ngrams", "ngram_pos"]

    def __init__(self, filepath):
        self.name = os.path.basename(filepath)
        with open(filepath, "rb") as input_file:
            self.ngrams: List[List[int]] = orjson.loads(input_file.read())
        self.ngram_pos: List[int] = [ngram[0] for ngram in self.ngrams]

    def get_ngrams(self, start_byte, end_byte) -> List[int]:
        start_index = bisect.bisect_left(self.ngram_pos, start_byte)
        end_index = bisect.bisect_left(self.ngram_pos, end_byte)
        ngrams = [ngram for _, ngram in self.ngrams[start_index:end_index]]
        return ngrams


def banality_auto_detect(
    filepath: str,
    common_ngrams_file: str,
    ngram_doc_path: str,
    store_banalities: bool,
    count: Optional[int],
    percentage: float = 0.1,
):
    """Detect banalities automatically based on frequent ngram over-representation"""
    # Count number of ngrams to keep
    output = subprocess.check_output(["wc", "-l", common_ngrams_file]).decode("utf-8")
    total_ngrams = int(output.split(" ")[0])
    top_ngrams = floor(total_ngrams * percentage / 100)

    common_ngrams = set()
    with open(common_ngrams_file, encoding="utf8") as input_file:
        for _ in range(top_ngrams):
            try:  # TODO: investigate why we don't always get numbers
                common_ngrams.add(int(next(input_file)))
            except ValueError:
                pass

    banalities_found = 0
    with lz4.frame.open(f"{filepath}.banal.lz4", mode="wb") as output_file:
        with lz4.frame.open(filepath) as input_file:
            source_ngram_doc = None
            for line in tqdm(input_file, total=count, desc="Running banality auto-detection...", leave=False):
                alignment: Dict[str, Any] = orjson.loads(line)
                if source_ngram_doc is None or source_ngram_doc.name != alignment["source_ngrams"]:
                    source_ngram_doc = NgramDoc(os.path.join(ngram_doc_path, alignment["source_ngrams"]))
                ngrams_in_file = source_ngram_doc.get_ngrams(
                    int(alignment["source_start_byte"]), int(alignment["source_end_byte"])
                )
                common_ngram_matches = sum(1 for ngram in ngrams_in_file if ngram in common_ngrams)
                if (
                    ngrams_in_file and common_ngram_matches / len(ngrams_in_file) * 100 >= 90
                ):  # if 90 (or more) % of ngrams are common ngrams
                    alignment["banality"] = True
                    banalities_found += 1
                    if store_banalities is True:
                        output_file.write(orjson.dumps(alignment) + b"\n")  # type: ignore
    os.system(f"rm {filepath} && mv {filepath}.banal.lz4 {filepath}")
    return banalities_found


def phrase_matcher(filepath: str, banality_phrases_path: str, count: Optional[int]):
    """Detect banalities based on user provided phrases"""
    # TODO: split results into banality and results we keep: the assumption being that alignments
    # that match should be always dismissed.
    with open(banality_phrases_path, encoding="utf8") as input_file:
        banality_phrases = {phrase.strip().lower() for phrase in input_file}
    passages_filtered = 0
    filtered_file_name = filepath.replace("alignments.jsonl", "filtered_passages")
    with lz4.frame.open(filtered_file_name) as filtered_passages:
        with lz4.frame.open(f"{filepath}.keep.lz4", mode="wb") as output_file:
            with lz4.frame.open(filepath) as input_file:
                for line in tqdm(
                    input_file, total=count, desc="Running phrase-based banality detection...", leave=True
                ):
                    alignment: Dict[str, Any] = orjson.loads(line)
                    for phrase in banality_phrases:
                        if phrase in alignment["source_passage"].lower():
                            filtered_passages.write(alignment["source_passage"])
                            passages_filtered += 1
                            break
                    output_file.write(orjson.dumps(alignment) + b"\n")  # type: ignore
    os.system(f"rm {filepath} && mv {filepath}.keep.lz4 {filepath}")
    return passages_filtered


if __name__ == "__main__":
    import sys

    filepath = sys.argv[1]
    # ngrams_file = sys.argv[2]
    # ngram_doc_path = sys.argv[3]
    # percentage = float(sys.argv[4])
    # with open(filepath.replace("alignments.jsonl.lz4", "count.txt"), "rb") as input_file:
    #     count = int(input_file.read().strip())
    # total = banality_auto_detect(filepath, ngrams_file, ngram_doc_path, count, percentage=percentage)
    phrase_path = sys.argv[2]
    total = phrase_matcher(filepath, phrase_path, 71228098)
    print(total, "banalities found.")
