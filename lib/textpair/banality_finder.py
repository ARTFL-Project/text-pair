"""Banality detection"""

import os
from math import floor
from typing import Any, Dict, List, Tuple, Optional

import lz4.frame
import orjson
from tqdm import tqdm


class NgramDoc:
    """Doc with various properties"""

    __slots__ = ["name", "ngrams", "ngram_pos"]

    def __init__(self, filepath):
        self.name = os.path.basename(filepath)
        with open(filepath, "rb") as input_file:
            ngram_doc: Dict[str, List[List[int]]] = orjson.loads(input_file.read())
        self.ngrams: List[Tuple[int, int, int]] = [
            (start_byte, end_byte, int(ngram)) for ngram in ngram_doc for _, start_byte, end_byte in ngram_doc[ngram]
        ]
        self.ngram_pos: Dict[int, int] = {}
        self.ngrams.sort(key=lambda x: x[0])
        self.ngram_pos = {ngram[0]: index for index, ngram in enumerate(self.ngrams)}

    def get_ngrams(self, start_byte, end_byte) -> List[int]:
        start_index = self.ngram_pos[start_byte]
        ngrams = []
        for _, ending_byte, ngram in self.ngrams[start_index:]:
            if ending_byte > end_byte:
                break
            ngrams.append(ngram)
        return ngrams


def banality_auto_detect(filepath: str, common_ngrams_file: str, ngram_doc_path: str, count: Optional[int]):
    """Detect banalities automatically based on frequent ngram over-representation"""
    with open(common_ngrams_file, "rb") as input_file:
        total_ngrams = sum(1 for _ in input_file)
    top_ngrams = floor(total_ngrams / 1000)
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
            for line in tqdm(input_file, total=count, desc="Running banality auto-detection...", leave=True):
                alignment: Dict[str, Any] = orjson.loads(line)
                if source_ngram_doc is None or source_ngram_doc.name != alignment["source_ngrams"]:
                    source_ngram_doc = NgramDoc(os.path.join(ngram_doc_path, alignment["source_ngrams"]))
                ngrams_in_file = source_ngram_doc.get_ngrams(
                    int(alignment["source_start_byte"]), int(alignment["source_end_byte"])
                )
                common_ngram_matches = sum(1 for ngram in ngrams_in_file if ngram in common_ngrams)
                if (
                    common_ngram_matches / len(ngrams_in_file) * 100 >= 90
                ):  # if 50 (or more) % of ngrams are common ngrams
                    alignment["banality"] = True
                    banalities_found += 1
                else:
                    alignment["banality"] = False
                output_file.write(orjson.dumps(alignment) + b"\n")  # type: ignore
    os.system(f"rm {filepath} && mv {filepath}.banal.lz4 {filepath}")
    return banalities_found


def banality_phrase_matcher(filepath: str, banality_phrases_path: str, count: Optional[int]):
    """Detect banalities based on user provided phrases"""
    with open(banality_phrases_path, encoding="utf8") as input_file:
        banality_phrases = {phrase.strip().lower() for phrase in input_file}
    banalities_found = 0
    with lz4.frame.open(f"{filepath}.banal.lz4", mode="wb") as output_file:
        with lz4.frame.open(filepath) as input_file:
            for line in tqdm(input_file, total=count, desc="Running phrase-based banality detection...", leave=True):
                alignment: Dict[str, Any] = orjson.loads(line)
                for phrase in banality_phrases:
                    if phrase in alignment["source_passage"].lower():
                        alignment["banality"] = True
                        banalities_found += 1
                    else:
                        alignment["banality"] = False
                output_file.write(orjson.dumps(alignment) + b"\n")  # type: ignore
    os.system(f"rm {filepath} && mv {filepath}.banal.lz4 {filepath}")
    return banalities_found


if __name__ == "__main__":
    import sys

    filepath = sys.argv[1]
    ngrams_file = sys.argv[2]
    ngram_doc_path = sys.argv[3]
    with open(filepath.replace("alignments.jsonl.lz4", "count.txt"), "rb") as input_file:
        count = int(input_file.read().strip())
    total = banality_auto_detect(filepath, ngrams_file, ngram_doc_path, count)
    print(total, "banalities found.")
