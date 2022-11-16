"""Banality detection"""

import os
from math import floor
from typing import Any, Dict, List, Tuple

import lz4.frame
import orjson
from tqdm import tqdm


class NgramDoc:
    """Doc with various properties"""

    def __init__(self, filepath):
        self.name = filepath
        with open(filepath, "rb") as input_file:
            print(filepath)
            ngram_doc: Dict[str, List[List[int]]] = orjson.loads(input_file.read())
        self.ngrams: List[Tuple[int, int, int]] = [
            (start_byte, end_byte, int(ngram))
            for ngram, positions in ngram_doc.items()
            for _, start_byte, end_byte in positions
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


def banality_auto_detect(filepath: str, common_ngrams_file: str, ngram_doc_path: str, count: int):
    with open(common_ngrams_file, "rb") as input_file:
        total_ngrams = sum(1 for _ in input_file)
    top_ngrams = floor(total_ngrams / 10000)
    with open(common_ngrams_file, encoding="utf8") as input_file:
        common_ngrams = {int(next(input_file)) for _ in range(top_ngrams)}
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
                    common_ngram_matches / len(ngrams_in_file) * 100 >= 50
                ):  # if 50 (or more) % of ngrams are common ngrams
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
    banality_auto_detect(filepath, ngrams_file, ngram_doc_path, count)
