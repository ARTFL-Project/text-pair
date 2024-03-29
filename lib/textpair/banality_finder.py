"""Banality detection"""

import bisect
import os
from math import floor
from typing import Any, Optional
import subprocess
import regex as re

import lz4.frame
import orjson
from tqdm import tqdm
import ahocorasick_rs


PUNCTUATION = re.compile(r"[\p{P}\p{S}\p{N}]+")
SPACES = re.compile(r"\p{Z}+")


def clean_text(text: str) -> str:
    """Clean text for banality detection"""
    text = text.lower().strip()
    text = PUNCTUATION.sub("", text)
    text = SPACES.sub(" ", text)
    return text


class NgramDoc:
    """Doc with various properties"""

    __slots__ = ["name", "ngrams", "ngram_pos"]

    def __init__(self, filepath):
        self.name = os.path.basename(filepath)
        with open(filepath, "rb") as input_file:
            self.ngrams: list[list[int]] = orjson.loads(input_file.read())
        self.ngram_pos: list[int] = [ngram[0] for ngram in self.ngrams]

    def get_ngrams(self, start_byte, end_byte) -> list[int]:
        """Get ngrams in a given range"""
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
    proportion: float,
    threshold: float,
):
    """Detect banalities automatically based on frequent ngram over-representation"""
    # Count number of ngrams to keep
    output = subprocess.check_output(["wc", "-l", common_ngrams_file]).decode("utf-8")
    total_ngrams = int(output.split(" ", maxsplit=1)[0])
    top_ngrams = floor(total_ngrams * proportion / 100)

    common_ngrams = set()
    with open(common_ngrams_file, encoding="utf8") as input_file:
        for _ in range(top_ngrams):
            try:  # TODO: investigate why we don't always get numbers
                common_ngrams.add(int(next(input_file)))
            except ValueError:
                pass

    banalities_found = 0
    with (
        lz4.frame.open(f"{filepath}.temp.lz4", mode="wb") as output_file,
        lz4.frame.open(
            f"{filepath.replace('alignments.jsonl', 'banal_alignments.jsonl')}", mode="wb"
        ) as banal_output_file,
        lz4.frame.open(filepath) as input_file,
    ):
        source_ngram_doc = None
        for line in tqdm(input_file, total=count, desc="Running banality auto-detection...", leave=False):
            alignment: dict[str, Any] = orjson.loads(line)
            if source_ngram_doc is None or source_ngram_doc.name != alignment["source_ngrams"]:
                source_ngram_doc = NgramDoc(os.path.join(ngram_doc_path, alignment["source_ngrams"]))
            ngrams_in_file = source_ngram_doc.get_ngrams(
                int(alignment["source_start_byte"]), int(alignment["source_end_byte"])
            )
            common_ngram_matches = sum(1 for ngram in ngrams_in_file if ngram in common_ngrams)
            if (
                ngrams_in_file and common_ngram_matches / len(ngrams_in_file) * 100 >= threshold
            ):  # if n % (or more) of ngrams are common ngrams
                alignment["banality"] = True
                banalities_found += 1
                if store_banalities is True:
                    output_file.write(orjson.dumps(alignment) + b"\n")  # type: ignore
                else:
                    banal_output_file.write(orjson.dumps(alignment) + b"\n")  # type: ignore
            else:
                alignment["banality"] = False
                output_file.write(orjson.dumps(alignment) + b"\n")  # type: ignore
    os.system(f"rm {filepath} && mv {filepath}.temp.lz4 {filepath}")
    return banalities_found


def clean_phrases(file: str):
    """Clean phrases for phrase-based banality detection"""
    with open(file, encoding="utf8") as input_file:
        for phrase in input_file:
            phrase = clean_text(phrase)
            if re.search(r"\w", phrase):
                yield phrase


def phrase_matcher(filepath: str, banality_phrases_path: str, count: Optional[int]):
    """Detect banalities based on user provided phrases"""
    print("Building tree for phrase-based banality detection...", end="", flush=True)
    ac = ahocorasick_rs.AhoCorasick(clean_phrases(banality_phrases_path))
    print("\r", end="")
    passages_filtered = 0
    filtered_file_name = filepath.replace("alignments.jsonl", "filtered_passages.jsonl")
    with (
        lz4.frame.open(filtered_file_name, mode="wb") as filtered_passages,
        lz4.frame.open(f"{filepath}.keep.lz4", mode="wb") as output_file,
        lz4.frame.open(filepath) as input_file,
    ):
        for line in tqdm(input_file, total=count, desc="Running phrase-based banality detection...", leave=False):
            alignment: dict[str, Any] = orjson.loads(line)
            banality = False
            if ac.find_matches_as_strings(clean_text(alignment["source_passage"])):
                banality = True
                passages_filtered += 1
                filtered_passages.write(line)  # type: ignore
            if banality is False:
                output_file.write(line)  # type: ignore
    os.system(f"rm {filepath} && mv {filepath}.keep.lz4 {filepath}")
    print("done")
    return passages_filtered


if __name__ == "__main__":
    import sys

    file_path = sys.argv[1]
    # ngrams_file = sys.argv[2]
    # ngram_doc_path = sys.argv[3]
    # percentage = float(sys.argv[4])
    # with open(filepath.replace("alignments.jsonl.lz4", "count.txt"), "rb") as input_file:
    #     count = int(input_file.read().strip())
    # total = banality_auto_detect(filepath, ngrams_file, ngram_doc_path, count, percentage=percentage)
    phrase_path = sys.argv[2]
    total = phrase_matcher(file_path, phrase_path, int(sys.argv[3]))
    print(total, "banalities found.")
