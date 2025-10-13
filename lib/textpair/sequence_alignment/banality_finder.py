"""Banality detection"""

import asyncio
import bisect
import os
import subprocess
from math import floor
from typing import Any, Optional

import ahocorasick_rs
import lz4.frame
import orjson
import regex as re
from tqdm import tqdm
from transformers import pipeline

from textpair.llm_evaluation import AsyncLLMEvaluator

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


async def banality_llm_post_eval(
    input_path: str,
    model_path: str,
    context_window: int,
    concurrency_limit: int,
    port: int,
    store_banalities: bool,
) -> int:
    """
    LLM-based post-evaluation of banalities detected by earlier stages using three-pass approach.

    Pass 1: Identify indices of passages flagged as banal
    Pass 2: Re-read file, batch evaluate only banal passages, track indices to rescue
    Pass 3: Re-read file, update banality flags for rescued passages, write output

    Args:
        input_path: Path to input alignments file (lz4 compressed) with banality flags
        model_path: Path to LLM model or HuggingFace model ID
        store_banalities: Whether to keep banalities in output
        port: Port for llama-server
        context_window: Context window size for the model
        concurrency_limit: Concurrency limit for LLM requests

    Returns:
        Number of banalities confirmed by LLM
    """
    # Initialize LLM evaluator
    evaluator = AsyncLLMEvaluator(
        model_path=model_path,
        port=port,
        context_window=context_window,
        concurrency_limit=concurrency_limit
    )

    try:
        evaluator.start_server()
        print(f"LLM server started successfully on port {port}")

        # Prepare output
        temp_output_path = input_path.replace(".jsonl.lz4", ".jsonl_temp.lz4")
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)

        # PASS 1: Identify indices of banal passages
        banal_indices = []

        with lz4.frame.open(input_path, "rb") as f_in:
            for idx, line_b in enumerate(f_in):
                alignment = orjson.loads(line_b)
                if alignment.get("banality") is True:
                    banal_indices.append(idx)

        num_lines = idx + 1  # Total number of alignments
        num_banal = len(banal_indices)

        print(f"Total alignments: {num_lines}")
        print(f"Banal passages to evaluate: {num_banal}")

        if num_banal == 0:
            print("No banal passages found. Skipping LLM evaluation.")
            return 0

        # PASS 2: Evaluate banal passages in batches, track rescues
        batch_size = min(concurrency_limit // 2, 4)
        non_banal_indices = set()  # Indices to flip from banal to non-banal
        scores_map = {}  # Store scores for all evaluated passages

        banal_passages = []
        banal_idx_batch = []
        banal_set = set(banal_indices)  # For fast lookup
        next_banal_pos = 0  # Position in banal_indices list

        with lz4.frame.open(input_path, "rb") as f_in, \
             tqdm(total=num_banal, desc="LLM evaluation of banal passages") as pbar:

            for idx, line_b in enumerate(f_in):
                # Check if this is a banal passage
                if idx in banal_set:
                    alignment = orjson.loads(line_b)
                    passage = alignment.get("target_passage", "")
                    banal_passages.append(passage)
                    banal_idx_batch.append(idx)

                    # Process batch when full
                    if len(banal_passages) >= batch_size * 10:
                        # Evaluate with LLM
                        results = await evaluator.score_scholarly_interest_batch(
                            passages=banal_passages,
                            batch_size=batch_size,
                            show_progress=False
                        )

                        # Process results
                        for batch_idx, (score, is_banal) in enumerate(results):
                            original_idx = banal_idx_batch[batch_idx]
                            scores_map[original_idx] = score

                            # If LLM says it's NOT banal, mark for rescue
                            if not is_banal:
                                non_banal_indices.add(original_idx)

                            pbar.update(1)

                        banal_passages = []
                        banal_idx_batch = []

            # Process remaining batch
            if banal_passages:
                results = await evaluator.score_scholarly_interest_batch(
                    passages=banal_passages,
                    batch_size=len(banal_passages),
                    show_progress=False
                )

                for batch_idx, (score, is_banal) in enumerate(results):
                    original_idx = banal_idx_batch[batch_idx]
                    scores_map[original_idx] = score

                    if not is_banal:
                        non_banal_indices.add(original_idx)

                    pbar.update(1)

        banalities_rescued = len(non_banal_indices)
        banalities_confirmed = num_banal - banalities_rescued

        print(f"\nLLM evaluated {num_banal} passages")
        print(f"Banalities confirmed: {banalities_confirmed}")
        print(f"Banalities rescued (reclassified as interesting): {banalities_rescued}")

        # PASS 3: Re-read file, update flags, write output
        lines_written = 0

        with lz4.frame.open(input_path, "rb") as f_in, \
             lz4.frame.open(temp_output_path, "wb") as output_file, \
             tqdm(total=num_lines, desc="Writing output") as pbar:

            for idx, line_b in enumerate(f_in):
                alignment = orjson.loads(line_b)

                # Update banality flag if this passage was rescued
                if idx in non_banal_indices:
                    alignment["banality"] = False
                    alignment["llm_rescued"] = True
                    alignment["formulaic_score"] = scores_map.get(idx, -1)
                elif idx in banal_set:
                    # Was banal and still is, add score
                    alignment["formulaic_score"] = scores_map.get(idx, -1)

                # Decide whether to write based on store_banalities flag
                should_write = True
                if alignment.get("banality") is True and not store_banalities:
                    should_write = False

                if should_write:
                    output_file.write(orjson.dumps(alignment) + b"\n")  # type: ignore
                    lines_written += 1

                pbar.update(1)

        print(f"Lines written to output: {lines_written}")

        # Replace original file with updated version
        os.remove(input_path)
        os.rename(temp_output_path, input_path)
        print(f"Updated file: {input_path}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        print("Stopping llama-server...")
        evaluator.stop_server()
        if evaluator._session and not evaluator._session.closed:
            await evaluator._session.close()
        print("Server stopped.")

    return banalities_confirmed


if __name__ == "__main__":
    import sys

    file_path = sys.argv[1]
    ngrams_file = sys.argv[2]
    ngram_doc_path = sys.argv[3]
    percentage = float(sys.argv[4])
    with open(file_path.replace("alignments.jsonl.lz4", "count.txt"), "rb") as input_file:
        count = int(input_file.read().strip())
    total = banality_auto_detect(file_path, ngrams_file, ngram_doc_path, True, count, 0.25, percentage)
    # phrase_path = sys.argv[2]
    # total = phrase_matcher(file_path, phrase_path, int(sys.argv[3]))
    # print(total, "banalities found.")
    # total = asyncio.run(zero_shot_banality_detection(file_path, "facebook/bart-large-mnli", store_banalities=True))