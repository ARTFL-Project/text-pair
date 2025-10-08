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


async def zero_shot_banality_detection(
    input_path: str,
    zero_shot_model: str,
    store_banalities: bool,
) -> int:
    """
    Zero-shot classification-based banality detection.

    Classifies passages as SUBSTANTIVE vs non-substantive (banality).
    Anything not classified as substantive content is considered a banality.

    Args:
        input_path: Path to input alignments file (lz4 compressed)
        zero_shot_model: Hugging Face model for zero-shot classification
        store_banalities: Whether to keep banalities in output

    Returns:
        Number of banalities found
    """
    print(f"Loading zero-shot classifier: {zero_shot_model}")
    classifier = pipeline(
        "zero-shot-classification",
        model=zero_shot_model,
        device=0  # Use GPU if available
    )

    # Define categories for classification
    # We only care about: is it substantive or not?
    candidate_labels = [
        "Standardized phrases used to begin or end correspondence, including greetings, salutations, closings, and formulaic farewells",
        "Standard and conventional notes regarding printing, authorship, translation, or publication details, often found as paratextual elements",
        "Passage which contains only titles of persons, including honorifics, ranks, and titles of nobility",
        "Content that conveys specific information through developed narrative, unique description, argument, unique opinion, or detailed dialogue"
    ]

    # Prepare output
    temp_output_path = input_path.replace(".jsonl.lz4", ".jsonl_temp.lz4")
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    # Count lines for progress
    print("Counting alignments...")
    with lz4.frame.open(input_path, "rb") as f_count:
        num_lines = sum(1 for _ in f_count)

    if num_lines == 0:
        print("Input file is empty.")
        return 0

    print(f"Processing {num_lines} alignments with zero-shot classification...")

    banalities_found = 0
    lines_written = 0
    batch_size = 32  # Process 32 passages at a time

    with (lz4.frame.open(temp_output_path, "wb") as output_file,
          lz4.frame.open(input_path, "rb") as f_in,
          tqdm(total=num_lines, desc="Zero-shot banality detection") as pbar):

        batch = []
        batch_alignments = []

        for line_b in f_in:
            alignment = orjson.loads(line_b)

            # Pre-filter: very long passages are substantive
            if len(alignment.get("target_passage", "")) > 3000:
                alignment["zero_shot_classification"] = "SUBSTANTIVE"
                alignment["banality"] = False
                output_file.write(orjson.dumps(alignment) + b"\n")
                lines_written += 1
                pbar.update(1)
                continue

            batch.append(alignment["target_passage"])
            batch_alignments.append(alignment)

            # Process batch
            if len(batch) >= batch_size:
                results = classifier(
                    batch,
                    candidate_labels,
                    multi_label=False,
                    batch_size=batch_size
                )

                for alignment, result in zip(batch_alignments, results):
                    top_label = result["labels"][0]

                    # The last label is "substantive content" - if that's the top choice, it's not banal
                    if top_label == candidate_labels[-1]:  # Substantive
                        alignment["zero_shot_classification"] = "SUBSTANTIVE"
                        alignment["banality"] = False
                    else:  # Any other category = banality
                        alignment["zero_shot_classification"] = "BANAL"
                        alignment["banality"] = True
                        banalities_found += 1
                        if not store_banalities:
                            pbar.update(1)
                            continue

                    output_file.write(orjson.dumps(alignment) + b"\n")
                    lines_written += 1
                    pbar.update(1)

                batch = []
                batch_alignments = []

        # Process remaining batch
        if batch:
            results = classifier(
                batch,
                candidate_labels,
                multi_label=False,
                batch_size=len(batch)
            )

            for alignment, result in zip(batch_alignments, results):
                top_label = result["labels"][0]

                if top_label == candidate_labels[-1]:
                    alignment["zero_shot_classification"] = "SUBSTANTIVE"
                    alignment["banality"] = False
                else:
                    alignment["zero_shot_classification"] = "BANAL"
                    alignment["banality"] = True
                    banalities_found += 1
                    if not store_banalities:
                        pbar.update(1)
                        continue

                output_file.write(orjson.dumps(alignment) + b"\n")
                lines_written += 1
                pbar.update(1)

    print(f"\nZero-shot banality detection complete.")
    print(f"Total banalities found: {banalities_found}")
    print(f"Lines written: {lines_written}")

    os.remove(input_path)
    os.rename(temp_output_path, input_path)

    return banalities_found


async def banality_llm_post_eval(
    input_path: str,
    model_path: str,
    store_banalities: bool,
    port: int = 8090,
    context_window: int = 8192,
    concurrency_limit: int = 8,
) -> int:
    """
    LLM-based post-evaluation of banalities detected by earlier stages.

    Re-evaluates passages already flagged as banalities using an LLM
    to score their scholarly interest. Can rescue false positives.

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

        # Count lines for progress
        print("Counting alignments...")
        num_lines = 0
        try:
            with lz4.frame.open(input_path, "rb") as f_count:
                num_lines = sum(1 for _ in f_count)
            if num_lines == 0:
                print("Input file is empty.")
                return 0
        except Exception as e:
            print(f"Error reading input file {input_path}: {e}")
            return 0

        print(f"Processing {num_lines} alignments with LLM post-evaluation...")

        # Process in smaller batches for Stage 2 (longer prompts)
        stage2_batch_size = min(concurrency_limit // 2, 4)
        lines_written_count = 0
        banalities_confirmed = 0
        banalities_rescued = 0
        llm_evaluated_count = 0

        with (lz4.frame.open(temp_output_path, "wb") as output_file,
              lz4.frame.open(input_path, "rb") as f_in,
              tqdm(total=num_lines, desc="LLM post-evaluation") as pbar):

            batch_to_eval = []
            batch_no_eval = []

            for line_b in f_in:
                alignment = orjson.loads(line_b)

                # Only re-evaluate passages flagged as banalities by Stage 1
                if alignment.get("banality") is True:
                    batch_to_eval.append(alignment)
                else:
                    # Not a banality, keep as is
                    batch_no_eval.append(alignment)

                # Process batch when it reaches size
                if len(batch_to_eval) >= stage2_batch_size * 10:
                    # Score the banalities with LLM
                    scored = await evaluator.score_scholarly_interest_batch(
                        alignments=batch_to_eval,
                        batch_size=stage2_batch_size,
                        show_progress=False
                    )

                    llm_evaluated_count += len(scored)

                    # Process LLM results
                    for result in scored:
                        if isinstance(result, dict) and "error" not in result:
                            if result.get("banality") is True:
                                banalities_confirmed += 1
                                if not store_banalities:
                                    pbar.update(1)
                                    continue
                            else:
                                # LLM says it's substantive, rescue it
                                banalities_rescued += 1

                            output_file.write(orjson.dumps(result) + b"\n")  # type: ignore
                            lines_written_count += 1
                        pbar.update(1)

                    batch_to_eval = []

                # Write non-banalities
                for alignment in batch_no_eval:
                    output_file.write(orjson.dumps(alignment) + b"\n")  # type: ignore
                    lines_written_count += 1
                    pbar.update(1)

                batch_no_eval = []

            # Process remaining batches
            if batch_to_eval:
                scored = await evaluator.score_scholarly_interest_batch(
                    alignments=batch_to_eval,
                    batch_size=stage2_batch_size,
                    show_progress=False
                )

                llm_evaluated_count += len(scored)

                for result in scored:
                    if isinstance(result, dict) and "error" not in result:
                        if result.get("banality") is True:
                            banalities_confirmed += 1
                            if not store_banalities:
                                pbar.update(1)
                                continue
                        else:
                            banalities_rescued += 1

                        output_file.write(orjson.dumps(result) + b"\n")  # type: ignore
                        lines_written_count += 1
                    pbar.update(1)

            for alignment in batch_no_eval:
                output_file.write(orjson.dumps(alignment) + b"\n")  # type: ignore
                lines_written_count += 1
                pbar.update(1)

            pbar.close()

        print(f"\nLLM post-evaluation complete.")
        print(f"Passages evaluated by LLM: {llm_evaluated_count}")
        print(f"Banalities confirmed: {banalities_confirmed}")
        print(f"Banalities rescued (reclassified as substantive): {banalities_rescued}")
        print(f"Total lines written: {lines_written_count}")

        os.remove(input_path)
        os.rename(temp_output_path, input_path)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        print("Stopping llama-server...")
        evaluator.stop_server()
        if evaluator._session and not evaluator._session.closed:
            await evaluator.close_session()
        print("Server stopped.")

    return banalities_confirmed


if __name__ == "__main__":
    import sys

    file_path = sys.argv[1]
    # ngrams_file = sys.argv[2]
    # ngram_doc_path = sys.argv[3]
    # percentage = float(sys.argv[4])
    # with open(filepath.replace("alignments.jsonl.lz4", "count.txt"), "rb") as input_file:
    #     count = int(input_file.read().strip())
    # total = banality_auto_detect(filepath, ngrams_file, ngram_doc_path, count, percentage=percentage)
    # phrase_path = sys.argv[2]
    # total = phrase_matcher(file_path, phrase_path, int(sys.argv[3]))
    # print(total, "banalities found.")
    total = asyncio.run(zero_shot_banality_detection(file_path, "tasksource/ModernBERT-large-nli", store_banalities=True))