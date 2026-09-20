"""Banality detection"""

import os
import subprocess
from math import floor
from typing import Any, Optional, Union

import ahocorasick_rs
import lz4.frame
import msgspec
import numpy as np
import orjson
import regex as re
from numba import njit
from tqdm import tqdm

from . import ngram_binary


@njit(nogil=True, cache=True)
def _fill_table(keys, table, used):
    """Insert every key into an open-addressing table. Returns how many were new."""
    mask = np.uint64(table.size - 1)
    inserted = 0
    for i in range(keys.size):
        key = keys[i]
        # Fibonacci hashing, as in ngram_index: these keys are already hashes,
        # but they are the head of a frequency ordering rather than a spread.
        scattered = np.uint64(key) * np.uint64(0x9E3779B97F4A7C15)
        scattered ^= scattered >> np.uint64(29)
        slot = scattered & mask
        while used[slot] == 1 and table[slot] != key:
            slot = (slot + np.uint64(1)) & mask
        if used[slot] == 0:
            inserted += 1
            table[slot] = key
            used[slot] = 1
    return inserted


@njit(nogil=True, cache=True)
def _count_present(keys, table, used):
    """How many of `keys` are in the table."""
    mask = np.uint64(table.size - 1)
    found = 0
    for i in range(keys.size):
        key = keys[i]
        scattered = np.uint64(key) * np.uint64(0x9E3779B97F4A7C15)
        scattered ^= scattered >> np.uint64(29)
        slot = scattered & mask
        while used[slot] == 1:
            if table[slot] == key:
                found += 1
                break
            slot = (slot + np.uint64(1)) & mask
    return found


class CommonNgrams:
    """Membership in the most frequent keys of a corpus.

    An open-addressing table rather than a set. A real config asks for the top
    10%, which on frantext is 7.7M keys: 0.15GB here against 1.5GB as a Python
    set, counting the list of boxed ints it has to be built from. Counting a
    passage's hits is then one call rather than a loop over them.
    """

    __slots__ = ("table", "used", "size")

    def __init__(self, keys: np.ndarray):
        keys = np.ascontiguousarray(keys, dtype=np.int64)
        # Under half full, so a miss ends at the first empty slot it reaches.
        slots = 1 << max(int(keys.size * 2).bit_length(), 4)
        self.table = np.zeros(slots, dtype=np.int64)
        self.used = np.zeros(slots, dtype=np.uint8)
        self.size = int(_fill_table(keys, self.table, self.used)) if keys.size else 0

    def __len__(self) -> int:
        return self.size

    def count_in(self, keys: np.ndarray) -> int:
        """How many of `keys`, an int64 array, are common."""
        return int(_count_present(keys, self.table, self.used))


def load_common_ngrams(path: str, proportion: float) -> CommonNgrams:
    """The most frequent `proportion` percent of keys.

    ngram_index writes an int64 array ordered by descending corpus frequency.
    Older index directories have a text file of one decimal per line instead;
    both are read here so an index built before the change still works.
    """
    if path.endswith(".bin") and os.path.exists(path):
        # Only the head of the file is wanted, and on a corpus this size the
        # rest of it is half a gigabyte to read and throw away.
        wanted = floor(os.path.getsize(path) // 8 * proportion / 100)
        keys = np.fromfile(path, dtype=np.int64, count=wanted) if wanted else np.empty(0, np.int64)
        return CommonNgrams(keys)

    legacy = path[: -len(".bin")] + ".txt" if path.endswith(".bin") else path
    with open(legacy, "rb") as handle:
        total = sum(1 for _ in handle)
    wanted = floor(total * proportion / 100)
    decoded: list[int] = []
    with open(legacy, encoding="utf8") as handle:
        for _ in range(wanted):
            try:
                decoded.append(int(next(handle)))
            except ValueError:
                pass
    return CommonNgrams(np.array(decoded, dtype=np.int64))


PUNCTUATION = re.compile(r"[\p{P}\p{S}\p{N}]+")
SPACES = re.compile(r"\p{Z}+")


def clean_text(text: str) -> str:
    """Clean text for banality detection"""
    text = text.lower().strip()
    text = PUNCTUATION.sub("", text)
    text = SPACES.sub(" ", text)
    return text


class NgramDoc:
    """One document's n-grams in order, as the two columns of its ngrams_in_order file.

    The file is binary and mmap-shaped, so this is a read and two `np.frombuffer` views
    rather than a parse: 4.7ms of orjson per document became nothing measurable, and the
    filter opens one document per source in the results.

    `start_bytes` is widened to int64 on the way in: searching the int32 column
    the file stores for a Python int promotes the pair, which casts the whole
    column on every call and cost eight times the search itself.
    """

    __slots__ = ["name", "keys", "start_bytes"]

    def __init__(self, filepath):
        self.name = os.path.basename(filepath)
        with open(filepath, "rb") as input_file:
            keys, start_bytes = ngram_binary.order_columns(input_file.read(), filepath)
        self.keys = keys
        self.start_bytes = start_bytes.astype(np.int64)

    def span(self, start_byte: int, end_byte: int) -> tuple[int, int]:
        """Index range of the n-grams starting in [start_byte, end_byte).

        Both bounds in one search: at these sizes the call costs more than the
        binary search inside it.
        """
        bounds = self.start_bytes.searchsorted(
            np.array((start_byte, end_byte), dtype=np.int64), "left")
        return int(bounds[0]), int(bounds[1])

    def get_ngrams(self, start_byte, end_byte) -> list[int]:
        """The keys of every n-gram starting in [start_byte, end_byte)."""
        low, high = self.span(start_byte, end_byte)
        return self.keys[low:high].tolist()


class _Passage(msgspec.Struct):
    """What the automatic filter reads out of an alignment.

    A record carries around a hundred fields, nearly all of them metadata this
    never looks at, so they are skipped rather than decoded into a dict only to
    be encoded straight back.
    """

    source_ngrams: str
    source_start_byte: Union[int, str]
    source_end_byte: Union[int, str]
    # UNSET only when the record has no banality field at all, which is what
    # lets the verdict be spliced in rather than the record rewritten.
    banality: Union[bool, None, msgspec.UnsetType] = msgspec.UNSET


class _Verdict(msgspec.Struct):
    """The banality flag alone, for the pass that only sorts records by it."""

    banality: Union[bool, None] = None


class _SourcePassage(msgspec.Struct):
    """The source text alone, for phrase matching."""

    source_passage: str


_DECODE_PASSAGE = msgspec.json.Decoder(_Passage).decode
_DECODE_VERDICT = msgspec.json.Decoder(_Verdict).decode
_DECODE_SOURCE = msgspec.json.Decoder(_SourcePassage).decode
_BANALITY_FIELD = {True: b',"banality":true', False: b',"banality":false'}
# Documents kept open while scanning results. Oldest out first, and the
# largest frantext document is 16MB of columns, so the ceiling is small.
_DOCUMENTS_HELD = 8


def _with_banality(line: bytes, banal: bool, present: Any) -> bytes:
    """`line` with its banality field set to `banal`.

    Spliced into the record rather than decoding a hundred fields and encoding
    them back to change one. orjson wrote these files and reproduces them byte
    for byte, so the result is what the round trip produced. Anything the
    splice cannot account for goes through the round trip instead: a field
    already present has to keep its place, and a record not ending where one
    should is not ours to guess at.
    """
    if present is msgspec.UNSET:
        body = line[:-1] if line.endswith(b"\n") else line
        if body.endswith(b"}") and not body.endswith(b"{}"):
            return body[:-1] + _BANALITY_FIELD[banal] + b"}\n"
    alignment: dict[str, Any] = orjson.loads(line)
    alignment["banality"] = banal
    return orjson.dumps(alignment) + b"\n"


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
    common_ngrams = load_common_ngrams(common_ngrams_file, proportion)

    banalities_found = 0
    with (
        lz4.frame.open(f"{filepath}.temp.lz4", mode="wb") as output_file,
        lz4.frame.open(filepath) as input_file,
    ):
        # Results come grouped by source document, but not strictly: a frantext
        # run opens 2,325 distinct documents 3,642 times. A few documents of
        # history turns most of that back into a hit.
        loaded: dict[str, NgramDoc] = {}
        for line in tqdm(
            input_file,
            total=count,
            desc="Running banality auto-detection...",
            leave=False,
        ):
            passage = _DECODE_PASSAGE(line)
            source_ngram_doc = loaded.get(passage.source_ngrams)
            if source_ngram_doc is None:
                source_ngram_doc = NgramDoc(os.path.join(ngram_doc_path, passage.source_ngrams))
                if len(loaded) >= _DOCUMENTS_HELD:
                    del loaded[next(iter(loaded))]
                loaded[passage.source_ngrams] = source_ngram_doc
            low, high = source_ngram_doc.span(
                int(passage.source_start_byte), int(passage.source_end_byte)
            )
            # if n % (or more) of ngrams are common ngrams
            banality = high > low and (
                common_ngrams.count_in(source_ngram_doc.keys[low:high]) / (high - low) * 100
                >= threshold
            )
            banalities_found += banality
            # Always write to main file with banality flag set
            output_file.write(_with_banality(line, banality, passage.banality))  # type: ignore
    os.replace(f"{filepath}.temp.lz4", filepath)
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
        for line in tqdm(
            input_file,
            total=count,
            desc="Running phrase-based banality detection...",
            leave=False,
        ):
            banality = False
            if ac.find_matches_as_strings(clean_text(_DECODE_SOURCE(line).source_passage)):
                banality = True
                passages_filtered += 1
                filtered_passages.write(line)  # type: ignore
            if banality is False:
                output_file.write(line)  # type: ignore
    os.replace(f"{filepath}.keep.lz4", filepath)
    print("done")
    return passages_filtered


def separate_banalities(filepath: str, count: Optional[int]) -> int:
    """
    Separate passages flagged as banal into a separate file and remove them from main alignments.
    Should be called AFTER all banality detection and LLM evaluation is complete.

    Args:
        filepath: Path to alignments file
        count: Total number of alignments (for progress bar)

    Returns:
        Number of banalities separated
    """
    banalities_separated = 0
    banal_file_name = filepath.replace("alignments.jsonl", "banal_alignments.jsonl")

    with (
        lz4.frame.open(banal_file_name, mode="wb") as banal_output_file,
        lz4.frame.open(f"{filepath}.keep.lz4", mode="wb") as output_file,
        lz4.frame.open(filepath) as input_file,
    ):
        for line in tqdm(input_file, total=count, desc="Separating banalities...", leave=False):
            if _DECODE_VERDICT(line).banality is True:
                banalities_separated += 1
                banal_output_file.write(line)  # type: ignore
            else:
                output_file.write(line)  # type: ignore

    os.replace(f"{filepath}.keep.lz4", filepath)
    return banalities_separated


async def banality_llm_post_eval(
    input_path: str,
    model_path: str,
    context_window: int,
    concurrency_limit: int,
    port: int,
    store_banalities: bool,
    base_url: str = "",
    api_key: str = "",
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
        base_url: Optional external API base URL
        api_key: Optional API key for external server

    Returns:
        Number of banalities confirmed by LLM
    """
    # Initialize LLM evaluator
    from textpair_llm.llm_evaluation import AsyncLLMEvaluator

    evaluator = AsyncLLMEvaluator(
        model_path=model_path,
        port=port,
        context_window=context_window,
        concurrency_limit=concurrency_limit,
        base_url=base_url,
        api_key=api_key,
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

        with (
            lz4.frame.open(input_path, "rb") as f_in,
            tqdm(total=num_banal, desc="LLM evaluation of banal passages") as pbar,
        ):
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
                            show_progress=False,
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
                    show_progress=False,
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

        with (
            lz4.frame.open(input_path, "rb") as f_in,
            lz4.frame.open(temp_output_path, "wb") as output_file,
            tqdm(total=num_lines, desc="Writing output") as pbar,
        ):
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
    # ngrams_file = sys.argv[2]
    # ngram_doc_path = sys.argv[3]
    # percentage = float(sys.argv[4])
    # with open(file_path.replace("alignments.jsonl.lz4", "count.txt"), "rb") as input_file:
    #     count = int(input_file.read().strip())
    # total = banality_auto_detect(file_path, ngrams_file, ngram_doc_path, True, count, 0.25, percentage)
    phrase_path = sys.argv[2]
    total = phrase_matcher(file_path, phrase_path, int(sys.argv[3]))
    print(total, "banalities found.")
    # total = asyncio.run(zero_shot_banality_detection(file_path, "facebook/bart-large-mnli", store_banalities=True))
