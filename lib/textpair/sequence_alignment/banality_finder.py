"""Banality detection"""

import contextlib
import io
import multiprocessing as mp
import os
import struct
import subprocess
from math import floor
from shlex import quote
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


class _Filtered(msgspec.Struct):
    """What both filters need, for the pass that runs them together."""

    source_passage: str
    source_ngrams: str
    source_start_byte: Union[int, str]
    source_end_byte: Union[int, str]
    banality: Union[bool, None, msgspec.UnsetType] = msgspec.UNSET


class _Verdict(msgspec.Struct):
    """The banality flag alone, for the pass that only sorts records by it."""

    banality: Union[bool, None] = None


class _SourcePassage(msgspec.Struct):
    """The source text alone, for phrase matching."""

    source_passage: str


_DECODE_PASSAGE = msgspec.json.Decoder(_Passage).decode
_DECODE_VERDICT = msgspec.json.Decoder(_Verdict).decode
_DECODE_FILTERED = msgspec.json.Decoder(_Filtered).decode
_DECODE_SOURCE = msgspec.json.Decoder(_SourcePassage).decode
_BANALITY_FIELD = {True: b',"banality":true', False: b',"banality":false'}
# Documents kept open while scanning results. Oldest out first, and the
# largest frantext document is 16MB of columns, so the ceiling is small.
_DOCUMENTS_HELD = 8


_LZ4_MAGIC = 0x184D2204
_LZ4_SKIPPABLE = 0x184D2A50          # magic | 0x0..0xF


def frame_bounds(path: str) -> list[tuple[int, int]]:
    """(offset, length) of every lz4 frame in `path`, from the headers alone.

    An alignments file is a concatenation of frames, one per chunk the writers emitted,
    and any frame-aligned range of it decodes on its own. So these are independent units
    of work already in output order: a worker takes a run of them, rewrites it, and the
    results concatenate back with the records in the order they came.
    """
    bounds: list[tuple[int, int]] = []
    with open(path, "rb") as handle:
        while True:
            start = handle.tell()
            head = handle.read(4)
            if len(head) < 4:
                break
            magic = struct.unpack("<I", head)[0]
            if magic & 0xFFFFFFF0 == _LZ4_SKIPPABLE:
                handle.seek(struct.unpack("<I", handle.read(4))[0], 1)
                bounds.append((start, handle.tell() - start))
                continue
            if magic != _LZ4_MAGIC:
                raise ValueError(f"{path}: not an lz4 frame at offset {start}")
            flag, _block_descriptor = handle.read(2)
            if flag >> 6 != 1:
                raise ValueError(f"{path}: unsupported frame version at offset {start}")
            handle.seek(8 * bool(flag & 0x08) + 4 * bool(flag & 0x01) + 1, 1)
            block_checksum = 4 * bool(flag & 0x10)
            while True:
                raw = handle.read(4)
                if len(raw) < 4:
                    raise ValueError(f"{path}: truncated block at offset {start}")
                size = struct.unpack("<I", raw)[0]
                if size == 0:
                    break
                handle.seek((size & 0x7FFFFFFF) + block_checksum, 1)
            handle.seek(4 * bool(flag & 0x04), 1)
            bounds.append((start, handle.tell() - start))
    return bounds


def _segments(bounds: list[tuple[int, int]], parts: int) -> list[tuple[int, int]]:
    """`bounds` grouped into at most `parts` runs of roughly equal bytes."""
    total = sum(length for _, length in bounds)
    if parts < 2 or len(bounds) < 2 or total == 0:
        return [(bounds[0][0], total)] if bounds else []
    target = total / parts
    out: list[tuple[int, int]] = []
    start = bounds[0][0]
    taken = 0
    for offset, length in bounds:
        taken += length
        if taken >= target and len(out) < parts - 1:
            out.append((start, taken))
            start = offset + length
            taken = 0
    if taken:
        out.append((start, taken))
    return out


class _Window(io.RawIOBase):
    """One byte range of a file as a stream.

    A frame-aligned range decodes on its own only if the reader stops at the end of it,
    which a plain handle will not do.
    """

    def __init__(self, path: str, offset: int, size: int):
        self._handle = open(path, "rb")
        self._handle.seek(offset)
        self._left = size

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        if self._left <= 0:
            return 0
        view = memoryview(buffer)
        if len(view) > self._left:
            view = view[: self._left]
        read = self._handle.readinto(view)
        self._left -= read
        return read

    def close(self):
        try:
            self._handle.close()
        finally:
            super().close()


def _lz4_window(path: str, offset: int, size: int):
    return lz4.frame.open(io.BufferedReader(_Window(path, offset, size)), "rb")


def _concatenate(parts: list[str], target: str):
    """lz4 frames concatenate, so joining the parts is a byte copy.

    Through `cat` rather than a read/write loop: it moves the bytes inside the kernel,
    which on a file this size is the difference between the copy costing more than the
    pass and costing a fraction of it.
    """
    subprocess.run(["bash", "-c", "cat " + " ".join(quote(part) for part in parts)
                    + " > " + quote(target)], check=False)
    for part in parts:
        os.remove(part)


# Set before the workers are forked, so a table of over a gigabyte is inherited rather
# than pickled to each of them.
_SHARED: dict[str, Any] = {}
# Records a worker gets through between two touches of the shared progress counter.
_PROGRESS_STRIDE = 2000
# Below this a pass is quicker than forking for it.
_PARALLEL_FLOOR = 200_000
# Records per output frame, as bytes of them. A pass splits the file it reads on frame
# boundaries, so writing one frame per worker would leave the next pass only as many
# pieces as this one had workers. Near what the chunk writers produce.
_FRAME_BYTES = 4 << 20


class _Framer:
    """A record sink that closes an lz4 frame every `_FRAME_BYTES` and starts another.

    Frames concatenate, so a file of many of them reads back as one stream while staying
    divisible for whatever pass comes next.
    """

    __slots__ = ("_handle", "_buffer", "_held", "_limit", "_level", "_wrote")

    def __init__(self, handle, limit: int = _FRAME_BYTES, level: int = 1):
        self._handle = handle
        self._buffer: list[bytes] = []
        self._held = 0
        self._limit = limit
        self._level = level
        self._wrote = False

    def write(self, record: bytes):
        self._buffer.append(record)
        self._held += len(record)
        if self._held >= self._limit:
            self.flush()

    def flush(self):
        if self._buffer:
            self._handle.write(lz4.frame.compress(b"".join(self._buffer),
                                                  compression_level=self._level))
            self._buffer.clear()
            self._held = 0
            self._wrote = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.flush()
        if not self._wrote:
            # No records at all: a phrase list that matched nothing still has to leave a
            # readable file, and an empty one is not an lz4 stream.
            self._handle.write(lz4.frame.compress(b""))
            self._wrote = True


def _work_split(path: str, workers: int) -> list[tuple[int, int]]:
    """Frame ranges to hand out, or one range meaning "do it here".

    A file written by anything other than the chunk writers is one frame, and a small
    one is not worth a pool, so both come back as a single segment.
    """
    if workers < 2 or os.path.getsize(path) < _PARALLEL_FLOOR:
        return [(0, os.path.getsize(path))]
    try:
        bounds = frame_bounds(path)
    except (ValueError, OSError):
        return [(0, os.path.getsize(path))]
    return _segments(bounds, workers)


def _run_segments(worker, path: str, segments: list[tuple[int, int]],
                  count: Optional[int], description: str):
    """Run `worker` over `segments` in a forked pool, drawing one bar over all of them.

    Returns what the workers returned, in segment order -- so a caller concatenating the
    part files out of it preserves the record order the passage merger depends on. The
    passes differ in how many counters and output files they produce, so aggregating is
    left to them.
    """
    ctx = mp.get_context("fork")
    progress = ctx.Value("q", 0)
    _SHARED["progress"] = progress
    jobs = [(path, offset, size, f"{path}.part{index:05d}")
            for index, (offset, size) in enumerate(segments)]
    with ctx.Pool(len(jobs)) as pool:
        pending = pool.map_async(worker, jobs)
        with tqdm(total=count, desc=description, leave=False) as bar:
            while not pending.ready():
                pending.wait(0.5)
                bar.update(min(progress.value, bar.total or progress.value) - bar.n)
            bar.update(min(progress.value, bar.total or progress.value) - bar.n)
        return pending.get()


def _bump(progress, since: int) -> int:
    """Add `since` to the shared counter if it is worth the lock. Returns what is left."""
    if progress is None:
        return 0
    if since >= _PROGRESS_STRIDE:
        with progress.get_lock():
            progress.value += since
        return 0
    return since


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


def _detect_records(lines, matcher, common_ngrams, ngram_doc_path, threshold,
                    output_file, filtered_passages, progress=None) -> tuple[int, int]:
    """One pass over `lines` applying whichever verdicts were asked for.

    `matcher` None leaves the phrase list out; `common_ngrams` None leaves the n-gram
    verdict out and keeps records as they came; both together is the pass that exists so
    two verdicts do not read the file twice. A phrase hit is filtered out and never
    reaches the n-gram test, which is the order the two had when they ran separately.

    Returns (passages filtered, banalities found).
    """
    # Only the fields the configured verdicts need: a record carries around a hundred.
    if matcher is None:
        decode = _DECODE_PASSAGE
    elif common_ngrams is None:
        decode = _DECODE_SOURCE
    else:
        decode = _DECODE_FILTERED
    filtering = matcher is not None
    flagging = common_ngrams is not None
    passages_filtered = 0
    banalities_found = 0
    since = 0
    # Results come grouped by source document, but not strictly: a frantext
    # run opens 2,325 distinct documents 3,642 times. A few documents of
    # history turns most of that back into a hit.
    loaded: dict[str, NgramDoc] = {}
    for line in lines:
        since += 1
        since = _bump(progress, since)
        passage = decode(line)
        if filtering and matcher.find_matches_as_strings(clean_text(passage.source_passage)):
            passages_filtered += 1
            filtered_passages.write(line)
            continue
        if not flagging:
            output_file.write(line)
            continue
        document = loaded.get(passage.source_ngrams)
        if document is None:
            document = NgramDoc(os.path.join(ngram_doc_path, passage.source_ngrams))
            if len(loaded) >= _DOCUMENTS_HELD:
                del loaded[next(iter(loaded))]
            loaded[passage.source_ngrams] = document
        low, high = document.span(
            int(passage.source_start_byte), int(passage.source_end_byte)
        )
        # if n % (or more) of ngrams are common ngrams
        banality = high > low and (
            common_ngrams.count_in(document.keys[low:high]) / (high - low) * 100 >= threshold
        )
        banalities_found += banality
        output_file.write(_with_banality(line, banality, passage.banality))
    if progress is not None and since:
        with progress.get_lock():
            progress.value += since
    return passages_filtered, banalities_found


def _detect_segment(job: tuple[str, int, int, str]) -> tuple[int, int, str, Optional[str]]:
    """One frame range through the configured verdicts.

    The phrase tree and the frequent-key table are inherited through the fork rather than
    passed: both are large and neither is written to.
    """
    path, offset, size, base = job
    matcher = _SHARED.get("matcher")
    keep_path = f"{base}.keep"
    filtered_path = f"{base}.filtered" if matcher is not None else None
    with contextlib.ExitStack() as stack:
        input_file = stack.enter_context(_lz4_window(path, offset, size))
        output_file = stack.enter_context(
            _Framer(stack.enter_context(open(keep_path, "wb"))))
        filtered_passages = None
        if filtered_path is not None:
            filtered_passages = stack.enter_context(
                _Framer(stack.enter_context(open(filtered_path, "wb"))))
        filtered, banal = _detect_records(
            input_file, matcher, _SHARED.get("common_ngrams"),
            _SHARED.get("ngram_doc_path", ""), _SHARED.get("threshold", 0.0),
            output_file, filtered_passages, _SHARED["progress"])
    return filtered, banal, keep_path, filtered_path


def _detect_pass(filepath: str, count: Optional[int], workers: int, description: str,
                 matcher=None, common_ngrams=None, ngram_doc_path: str = "",
                 threshold: float = 0.0) -> tuple[int, int]:
    """Rewrite the results file with whatever verdicts were given.

    A filtered file is written only when there is a phrase list to filter on, so a run
    that only flags banalities leaves no empty one behind.
    """
    keep = f"{filepath}.keep.lz4"
    filtered_name = (filepath.replace("alignments.jsonl", "filtered_passages.jsonl")
                     if matcher is not None else None)
    segments = _work_split(filepath, workers)

    if len(segments) < 2:
        with contextlib.ExitStack() as stack:
            input_file = stack.enter_context(lz4.frame.open(filepath))
            output_file = stack.enter_context(
                _Framer(stack.enter_context(open(keep, "wb"))))
            filtered_passages = None
            if filtered_name is not None:
                filtered_passages = stack.enter_context(
                    _Framer(stack.enter_context(open(filtered_name, "wb"))))
            filtered, banal = _detect_records(
                tqdm(input_file, total=count, desc=description, leave=False),
                matcher, common_ngrams, ngram_doc_path, threshold,
                output_file, filtered_passages)
        os.replace(keep, filepath)
        return filtered, banal

    _SHARED.update(matcher=matcher, common_ngrams=common_ngrams,
                   ngram_doc_path=ngram_doc_path, threshold=threshold)
    results = _run_segments(_detect_segment, filepath, segments, count, description)
    filtered = sum(result[0] for result in results)
    banal = sum(result[1] for result in results)
    _concatenate([result[2] for result in results], keep)
    if filtered_name is not None:
        _concatenate([result[3] for result in results], filtered_name)
    os.replace(keep, filepath)
    return filtered, banal


def banality_auto_detect(
    filepath: str,
    common_ngrams_file: str,
    ngram_doc_path: str,
    count: Optional[int],
    proportion: float,
    threshold: float,
    workers: int = 1,
):
    """Detect banalities automatically based on frequent ngram over-representation"""
    common_ngrams = load_common_ngrams(common_ngrams_file, proportion)
    _, banalities_found = _detect_pass(
        filepath, count, workers, "Running banality auto-detection...",
        common_ngrams=common_ngrams, ngram_doc_path=ngram_doc_path, threshold=threshold)
    return banalities_found


def clean_phrases(file: str):
    """Clean phrases for phrase-based banality detection"""
    with open(file, encoding="utf8") as input_file:
        for phrase in input_file:
            phrase = clean_text(phrase)
            if re.search(r"\w", phrase):
                yield phrase


def _phrase_tree(banality_phrases_path: str):
    """The Aho-Corasick tree over the phrase list, built before any fork."""
    print("Building tree for phrase-based banality detection...", end="", flush=True)
    matcher = ahocorasick_rs.AhoCorasick(clean_phrases(banality_phrases_path))
    print("\r", end="")
    return matcher


def phrase_matcher(filepath: str, banality_phrases_path: str, count: Optional[int],
                   workers: int = 1):
    """Detect banalities based on user provided phrases"""
    filtered, _ = _detect_pass(
        filepath, count, workers, "Running phrase-based banality detection...",
        matcher=_phrase_tree(banality_phrases_path))
    print("done")
    return filtered


def filter_and_flag(
    filepath: str,
    banality_phrases_path: str,
    common_ngrams_file: str,
    ngram_doc_path: str,
    count: Optional[int],
    proportion: float,
    threshold: float,
    workers: int = 1,
) -> tuple[int, int]:
    """Phrase filtering and automatic banality detection in one pass.

    Run one after the other they read and rewrite the whole result file twice, and decode
    each record twice, for two verdicts that need one decode between them. Returns
    (passages filtered, banalities found); the files written are the ones the two passes
    write separately, with the same contents.
    """
    return _detect_pass(
        filepath, count, workers, "Filtering passages and detecting banalities...",
        matcher=_phrase_tree(banality_phrases_path),
        common_ngrams=load_common_ngrams(common_ngrams_file, proportion),
        ngram_doc_path=ngram_doc_path, threshold=threshold)


def _separate_records(lines, output_file, banal_output_file, progress=None) -> int:
    """Send every record of `lines` to one file or the other on its banality flag."""
    banalities_separated = 0
    since = 0
    for line in lines:
        since += 1
        since = _bump(progress, since)
        if _DECODE_VERDICT(line).banality is True:
            banalities_separated += 1
            banal_output_file.write(line)
        else:
            output_file.write(line)
    if progress is not None and since:
        with progress.get_lock():
            progress.value += since
    return banalities_separated


def _separate_segment(job: tuple[str, int, int, str]) -> tuple[int, str, str]:
    """One frame range sorted into kept and banal."""
    path, offset, size, base = job
    keep_path, banal_path = f"{base}.keep", f"{base}.banal"
    with (
        _lz4_window(path, offset, size) as input_file,
        open(keep_path, "wb") as keep_raw,
        _Framer(keep_raw) as output_file,
        open(banal_path, "wb") as banal_raw,
        _Framer(banal_raw) as banal_output_file,
    ):
        separated = _separate_records(input_file, output_file, banal_output_file,
                                      _SHARED["progress"])
    return separated, keep_path, banal_path


def separate_banalities(filepath: str, count: Optional[int], workers: int = 1) -> int:
    """
    Separate passages flagged as banal into a separate file and remove them from main alignments.
    Should be called AFTER all banality detection and LLM evaluation is complete.

    Args:
        filepath: Path to alignments file
        count: Total number of alignments (for progress bar)

    Returns:
        Number of banalities separated
    """
    banal_file_name = filepath.replace("alignments.jsonl", "banal_alignments.jsonl")
    keep = f"{filepath}.keep.lz4"
    segments = _work_split(filepath, workers)

    if len(segments) < 2:
        with (
            open(banal_file_name, "wb") as banal_raw,
            _Framer(banal_raw) as banal_output_file,
            open(keep, "wb") as keep_raw,
            _Framer(keep_raw) as output_file,
            lz4.frame.open(filepath) as input_file,
        ):
            banalities_separated = _separate_records(
                tqdm(input_file, total=count, desc="Separating banalities...", leave=False),
                output_file, banal_output_file)
        os.replace(keep, filepath)
        return banalities_separated

    results = _run_segments(_separate_segment, filepath, segments, count,
                            "Separating banalities...")
    banalities_separated = sum(result[0] for result in results)
    _concatenate([result[1] for result in results], keep)
    _concatenate([result[2] for result in results], banal_file_name)
    os.replace(keep, filepath)
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
