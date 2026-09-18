"""Corpus-wide n-gram index.

  most_common_ngrams.txt keys by corpus frequency, descending. The banality
                         filter reads the first `proportion` percent, so this
                         order is load-bearing. Always written.
  index.tab              `ngram<TAB>key` for every distinct n-gram. Read only by
                         the aligner's --debug tracer, which builds a
                         key -> ngram dict, so it is written only when the
                         generator was asked for it -- the same `debug` flag the
                         tracer is gated on.

Frequencies come from `ngrams/*.bin`, not from n-gram text: each document's CSR
already holds its distinct keys and the offsets that give their counts, so the
whole corpus is a numpy aggregation over integer columns. Counting the text
instead meant a per-distinct-n-gram Python loop, which at 70M n-grams was most
of the generation stage.

It is also the frequency the aligner acts on. Its inverted index is over keys,
so a key's corpus frequency is the sum over the n-grams that hash to it -- with
a 64-bit key that is one n-gram short of certainty, 1.3e-04 collisions expected
over 70M distinct n-grams, where a 32-bit key had 566,635 of them.

Nothing proportional to the corpus stays resident: the aggregation spills by key
range, and frequency ordering is a counting sort into per-count buckets rather
than a sort.

See PREPROCESSING_REWRITE.md for what this replaced and why.
"""

from __future__ import annotations

import os
import resource
import shutil
import subprocess
import tempfile
from glob import glob
from typing import Iterator

import numpy as np

# Counts at or below this get an exact bucket; above it, power-of-two bands.
EXACT_MAX = 255

# Key-range buckets for the counting pass. Each holds total_postings / KEY_BUCKETS
# (key, count) pairs, so this is what bounds memory during aggregation.
KEY_BITS = 8
KEY_BUCKETS = 1 << KEY_BITS
_KEY_EDGES = [((bucket << (64 - KEY_BITS)) - (1 << 63)) for bucket in range(KEY_BUCKETS + 1)]

# Keys are the low 64 bits of mmh3's 128-bit hash, signed. Counts stay int32: they are
# per-document occurrence counts, and widening them would inflate the spill for nothing.
KEY_DTYPE = np.int64
COUNT_DTYPE = np.int32
# Ceiling on files handed to one sort -m. BSD sort's behaviour with thousands of
# inputs is not something to rely on, so batches stay moderate and rounds do the
# rest.
MAX_BATCH = 1024
# Descriptors to leave for everything else when sizing a batch.
FD_MARGIN = 64


def build(output_path: str, write_index_tab: bool = False) -> int:
    """Write index/most_common_ngrams.txt, and index/index.tab when asked.

    Returns the number of distinct keys.
    """
    index_dir = os.path.join(output_path, "index")
    os.makedirs(index_dir, exist_ok=True)
    common_path = os.path.join(index_dir, "most_common_ngrams.txt")
    index_path = os.path.join(index_dir, "index.tab")
    scratch = os.path.join(output_path, "index_scratch")
    shutil.rmtree(scratch, ignore_errors=True)
    os.makedirs(scratch, exist_ok=True)

    try:
        binaries = sorted(glob(os.path.join(output_path, "ngrams", "*.bin")))
        buckets = _Buckets(scratch)
        distinct = 0
        for keys, totals in _aggregate(binaries, scratch):
            buckets.add_many(keys, totals)
            distinct += keys.size
        buckets.write_descending(common_path)

        if write_index_tab:
            _write_index_tab(output_path, index_path, scratch)
        elif os.path.exists(index_path):
            # A stale index.tab from an earlier run would name the wrong ngrams.
            os.remove(index_path)
        return distinct
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ----------------------------------------------------------------------
# Frequencies, from the per-document CSR indexes
# ----------------------------------------------------------------------


def _aggregate(binaries: list[str], scratch: str) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Total occurrences per key, a key range at a time, ascending.

    Pass one spills each document's (key, count) columns into the bucket its keys
    fall in; pass two aggregates one bucket at a time. Keys arrive sorted from the
    CSR, so splitting a document across buckets is a searchsorted and some slices.
    """
    from . import ngram_binary

    directory = os.path.join(scratch, "keys")
    os.makedirs(directory, exist_ok=True)
    key_files = [open(os.path.join(directory, f"{b}.k"), "wb") for b in range(KEY_BUCKETS)]
    count_files = [open(os.path.join(directory, f"{b}.c"), "wb") for b in range(KEY_BUCKETS)]
    interior = np.array(_KEY_EDGES[1:-1], dtype=np.int64)
    try:
        for path in binaries:
            with open(path, "rb") as handle:
                buffer = handle.read()
            keys, offsets = ngram_binary.columns(buffer, path)[:2]
            if keys.size == 0:
                continue
            # A TPNG0001 index hands back int32 keys; the spill is int64 throughout.
            keys = keys.astype(KEY_DTYPE, copy=False)
            counts = (offsets[1:] - offsets[:-1]).astype(COUNT_DTYPE)
            splits = np.searchsorted(keys, interior)
            previous = 0
            for bucket, stop in enumerate(np.append(splits, keys.size)):
                if stop > previous:
                    key_files[bucket].write(keys[previous:stop].tobytes())
                    count_files[bucket].write(counts[previous:stop].tobytes())
                previous = stop
    finally:
        for handle in (*key_files, *count_files):
            handle.close()

    for bucket in range(KEY_BUCKETS):
        key_path = os.path.join(directory, f"{bucket}.k")
        count_path = os.path.join(directory, f"{bucket}.c")
        keys = np.fromfile(key_path, dtype=KEY_DTYPE)
        os.remove(key_path)
        if keys.size == 0:
            os.remove(count_path)
            continue
        counts = np.fromfile(count_path, dtype=COUNT_DTYPE)
        os.remove(count_path)
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        counts = counts[order]
        starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        yield keys[starts], np.add.reduceat(counts.astype(np.int64), starts)


# ----------------------------------------------------------------------
# Merging and counting
# ----------------------------------------------------------------------

# Byte ordering, and the collation the workers sorted in.
SORT_ENV = {"LC_ALL": "C", "LANG": "C"}

_SORT: tuple[str, bool] | None = None


def sort_program() -> tuple[str, bool]:
    """(program, supports --files0-from). Probed once.

    Probed by merging two real files, because GNU sort fails on an empty file
    list and so cannot be probed with one.
    """
    global _SORT
    if _SORT is not None:
        return _SORT
    environment = {**os.environ, **SORT_ENV}
    with tempfile.TemporaryDirectory() as probe_dir:
        first = os.path.join(probe_dir, "a")
        second = os.path.join(probe_dir, "b")
        listing = os.path.join(probe_dir, "list")
        with open(first, "wb") as handle:
            handle.write(b"a\nc\n")
        with open(second, "wb") as handle:
            handle.write(b"b\nd\n")
        with open(listing, "wb") as handle:
            handle.write(first.encode() + b"\0" + second.encode() + b"\0")
        for candidate in ("sort", "gsort"):
            merged = _probe(candidate, ["-m", first, second], environment)
            if merged != b"a\nb\nc\nd\n":
                continue
            listed = _probe(candidate, ["-m", f"--files0-from={listing}"], environment)
            _SORT = (candidate, listed == b"a\nb\nc\nd\n")
            return _SORT
    raise RuntimeError(
        "no working sort(1) found: the ngram index is built with `sort -m`. "
        "Install coreutils, or put a POSIX sort on PATH."
    )


def _probe(program: str, arguments: list[str], environment: dict) -> bytes | None:
    try:
        result = subprocess.run([program, *arguments], capture_output=True,
                                timeout=30, env=environment)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _batch_size(paths: list[str]) -> int:
    """How many files one sort invocation can take by argument and fd limits."""
    try:
        arg_max = os.sysconf("SC_ARG_MAX")
    except (AttributeError, ValueError, OSError):
        arg_max = 1 << 18
    environment_size = sum(len(name) + len(value) + 2 for name, value in os.environ.items())
    longest = max(len(path) for path in paths) + 1
    by_arguments = max((arg_max - environment_size - 8192) // longest, 2)
    by_descriptors = max(_descriptor_limit() - FD_MARGIN, 2)
    return min(by_arguments, by_descriptors, MAX_BATCH)


def _descriptor_limit() -> int:
    """The soft descriptor limit, raised toward the hard one if it is low.

    macOS ships a soft limit of 256, which would otherwise force several more
    merge rounds than the machine actually needs.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = MAX_BATCH + FD_MARGIN + 64
    if soft < target:
        ceiling = target if hard == resource.RLIM_INFINITY else min(target, hard)
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (ceiling, hard))
            soft = ceiling
        except (ValueError, OSError):
            pass
    return soft


def _write_index_tab(output_path: str, index_path: str, scratch: str) -> None:
    """Merge the per-document n-gram lists into index.tab.

    Each `temp/{id}` is written sorted by the worker that produced it, so this is
    a merge and a dedupe, entirely in sort(1) and uniq(1) -- no counting, since
    the frequencies come from the binary indexes. `cat` would be wrong here: over
    files that do not end in a newline it welds the last n-gram of one document
    to the first of the next, which is where the non-numeric keys that
    banality_finder and tracing still guard against came from. sort -m treats a
    final incomplete line as a line.
    """
    paths = sorted(glob(os.path.join(output_path, "temp", "*")))
    if not paths:
        open(index_path, "wb").close()
        return
    program, supports_listing = sort_program()
    environment = {**os.environ, **SORT_ENV}
    if not supports_listing:
        paths = _reduce(paths, scratch, program, environment)
        command = [program, "-m", "-T", scratch, *paths]
    else:
        listing = os.path.join(scratch, "filelist0")
        with open(listing, "wb") as handle:
            for path in paths:
                handle.write(path.encode("utf8") + b"\0")
        command = [program, "-m", f"--files0-from={listing}", "-T", scratch]

    with open(index_path, "wb") as index_file:
        sort = subprocess.Popen(command, stdout=subprocess.PIPE, env=environment)
        uniq = subprocess.Popen(["uniq"], stdin=sort.stdout, stdout=index_file,
                                env=environment)
        if sort.stdout is not None:
            sort.stdout.close()
        _finish(uniq, "uniq")
        _finish(sort, program)


def _reduce(paths: list[str], scratch: str, program: str,
            environment: dict) -> list[str]:
    """Merge in batches until one invocation can take what is left.

    Only needed where sort has no --files0-from: 100k paths do not fit in a
    command line. Intermediates hold merged lines with duplicates intact, so the
    counting stays a single uniq -c at the end.
    """
    batch = _batch_size(paths)
    level = 0
    while len(paths) > batch:
        outputs = []
        directory = os.path.join(scratch, f"level{level}")
        os.makedirs(directory, exist_ok=True)
        for number in range(0, len(paths), batch):
            group = paths[number : number + batch]
            output = os.path.join(directory, str(number // batch))
            with open(output, "wb") as handle:
                subprocess.run([program, "-m", "-T", scratch, *group],
                               stdout=handle, env=environment, check=True)
            outputs.append(output)
        if level:
            shutil.rmtree(os.path.join(scratch, f"level{level - 1}"), ignore_errors=True)
        paths = outputs
        level += 1
    return paths


def _finish(process: subprocess.Popen, name: str) -> None:
    if process.stdout is not None:
        process.stdout.close()
    process.wait()
    if process.returncode not in (0, None):
        raise RuntimeError(f"{name} exited {process.returncode} while building the ngram index")


# ----------------------------------------------------------------------
# Frequency ordering by counting sort
# ----------------------------------------------------------------------


def _bit_length(counts: np.ndarray) -> np.ndarray:
    """int.bit_length(), elementwise."""
    bits = np.floor(np.log2(np.maximum(counts, 1))).astype(np.int64) + 1
    # log2 in double can land a hair either side of an exact power of two, so
    # correct any off-by-one rather than trusting it.
    one = np.int64(1)
    bits[counts >= (one << bits)] += 1
    low = (bits > 1) & (counts < (one << (bits - 1)))
    bits[low] -= 1
    return bits


class _Buckets:
    """Append-only key buckets on disk, one per count or count band."""

    def __init__(self, directory: str):
        self.directory = os.path.join(directory, "buckets")
        os.makedirs(self.directory, exist_ok=True)
        self.handles: dict[int, object] = {}

    def add_many(self, keys: np.ndarray, counts: np.ndarray) -> None:
        """Append a block of keys, each to the bucket for its count."""
        bands = np.where(
            counts <= EXACT_MAX,
            counts,
            EXACT_MAX + np.maximum(_bit_length(counts) - 8, 1),
        )
        order = np.argsort(bands, kind="stable")
        bands = bands[order]
        keys = keys[order]
        counts = counts[order]
        starts = np.flatnonzero(np.r_[True, bands[1:] != bands[:-1]])
        stops = np.r_[starts[1:], bands.size]
        for start, stop in zip(starts, stops):
            band = int(bands[start])
            handle = self._handle(band)
            if band <= EXACT_MAX:
                # The count is implied by the bucket, so only keys are stored.
                handle.write(keys[start:stop].astype(KEY_DTYPE).tobytes())
            else:
                block = np.empty((stop - start, 2), dtype=np.int64)
                block[:, 0] = counts[start:stop]
                block[:, 1] = keys[start:stop]
                handle.write(block.tobytes())

    def _handle(self, band: int):
        handle = self.handles.get(band)
        if handle is None:
            handle = self.handles[band] = open(os.path.join(self.directory, str(band)), "wb")
        return handle

    def write_descending(self, path: str) -> None:
        """Concatenate buckets highest count first, into text."""
        for handle in self.handles.values():
            handle.close()
        with open(path, "wb") as output:
            for band in sorted(self.handles, reverse=True):
                bucket_path = os.path.join(self.directory, str(band))
                if band <= EXACT_MAX:
                    keys = np.fromfile(bucket_path, dtype=KEY_DTYPE)
                else:
                    block = np.fromfile(bucket_path, dtype=np.int64).reshape(-1, 2)
                    # Stable, so equal counts keep the lexicographic order they
                    # were appended in.
                    block = block[np.argsort(-block[:, 0], kind="stable")]
                    keys = block[:, 1]
                _write_keys(output, keys)
                os.remove(bucket_path)


def _write_keys(output, keys: np.ndarray) -> None:
    """Keys as one decimal per line, formatted in blocks."""
    block_size = 1 << 18
    for start in range(0, keys.size, block_size):
        block = keys[start : start + block_size].tolist()
        output.write(("\n".join(map(str, block)) + "\n").encode("ascii"))
