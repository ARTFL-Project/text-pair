"""Corpus-wide n-gram index.

  most_common_ngrams.bin int64 keys by corpus frequency, descending. The
                         banality filter reads the first `proportion` percent,
                         so this order is load-bearing. Always written. Binary
                         because formatting 77M keys as decimal was a third of
                         this stage, and the reader turns them straight back
                         into ints.
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
than a sort. The spill runs while the n-gram stage is still writing documents,
so what is left when the workers stop is the totalling.

Equally frequent keys are in no particular order -- they are hashes, and
imposing one would mean a sort the rest of this avoids -- so which of them
lands inside a `proportion` cut depends on the order documents were indexed in.

See PREPROCESSING_REWRITE.md for what this replaced and why.
"""

from __future__ import annotations

import os
import resource
import shutil
import subprocess
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from queue import Queue
from threading import Thread
from typing import Iterator

import numpy as np
from numba import njit

# Counts at or below this get an exact bucket; above it, power-of-two bands.
EXACT_MAX = 255

# Key-range buckets for the counting pass. Each holds total_postings / KEY_BUCKETS
# (key, count) pairs, so this is what bounds memory during aggregation.
KEY_BITS = 8
KEY_BUCKETS = 1 << KEY_BITS

# Entries gathered per bucket before a write. 256 buckets at this size is ~130MB
# of buffer at worst, against nearly a million small writes without it.
SPILL_BLOCK = 1 << 16

# Threads for the index passes. Past about eight the file reads stop being the
# limit and more only adds contention.
_SPILL_THREADS = min(8, (os.cpu_count() or 4))
# Documents handed to one lane before moving on to the next, when they are
# spilled as they arrive. Large enough not to wake a lane per document, small
# enough that every lane has work early in a run.
_LANE_RUN = 32
# Buckets aggregated ahead of the consumer. Each holds its key range's distinct
# keys, so this is what bounds the aggregation's memory.
_BUCKET_LOOKAHEAD = 8
_KEY_EDGES = [((bucket << (64 - KEY_BITS)) - (1 << 63)) for bucket in range(KEY_BUCKETS + 1)]

# Keys are the low 64 bits of mmh3's 128-bit hash, signed. Counts stay int32: they are
# per-document occurrence counts, and widening them would inflate the spill for nothing.
KEY_DTYPE = np.int64

# The frequency-ordered key list, read by the banality filter.
COMMON_NGRAMS = "most_common_ngrams.bin"
COUNT_DTYPE = np.int32
# Ceiling on files handed to one sort -m. BSD sort's behaviour with thousands of
# inputs is not something to rely on, so batches stay moderate and rounds do the
# rest.
MAX_BATCH = 1024
# Descriptors to leave for everything else when sizing a batch.
FD_MARGIN = 64


def build(output_path: str, write_index_tab: bool = False) -> int:
    """Write index/most_common_ngrams.bin, and index/index.tab when asked.

    For a tree whose per-document n-grams are already on disk. The n-gram stage
    feeds IncrementalIndex while it writes them instead, which is the same work
    with the workers to overlap it with; this is that path with everything
    handed over at once.

    Returns the number of distinct keys.
    """
    index = IncrementalIndex(output_path)
    for path in sorted(glob(os.path.join(output_path, "ngrams", "*.bin"))):
        index.add(path)
    return index.finish(write_index_tab=write_index_tab)


class IncrementalIndex:
    """Buckets each document's keys as it is written, rather than afterwards.

    Pass one of the aggregation is per-document, and the process running it sits
    at under a fifth of a core while the workers run, so it happens then: by the
    time the last document lands most of the corpus is already spilled and only
    the totalling is left.

    Which lane a document goes to follows the order documents are added, not
    when a lane got round to its queue, so the totals are summed in a fixed
    order. That order is not the one `build` uses, so the two disagree about
    which of two equally frequent keys comes first; they agree on every count.
    """

    def __init__(self, output_path: str, threads: int | None = None):
        self.output_path = output_path
        self.scratch = os.path.join(output_path, "index_scratch")
        shutil.rmtree(self.scratch, ignore_errors=True)
        self.directory = os.path.join(self.scratch, "keys")
        os.makedirs(self.directory, exist_ok=True)
        self.lanes = max(threads or _SPILL_THREADS, 1)
        self._added = 0
        self._failure: BaseException | None = None
        self._queues: list[Queue] = [Queue() for _ in range(self.lanes)]
        self._threads = [
            Thread(target=self._drain, args=(number,), daemon=True)
            for number in range(self.lanes)
        ]
        for thread in self._threads:
            thread.start()

    def add(self, path: str) -> None:
        """Hand over one document, once its n-gram index has been written."""
        self._queues[(self._added // _LANE_RUN) % self.lanes].put(path)
        self._added += 1

    def _drain(self, number: int) -> None:
        lane = _Lane(self.directory, number)
        queue = self._queues[number]
        while True:
            path = queue.get()
            if path is None:
                break
            # Keep draining after a failure so no producer blocks; finish raises.
            if self._failure is None:
                try:
                    lane.add(path)
                except BaseException as error:
                    self._failure = error
        try:
            lane.close()
        except BaseException as error:
            if self._failure is None:
                self._failure = error

    def finish(self, write_index_tab: bool = False) -> int:
        """Total the spilled counts and write the frequency-ordered key file.

        Returns the number of distinct keys.
        """
        for queue in self._queues:
            queue.put(None)
        for thread in self._threads:
            thread.join()
        try:
            if self._failure is not None:
                raise self._failure
            index_dir = os.path.join(self.output_path, "index")
            os.makedirs(index_dir, exist_ok=True)
            index_path = os.path.join(index_dir, "index.tab")
            buckets = _Buckets(self.scratch)
            distinct = 0
            for output in _total_spilled(self.directory, self.lanes, self.lanes):
                buckets.add_banded(output)
                distinct += output[0].size
            buckets.write_descending(os.path.join(index_dir, COMMON_NGRAMS))

            if write_index_tab:
                _write_index_tab(self.output_path, index_path, self.scratch)
            elif os.path.exists(index_path):
                # A stale index.tab from an earlier run would name the wrong ngrams.
                os.remove(index_path)
            return distinct
        finally:
            shutil.rmtree(self.scratch, ignore_errors=True)


# ----------------------------------------------------------------------
# Frequencies, from the per-document CSR indexes
# ----------------------------------------------------------------------


def _total_spilled(directory: str, lanes: int, threads: int) -> Iterator[tuple]:
    """Total each key range across the lanes that spilled it, ascending.

    Runs ahead of the consumer by a bounded window: every bucket at once would
    hold the whole corpus's distinct keys in memory.
    """
    with ThreadPoolExecutor(max_workers=max(threads, 1)) as pool:
        pending: deque = deque()
        for bucket in range(KEY_BUCKETS):
            pending.append(pool.submit(_aggregate_bucket, directory, bucket, lanes))
            if len(pending) >= _BUCKET_LOOKAHEAD:
                result = pending.popleft().result()
                if result is not None:
                    yield result
        while pending:
            result = pending.popleft().result()
            if result is not None:
                yield result


class _Lane:
    """One lane of the spill: the key ranges of the documents given to it.

    Each document contributes a slice to most buckets, so writing slices straight
    out is one call per (document, bucket). They are gathered per bucket and
    flushed in blocks instead. A flush reopens its file rather than holding 512
    of them open: at a block per flush that is a few thousand opens over a run,
    against a descriptor count no macOS default would allow.
    """

    __slots__ = ("directory", "number", "keys", "counts", "sizes", "interior")

    def __init__(self, directory: str, number: int):
        self.directory = directory
        self.number = number
        self.keys: list[list[np.ndarray]] = [[] for _ in range(KEY_BUCKETS)]
        self.counts: list[list[np.ndarray]] = [[] for _ in range(KEY_BUCKETS)]
        self.sizes = [0] * KEY_BUCKETS
        self.interior = np.array(_KEY_EDGES[1:-1], dtype=np.int64)

    def add(self, path: str) -> None:
        """Split one document's keys across the ranges they fall in."""
        from . import ngram_binary

        with open(path, "rb") as handle:
            buffer = handle.read()
        keys, offsets = ngram_binary.columns(buffer, path)[:2]
        if keys.size == 0:
            return
        # A TPNG0001 index hands back int32 keys; the spill is int64 throughout.
        keys = keys.astype(KEY_DTYPE, copy=False)
        counts = (offsets[1:] - offsets[:-1]).astype(COUNT_DTYPE)
        splits = np.searchsorted(keys, self.interior)
        previous = 0
        for bucket, stop in enumerate(np.append(splits, keys.size)):
            if stop > previous:
                self.keys[bucket].append(keys[previous:stop])
                self.counts[bucket].append(counts[previous:stop])
                self.sizes[bucket] += stop - previous
                if self.sizes[bucket] >= SPILL_BLOCK:
                    self.flush(bucket)
            previous = stop

    def flush(self, bucket: int) -> None:
        if not self.sizes[bucket]:
            return
        stem = os.path.join(self.directory, f"{self.number}-{bucket}")
        with open(f"{stem}.k", "ab") as handle:
            handle.write(np.concatenate(self.keys[bucket]).tobytes())
        with open(f"{stem}.c", "ab") as handle:
            handle.write(np.concatenate(self.counts[bucket]).tobytes())
        self.keys[bucket].clear()
        self.counts[bucket].clear()
        self.sizes[bucket] = 0

    def close(self) -> None:
        for bucket in range(KEY_BUCKETS):
            self.flush(bucket)


def _aggregate_bucket(directory: str, bucket: int, lanes: int):
    """Sum one key range across every thread's spill, in thread order."""
    key_blocks = []
    count_blocks = []
    for lane in range(lanes):
        stem = os.path.join(directory, f"{lane}-{bucket}")
        # A lane that never flushed this key range left no file behind.
        if not os.path.exists(f"{stem}.k"):
            continue
        keys = np.fromfile(f"{stem}.k", dtype=KEY_DTYPE)
        os.remove(f"{stem}.k")
        if keys.size:
            key_blocks.append(keys)
            count_blocks.append(np.fromfile(f"{stem}.c", dtype=COUNT_DTYPE))
        os.remove(f"{stem}.c")
    if not key_blocks:
        return None
    keys = key_blocks[0] if len(key_blocks) == 1 else np.concatenate(key_blocks)
    counts = count_blocks[0] if len(count_blocks) == 1 else np.concatenate(count_blocks)
    return _band_sort(*_sum_by_key(keys, counts))


def _band_sort(keys: np.ndarray, counts: np.ndarray):
    """Order one key range's keys by the bucket their count falls in.

    Done here rather than in the consumer because it is nearly all of what
    bucketing costs, and there are threads sitting behind this one waiting to
    hand over. What is left for the consumer is the writes, which have to stay
    in key-range order and so cannot be threaded.
    """
    bands = np.where(
        counts <= EXACT_MAX,
        counts,
        EXACT_MAX + np.maximum(_bit_length(counts) - 8, 1),
    )
    order = np.argsort(bands, kind="stable")
    bands = bands[order]
    starts = np.flatnonzero(np.r_[True, bands[1:] != bands[:-1]])
    return keys[order], counts[order], bands, starts


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


@njit(nogil=True, cache=True)
def _accumulate(keys, counts, table_key, table_slot, out_keys, out_counts):
    """Sum counts per distinct key through an open-addressing table.

    Returns the number of distinct keys, written to the head of the out arrays
    in order of first appearance.
    """
    mask = np.uint64(table_key.size - 1)
    found = 0
    for i in range(keys.size):
        key = keys[i]
        # Fibonacci hashing: the keys are already hashes, but their low bits are
        # what the bucket split consumed, so they need spreading again.
        scattered = np.uint64(key) * np.uint64(0x9E3779B97F4A7C15)
        scattered ^= scattered >> np.uint64(29)
        slot = scattered & mask
        while True:
            at = table_slot[slot]
            if at == -1:
                table_slot[slot] = found
                table_key[slot] = key
                out_keys[found] = key
                out_counts[found] = counts[i]
                found += 1
                break
            if table_key[slot] == key:
                out_counts[at] += counts[i]
                break
            slot = (slot + np.uint64(1)) & mask
    return found


def _sum_by_key(keys: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distinct keys and their summed counts, in order of first appearance.

    Hashing rather than sorting: a bucket holds one entry per (document, key),
    several times what it holds distinct keys, and nothing downstream needs them
    ordered. Keys are hashes, so "ascending key" was never a meaningful order --
    it only decided which of several equally frequent keys landed either side of
    the banality filter's cut, and first-appearance order decides that just as
    arbitrarily and just as reproducibly. Sorting them back cost 3.2s of a 8.2s
    index build and moved 0.7% of the keys at a 10% cut.
    """
    size = 1 << max(int(keys.size * 2 - 1).bit_length(), 4)
    table_key = np.zeros(size, dtype=KEY_DTYPE)
    table_slot = np.full(size, -1, dtype=np.int64)
    out_keys = np.empty(keys.size, dtype=KEY_DTYPE)
    out_counts = np.zeros(keys.size, dtype=np.int64)
    found = _accumulate(keys, counts, table_key, table_slot, out_keys, out_counts)
    return out_keys[:found], out_counts[:found]


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

    def add_banded(self, block) -> None:
        """Append one key range's keys, each to the bucket for its count.

        Takes what _band_sort produced: keys and counts already in bucket
        order, and where each bucket's run begins.
        """
        keys, counts, bands, starts = block
        stops = np.r_[starts[1:], bands.size]
        for start, stop in zip(starts, stops):
            band = int(bands[start])
            handle = self._handle(band)
            if band <= EXACT_MAX:
                # The count is implied by the bucket, so only keys are stored.
                handle.write(keys[start:stop].astype(KEY_DTYPE).tobytes())
            else:
                pairs = np.empty((stop - start, 2), dtype=np.int64)
                pairs[:, 0] = counts[start:stop]
                pairs[:, 1] = keys[start:stop]
                handle.write(pairs.tobytes())

    def _handle(self, band: int):
        handle = self.handles.get(band)
        if handle is None:
            handle = self.handles[band] = open(os.path.join(self.directory, str(band)), "wb")
        return handle

    def write_descending(self, path: str) -> None:
        """Concatenate buckets highest count first."""
        for handle in self.handles.values():
            handle.close()
        with open(path, "wb") as output:
            for band in sorted(self.handles, reverse=True):
                bucket_path = os.path.join(self.directory, str(band))
                if band <= EXACT_MAX:
                    keys = np.fromfile(bucket_path, dtype=KEY_DTYPE)
                else:
                    block = np.fromfile(bucket_path, dtype=KEY_DTYPE).reshape(-1, 2)
                    # Stable, so equal counts keep the order they were appended in.
                    block = block[np.argsort(-block[:, 0], kind="stable")]
                    keys = np.ascontiguousarray(block[:, 1])
                output.write(keys.tobytes())
                os.remove(bucket_path)
