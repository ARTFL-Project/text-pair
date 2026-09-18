"""Corpus-wide n-gram index, built from the per-document `temp/*` files.

  index.tab              `ngram<TAB>key` for every distinct n-gram, lexicographic.
                         Read only by the aligner's --debug tracer, which builds a
                         key -> ngram dict, so the order is for bisecting by hand
                         rather than for a consumer.
  most_common_ngrams.txt keys by corpus frequency, descending. The banality filter
                         reads the first `proportion` percent, so this order is
                         load-bearing.

Workers write each temp file sorted, so counting is a merge rather than a sort:
`LC_ALL=C sort -m | uniq -c`. LC_ALL=C is not optional -- byte order is the
collation the workers sorted in, and `sort -m` trusts its inputs, so any other
collation splits runs silently. GNU sort takes the whole file list through
--files0-from; BSD sort has none, so there the list is merged in batches of file
arguments, in rounds.

Frequency ordering is a counting sort into per-count buckets, so that at 100k+
text objects nothing proportional to the corpus is resident.

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

# Keys are mmh3 32-bit hashes, signed.
KEY_DTYPE = np.int32
# Keys buffered per bucket before hitting the disk.
BUCKET_FLUSH = 1 << 16

# Ceiling on files handed to one sort -m. BSD sort's behaviour with thousands of
# inputs is not something to rely on, so batches stay moderate and rounds do the
# rest.
MAX_BATCH = 1024
# Descriptors to leave for everything else when sizing a batch.
FD_MARGIN = 64


def build(output_path: str) -> int:
    """Merge temp/* into index/index.tab and index/most_common_ngrams.txt.

    Returns the number of distinct n-grams.
    """
    index_path = os.path.join(output_path, "index", "index.tab")
    common_path = os.path.join(output_path, "index", "most_common_ngrams.txt")
    scratch = os.path.join(output_path, "index_scratch")
    shutil.rmtree(scratch, ignore_errors=True)
    os.makedirs(scratch, exist_ok=True)

    try:
        paths = sorted(glob(os.path.join(output_path, "temp", "*")))
        if not paths:
            open(index_path, "wb").close()
            open(common_path, "wb").close()
            return 0

        buckets = _Buckets(scratch)
        distinct = 0
        with open(index_path, "wb") as index_file:
            write = index_file.write
            add = buckets.add
            for line, count in _counted(paths, scratch):
                key = _key_of(line)
                if key is None:
                    continue
                write(line)
                write(b"\n")
                add(key, count)
                distinct += 1
        buckets.write_descending(common_path)
        return distinct
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


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


def _counted(paths: list[str], scratch: str) -> Iterator[tuple[bytes, int]]:
    """Merge the worker files and count adjacent runs, as (line, occurrences).

    uniq -c writes the count right-aligned then exactly one space, so it is taken
    off by hand: an n-gram can begin with spaces of its own, which a whitespace
    class would eat.
    """
    program, supports_listing = sort_program()
    environment = {**os.environ, **SORT_ENV}

    if not supports_listing:
        paths = _reduce(paths, scratch, program, environment)

    if supports_listing:
        listing = os.path.join(scratch, "filelist0")
        with open(listing, "wb") as handle:
            for path in paths:
                handle.write(path.encode("utf8") + b"\0")
        command = [program, "-m", f"--files0-from={listing}", "-T", scratch]
    else:
        command = [program, "-m", "-T", scratch, *paths]

    sort = subprocess.Popen(command, stdout=subprocess.PIPE, env=environment)
    uniq = subprocess.Popen(["uniq", "-c"], stdin=sort.stdout,
                            stdout=subprocess.PIPE, env=environment)
    if sort.stdout is not None:
        sort.stdout.close()
    previous = b""
    try:
        for raw in uniq.stdout:  # type: ignore[union-attr]
            count, _, line = raw.lstrip(b" ").partition(b" ")
            if line.endswith(b"\n"):
                line = line[:-1]
            # sort -m trusts its inputs to be sorted. If a worker ever wrote an
            # unsorted file the runs would split silently, so check instead.
            if line < previous:
                raise RuntimeError(
                    "ngram index inputs are not in LC_ALL=C order "
                    f"({line!r} followed {previous!r}); the merge cannot be trusted"
                )
            previous = line
            yield line, int(count)
    finally:
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


def _key_of(line: bytes) -> int | None:
    """The key from a `ngram<TAB>key` line, or None if the line is not one."""
    ngram, tab, key = line.rpartition(b"\t")
    if not tab or not ngram:
        return None
    try:
        return int(key)
    except ValueError:
        return None


# ----------------------------------------------------------------------
# Frequency ordering by counting sort
# ----------------------------------------------------------------------


def _band(count: int) -> int:
    """Bucket index for a count. 1..EXACT_MAX are exact; above that, log bands."""
    if count <= EXACT_MAX:
        return count
    return EXACT_MAX + max(int(count).bit_length() - 8, 1)


class _Buckets:
    """Append-only key buckets on disk, one per count or count band."""

    def __init__(self, directory: str):
        self.directory = os.path.join(directory, "buckets")
        os.makedirs(self.directory, exist_ok=True)
        self.pending: dict[int, list[tuple[int, int]]] = {}
        self.handles: dict[int, object] = {}

    def add(self, key: int, count: int) -> None:
        band = _band(count)
        pending = self.pending.get(band)
        if pending is None:
            pending = self.pending[band] = []
        pending.append((count, key))
        if len(pending) >= BUCKET_FLUSH:
            self._flush(band)

    def _flush(self, band: int) -> None:
        pending = self.pending.get(band)
        if not pending:
            return
        handle = self.handles.get(band)
        if handle is None:
            handle = self.handles[band] = open(os.path.join(self.directory, str(band)), "wb")
        if band <= EXACT_MAX:
            # The count is implied by the bucket, so only keys are stored.
            handle.write(np.fromiter((key for _, key in pending), dtype=KEY_DTYPE,
                                     count=len(pending)).tobytes())
        else:
            block = np.empty((len(pending), 2), dtype=np.int64)
            block[:, 0] = [count for count, _ in pending]
            block[:, 1] = [key for _, key in pending]
            handle.write(block.tobytes())
        pending.clear()

    def write_descending(self, path: str) -> None:
        """Concatenate buckets highest count first, into text."""
        for band in list(self.pending):
            self._flush(band)
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
