#!/usr/bin/env python3
"""Parallel banality detection must produce exactly what the serial pass produces.

    test_banality_parallel.py

The pass splits the results file on lz4 frame boundaries and rewrites each range in its
own process, so what is checked here is that the split, the windowed reads, the
re-framing and the concatenation put the records back byte for byte and in order --
`alignment_merger.first_step_merge` reads that stream once and flushes whenever
source_doc_id changes, so a record out of place is a document merged twice.

Everything is synthesised: a frame layout to split, an n-gram index to look up and a
frequent-key file to count against. No corpus needed.
"""
import os
import shutil
import struct
import sys
import tempfile

import lz4.frame
import numpy as np
import orjson

from textpair.sequence_alignment import banality_finder as bf
from textpair.sequence_alignment import ngram_binary

FAILURES = []


def check(label, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        FAILURES.append(label)


def write_ngram_doc(path, n_ngrams, keys):
    """An ngrams_in_order file, as ngram_binary documents it: the magic, the count, the
    int64 keys in n-gram order, then their ascending int32 start bytes."""
    starts = np.arange(n_ngrams, dtype=np.int32) * 10
    with open(path, "wb") as handle:
        handle.write(ngram_binary.ORDER_MAGIC)
        handle.write(struct.pack("<q", n_ngrams))
        handle.write(keys.astype("<i8").tobytes())
        handle.write(starts.astype("<i4").tobytes())


def build_corpus(root, documents=6, ngrams=400):
    """An n-gram index, a frequent-key file, and the keys that are frequent."""
    order = os.path.join(root, "ngrams_in_order")
    os.makedirs(order, exist_ok=True)
    common = np.arange(1, 51, dtype=np.int64)          # keys 1..50 are "frequent"
    for doc in range(documents):
        # Half the keys frequent, half not, so verdicts differ between passages.
        keys = np.where(np.arange(ngrams) % 2 == 0,
                        np.arange(ngrams) % 50 + 1,
                        np.arange(ngrams) + 1000)
        write_ngram_doc(os.path.join(order, f"{doc}.bin"), ngrams, keys)
    common_path = os.path.join(root, "most_common_ngrams.bin")
    common.tofile(common_path)
    return order, common_path


def write_alignments(path, frames, per_frame, documents=6, with_field=False):
    """A results file of `frames` lz4 frames, the shape the chunk writers leave behind."""
    record = 0
    with open(path, "wb") as raw:
        for frame in range(frames):
            lines = []
            for _ in range(per_frame):
                start = (record % 30) * 10
                body = {
                    "source_doc_id": str(record % documents),
                    "source_ngrams": f"{record % documents}.bin",
                    "source_start_byte": start,
                    "source_end_byte": start + 300,
                    "source_passage": f"passage {record}",
                }
                if with_field:
                    body["banality"] = False
                lines.append(orjson.dumps(body) + b"\n")
                record += 1
            raw.write(lz4.frame.compress(b"".join(lines), compression_level=1))
    return record


def records_of(path):
    with lz4.frame.open(path, "rb") as handle:
        return handle.read()


def run(work_dir, frames, per_frame, workers, with_field=False):
    """Returns (verdict count, decompressed bytes) for one configuration."""
    order, common = build_corpus(work_dir)
    path = os.path.join(work_dir, f"alignments_{frames}_{workers}.lz4")
    total = write_alignments(path, frames, per_frame, with_field=with_field)
    found = bf.banality_auto_detect(path, common, order, False, total, 100.0, 40.0, workers)
    return found, records_of(path), total


def main():
    root = tempfile.mkdtemp(prefix="textpair_banality_")
    try:
        # The frame layouts that matter: one frame cannot be split, many frames split
        # unevenly, and more workers than frames must not lose or duplicate a range.
        for frames, per_frame in ((1, 500), (7, 300), (64, 50), (5, 1)):
            serial = run(root, frames, per_frame, 1)
            for workers in (2, 8, 96):
                parallel = run(root, frames, per_frame, workers)
                check(f"{frames} frame(s) x {per_frame}, {workers} workers: "
                      f"same records as serial", serial[1] == parallel[1])
                check(f"{frames} frame(s) x {per_frame}, {workers} workers: "
                      f"same verdict count ({serial[0]})", serial[0] == parallel[0])
                check(f"{frames} frame(s) x {per_frame}, {workers} workers: "
                      f"every record kept ({serial[2]})",
                      parallel[1].count(b"\n") == serial[2])

        # A record that already carries the field takes the decode-and-re-encode path
        # rather than the splice, in both passes.
        serial = run(root, 9, 100, 1, with_field=True)
        parallel = run(root, 9, 100, 16, with_field=True)
        check("existing banality field: same records as serial", serial[1] == parallel[1])
        check("existing banality field: same verdict count", serial[0] == parallel[0])

        # Frame boundaries have to be found without decompressing anything.
        path = os.path.join(root, "bounds.lz4")
        written = write_alignments(path, 11, 40)
        bounds = bf.frame_bounds(path)
        check("frame_bounds finds every frame", len(bounds) == 11)
        check("frame_bounds accounts for every byte",
              sum(length for _, length in bounds) == os.path.getsize(path))
        check("frame_bounds ranges are contiguous",
              all(bounds[i][0] + bounds[i][1] == bounds[i + 1][0]
                  for i in range(len(bounds) - 1)))
        # Each range has to decode on its own, which is what lets a worker take one.
        middle = bounds[5]
        with bf._lz4_window(path, middle[0], middle[1]) as handle:
            check("a single frame range decodes alone",
                  handle.read().count(b"\n") == 40)
        check("the file is still readable as one stream",
              records_of(path).count(b"\n") == written)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("all checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
