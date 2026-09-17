#!/usr/bin/env python3
"""Checks that a source document's chunk files stay together in `sort -V` order.

    test_chunk_order.py [--source-files DIR --source-metadata FILE
                         [--target-files DIR --target-metadata FILE]] [--threads N]

`alignment_merger.first_step_merge` reads the concatenated alignments once and flushes its
passage group whenever `source_doc_id` changes, so a document whose records arrive in two
separate stretches is merged twice as if it were two documents, silently. Nothing enforces
that; it follows from the chunk names carrying the source document first, and from the
`sort -V` the chunks are concatenated with (textpair/__main__.py).

The unit checks build chunk names with runner._jobs and sort them with the real `sort -V`.
Given a corpus, the end-to-end check aligns it, concatenates as the pipeline does and
asserts each source_doc_id occupies one contiguous run of records.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from textpair.sequence_alignment.aligner import align, runner
from textpair.sequence_alignment.aligner.kernels import NCOL


def sort_v(names):
    """The order the pipeline concatenates chunks in."""
    out = subprocess.run(["sort", "-zV"], input="\0".join(names) + "\0",
                         capture_output=True, text=True, check=True).stdout
    return [name for name in out.split("\0") if name]


def contiguous(names):
    """True if every source document's chunks form one unbroken run."""
    seen, previous = set(), None
    for name in names:
        source = name.split("-")[0]
        if source != previous:
            if source in seen:
                return False
            seen.add(source)
            previous = source
    return True


def synthetic(doc_ids, n_sources, threads, same_array):
    """Chunk names for an all-pairs result over `doc_ids`."""
    target_base = 0 if same_array else n_sources
    n_targets = len(doc_ids) - target_base
    rows = []
    for source in range(n_sources):
        first = source + 1 if same_array else target_base
        for target in range(first, len(doc_ids)):
            rows.append([source, target] + [0] * NCOL)
    if not rows:
        return []
    jobs = runner._jobs(np.array(rows, np.int32), (), doc_ids, n_targets, target_base,
                        threads, same_array)
    return [name for _slot, name, _lo, _hi in jobs]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-files", default="")
    parser.add_argument("--source-metadata", default="")
    parser.add_argument("--target-files", default="")
    parser.add_argument("--target-metadata", default="")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    failures = []

    def check(label, ok):
        print(f"{'PASS' if ok else 'FAIL'} {label}")
        if not ok:
            failures.append(label)

    numeric = [str(i) for i in range(1, 81)]
    # A document ID that is another's prefix is the case `sort -V` could interleave.
    prefixed = ["1", "1b", "2", "10", "10a"] + [str(i) for i in range(11, 60)]
    letters = [f"doc{i}" for i in range(1, 60)]
    for label, doc_ids, n_sources, same in (
            ("self-comparison, numeric IDs", numeric, len(numeric), True),
            ("separate corpora, numeric IDs", numeric, 20, False),
            ("separate corpora, prefix-colliding IDs", prefixed, 5, False),
            ("self-comparison, prefix-colliding IDs", prefixed, len(prefixed), True),
            ("separate corpora, non-numeric IDs", letters, 10, False)):
        names = synthetic(doc_ids, n_sources, args.threads, same)
        per_source = len(names) / max(1, len(set(n.split("-")[0] for n in names)))
        check(f"{label} ({len(names)} chunks, {per_source:.1f} per source)",
              bool(names) and contiguous(sort_v(names)))

    if args.source_files:
        workdir = tempfile.mkdtemp(prefix="textpair_chunk_check_")
        try:
            out = os.path.join(workdir, "out")
            count = align(output_path=out, source_files=args.source_files,
                          source_metadata=args.source_metadata,
                          target_files=args.target_files,
                          target_metadata=args.target_metadata, threads=args.threads)
            chunks = os.path.join(out, "result_batches")
            names = sort_v(os.listdir(chunks))
            check(f"corpus: {len(names)} chunks, "
                  f"{len(names) / max(1, len(set(n.split('-')[0] for n in names))):.1f} "
                  "per source, names contiguous", contiguous(names))
            stream = subprocess.run(
                ["bash", "-c", f"cd {chunks!r} && printf '%s\\0' "
                 + " ".join(repr(n) for n in names) + " | xargs -0 lz4cat"],
                capture_output=True, check=True).stdout.decode("utf8")
            docs, runs, previous, broken = set(), 0, None, False
            records = 0
            for line in stream.splitlines():
                if not line.strip():
                    continue
                records += 1
                doc = json.loads(line)["source_doc_id"]
                if doc != previous:
                    broken = broken or doc in docs
                    docs.add(doc)
                    previous = doc
                    runs += 1
            check(f"corpus: {records} records over {len(docs)} source documents in "
                  f"{runs} run(s)", not broken and records == count)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'all checks PASS'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
