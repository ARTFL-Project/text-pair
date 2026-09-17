#!/usr/bin/env python3
"""Checks that the aligner's input formats and batching do not change its output.

    test_aligner_paths.py [--source-files DIR --source-metadata FILE
                           [--target-files DIR --target-metadata FILE]] [--threads N]

Three things the aligner can vary without being allowed to change what it finds:

  - the ngram index format. Generation writes the binary columnar index; the loader also
    parses the older JSON. Both must give the same alignments, in self-comparison and
    with a separate target corpus.
  - source_batch / target_batch. Batching only bounds how much of the corpus is resident,
    so the records must be the same multiset however the corpus is sliced.
  - a batch count larger than the corpus, which must slice to one batch per document
    rather than fail.

Needs a JSON `ngrams/` directory to convert; with no arguments it uses the fixtures.
"""
import argparse
import collections
import glob
import hashlib
import os
import shutil
import sys
import tempfile

from textpair.sequence_alignment import ngram_binary
from textpair.sequence_alignment.aligner import align

HERE = os.path.dirname(os.path.abspath(__file__))


def tree(path):
    """Every output file's digest, less the config, which records the output path."""
    digests = {}
    for root, _dirs, names in os.walk(path):
        for name in sorted(names):
            if name == "alignment_config.ini":
                continue
            full = os.path.join(root, name)
            with open(full, "rb") as handle:
                digests[os.path.relpath(full, path)] = hashlib.md5(handle.read()).hexdigest()
    return digests


def records(path):
    """The alignment records as a multiset, independent of how chunks were named."""
    import lz4.frame
    counter = collections.Counter()
    for chunk in glob.glob(os.path.join(path, "**", "*.lz4"), recursive=True):
        with lz4.frame.open(chunk) as handle:
            for line in handle:
                counter[line] += 1
    return counter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-files", default="")
    parser.add_argument("--source-metadata", default="")
    parser.add_argument("--target-files", default="")
    parser.add_argument("--target-metadata", default="")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if not args.source_files:
        root = os.path.join(HERE, "fixtures", "no_byte_range")
        args.source_files = os.path.join(root, "ngrams")
        args.source_metadata = os.path.join(root, "metadata", "metadata.json")
    failures = []

    def check(label, got, want):
        ok = got == want
        print(f"{'PASS' if ok else 'FAIL'} {label}")
        if not ok:
            failures.append(label)
            if isinstance(got, dict) and isinstance(want, dict):
                for key in sorted(set(got) | set(want)):
                    if got.get(key) != want.get(key):
                        print(f"     {key}: {got.get(key)} vs {want.get(key)}")
            else:
                print(f"     got {got}\n     want {want}")

    work = tempfile.mkdtemp(prefix="textpair_paths_check_")
    try:
        # the same corpora, converted to the binary index
        binary = {}
        for side, directory in (("source", args.source_files), ("target", args.target_files)):
            if not directory:
                continue
            binary[side] = os.path.join(work, f"{side}_binary")
            ngram_binary.convert_directory(directory, binary[side], args.threads)

        def run(name, **over):
            out = os.path.join(work, name)
            shutil.rmtree(out, ignore_errors=True)
            params = dict(source_files=args.source_files,
                          source_metadata=args.source_metadata,
                          target_files=args.target_files,
                          target_metadata=args.target_metadata,
                          threads=args.threads, output_path=out)
            params.update(over)
            count = align(**params)
            return out, count

        json_self, n_self = run("json_self", target_files="", target_metadata="")
        bin_self, n_bin = run("bin_self", source_files=binary["source"],
                              target_files="", target_metadata="")
        check(f"self-comparison: binary index matches JSON ({n_self} alignments)",
              tree(bin_self), tree(json_self))
        check("self-comparison: same count", n_bin, n_self)

        if args.target_files:
            json_two, n_two = run("json_two")
            bin_two, n_bin_two = run("bin_two", source_files=binary["source"],
                                     target_files=binary["target"])
            check(f"two corpora: binary index matches JSON ({n_two} alignments)",
                  tree(bin_two), tree(json_two))
            check("two corpora: same count", n_bin_two, n_two)

        base_out, base_count = json_self, n_self
        base = records(base_out)
        for label, over in (("source_batch=3", {"source_batch": 3}),
                            ("target_batch=3", {"target_batch": 3}),
                            ("both batched", {"source_batch": 2, "target_batch": 2}),
                            ("source_batch beyond the corpus", {"source_batch": 10_000}),
                            ("target_batch beyond the corpus", {"target_batch": 10_000})):
            out, count = run(label.replace(" ", "_").replace("=", ""),
                             target_files="", target_metadata="", **over)
            check(f"{label}: same records", records(out), base)
            check(f"{label}: same count", count, base_count)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'all checks PASS'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
