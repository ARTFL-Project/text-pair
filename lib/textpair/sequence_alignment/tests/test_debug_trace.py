#!/usr/bin/env python3
"""Drift guard for debug._walk against kernels.match_passage.

    test_debug_trace.py [--source-files DIR --source-metadata FILE] [--threads N]

The trace is produced after matching by walking each pair's matches a second time, in
`debug._walk`, so that nothing is on the matching path. That walk must behave exactly
like `kernels.match_passage`. This feeds both the same matches, for every pair of a
corpus, and asserts they emit the same alignments. It also aligns the corpus with and
without `debug` and checks the counts and the trace's accounting agree.

With no arguments it runs the fixture corpora, which is fast but thin. Point it at a real
corpus for a meaningful check.
"""
import argparse
import os
import re
import shutil
import sys
import tempfile

import numpy as np

from textpair.sequence_alignment.aligner import align, debug, docorder, kernels, loader
from textpair.sequence_alignment.aligner.runner import DEFAULTS, _normalize

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = ("no_byte_range", "non_string_meta", "missing_text", "no_metadata")


def compare_kernels(source_files, source_metadata, params, threads):
    """Every pair's alignments from both implementations. Returns (pairs, mismatches)."""
    from textpair.sequence_alignment.aligner.gotext import load_metadata
    docs = docorder.get_files(source_files, load_metadata(source_metadata),
                              params["sort_by"])
    corpus = debug.Corpus(*loader.load_corpus([path for _, path in docs], threads))
    names = [doc for doc, _ in docs]
    pairs = mismatches = 0
    for source, target in debug._pairs(names, len(names), np.empty(0, np.int32), None):
        match, n = corpus.matches(source, target)
        if not n:
            continue
        pairs += 1
        rows, _blocks, _hidden = debug._walk(match, n, params,
                                             params["debug_minimum_ngrams"])
        # The kernel takes the packed layout: indices together in one int64 and byte
        # offsets reached through positions. Lay the pair's offsets out so position k is
        # its source and position n + k its target.
        s_idx, s_sb, s_eb, t_idx, t_sb, t_eb, _keys = match
        pair = ((np.asarray(s_idx, np.int64) << 32)
                | np.asarray(t_idx, np.int64))
        pos = ((np.arange(n, dtype=np.int64) << 32)
               | np.arange(n, 2 * n, dtype=np.int64))
        start_bytes = np.concatenate([np.asarray(s_sb, np.int32), np.asarray(t_sb, np.int32)])
        end_bytes = np.concatenate([np.asarray(s_eb, np.int32), np.asarray(t_eb, np.int32)])
        out, cnt = kernels.match_passage(
            pair, pos, n, start_bytes, end_bytes,
            params["matching_window_size"], params["max_gap"], params["flex_gap"],
            params["minimum_matching_ngrams"],
            params["minimum_matching_ngrams_in_window"])
        if [tuple(int(v) for v in row) for row in out[:cnt]] != \
                [tuple(int(v) for v in row) for row in rows]:
            mismatches += 1
    return pairs, mismatches


def check_corpus(name, source_files, source_metadata, threads, failures):
    params = _normalize({})
    workdir = tempfile.mkdtemp(prefix="textpair_debug_check_")

    def check(label, got, want):
        ok = got == want
        print(f"{'PASS' if ok else 'FAIL'} {name}: {label}")
        if not ok:
            failures.append(f"{name}: {label}")
            print(f"     got  {got}\n     want {want}")

    try:
        pairs, mismatches = compare_kernels(source_files, source_metadata, params, threads)
        check(f"_walk matches the kernel on all {pairs} pair(s)", mismatches, 0)

        common = dict(source_files=source_files, source_metadata=source_metadata,
                      threads=threads)
        plain, traced = os.path.join(workdir, "plain"), os.path.join(workdir, "traced")
        count = align(output_path=plain, **common)
        check("debug does not change the alignment count",
              align(output_path=traced, debug=True, **common), count)

        matches = merged_away = 0
        debug_dir = os.path.join(traced, "debug_output")
        for trace in sorted(os.listdir(debug_dir)) if os.path.isdir(debug_dir) else []:
            with open(os.path.join(debug_dir, trace), encoding="utf8") as handle:
                text = handle.read()
            matches += len(re.findall(r"## MATCH ##", text))
            merged_away += sum(int(n) for n in
                               re.findall(r"^(\d+) passage\(s\) merged", text, re.M))
        check("the trace accounts for every alignment", matches - merged_away, count)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-files", default="")
    parser.add_argument("--source-metadata", default="")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    failures = []
    if args.source_files:
        check_corpus(os.path.basename(args.source_files.rstrip("/")) or "corpus",
                     args.source_files, args.source_metadata, args.threads, failures)
    else:
        for fixture in FIXTURES:
            root = os.path.join(HERE, "fixtures", fixture)
            check_corpus(fixture, os.path.join(root, "ngrams"),
                         os.path.join(root, "metadata", "metadata.json"), args.threads,
                         failures)
    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'all checks PASS'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
