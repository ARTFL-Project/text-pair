#!/usr/bin/env python3
"""Drift guard for tracing's mirrors of matching.match_passage and matching.align_pair.

    check_tracing.py [--source-files DIR --source-metadata FILE] [--threads N]
        [--max-matches N]

Run this when you change the matcher: `tracing._walk` is a second implementation of it,
and this is what tells you the two have not drifted. It is not in `run_tests.py`, because
it checks the --debug trace rather than the alignments, and a --debug run over a whole
corpus is slow -- write_traces re-derives every pair, which --max-matches cannot bound,
since a trace has to describe the whole pair. Point it at a corpus deliberately.

The trace is produced after matching by walking each pair's matches a second time, in
`tracing._walk` and `tracing.pair_rows`, so that nothing is on the matching path. They
must behave exactly like `matching.match_passage` and `matching.align_pair`. This feeds
each pair the same matches, for every pair of a corpus, and asserts they emit the same
alignments, before merging and after. It also aligns the corpus with and
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

from textpair.sequence_alignment.aligner import (align, documents, matching,
                                                 ngram_loader, tracing)
from textpair.sequence_alignment.aligner.runner import _normalize

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = ("no_byte_range", "non_string_meta", "missing_text", "no_metadata")


def compare_kernels(source_files, source_metadata, params, threads, max_matches):
    """Every pair's alignments from both implementations.

    Returns (pairs, mismatches, truncated). `max_matches` bounds the matches taken from
    any one pair; both implementations get the same list, so the comparison is unaffected
    by it, and every pair is still walked. Without a bound this is the slowest check by
    far -- the walk is pure Python by design, and classical_chinese holds 20.5 million
    matches across its 1,891 pairs, 25 minutes for the two flex_gap settings.
    """
    from textpair.sequence_alignment.aligner.documents import load_metadata
    docs = documents.get_files(source_files, load_metadata(source_metadata),
                              params["sort_by"])
    corpus = tracing.Corpus(*ngram_loader.load_corpus([path for _, path in docs], threads))
    names = [doc for doc, _ in docs]
    pairs = mismatches = truncated = 0
    for source, target in tracing._pairs(names, len(names), np.empty(0, np.int32), None):
        match, n, stopped_early = corpus.matches(source, target, max_matches)
        if not n:
            continue
        pairs += 1
        truncated += stopped_early
        # Both ways of settling a mirror tie, alternating by pair.
        source_first = (source + target) % 2 == 0
        chains, _blocks, _hidden, _longest = tracing._walk(
            match, n, params, params["debug_minimum_ngrams"], source_first)
        rows, _blocks, _hidden, _merged, _coalesced = tracing.pair_rows(
            match, n, params, params["debug_minimum_ngrams"], source_first)
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
        n_blocks = int(np.unique(np.asarray(s_idx)).shape[0])

        def buffers():
            return (np.empty(n, np.int32), np.empty(n, np.int32), np.empty(n, np.uint8),
                    np.empty(n, np.int32), np.empty(n + 1, np.int32),
                    np.empty(n, np.int32),
                    np.empty((64, 4), np.int32), np.empty((64, matching.NCOL), np.int32))
        common = (params["matching_window_size"], params["max_gap"], params["flex_gap"],
                  params["minimum_matching_ngrams"],
                  params["minimum_matching_ngrams_in_window"])
        out, cnt, _spans, _longest = matching.match_passage(
            pair, pos, n, n_blocks, start_bytes, end_bytes, *common, source_first,
            *buffers())
        final, final_cnt, _spans, _out = matching.align_pair(
            pair, pos, n, n_blocks, start_bytes, end_bytes, *common,
            params["merge_passages_on_byte_distance"],
            params["merge_passages_on_ngram_distance"],
            params["passage_distance_multiplier"], source_first, *buffers())
        def as_tuples(array, count):
            return [tuple(int(v) for v in row) for row in array[:count]]
        if as_tuples(out, cnt) != [tuple(r) for r in chains] or \
                as_tuples(final, final_cnt) != [tuple(r) for r in rows]:
            mismatches += 1
    return pairs, mismatches, truncated


def check_corpus(name, source_files, source_metadata, threads, failures, overrides=None,
                 max_matches=0):
    params = _normalize(overrides or {})
    workdir = tempfile.mkdtemp(prefix="textpair_debug_check_")

    def check(label, got, want):
        ok = got == want
        print(f"{'PASS' if ok else 'FAIL'} {name}: {label}")
        if not ok:
            failures.append(f"{name}: {label}")
            print(f"     got  {got}\n     want {want}")

    try:
        pairs, mismatches, truncated = compare_kernels(source_files, source_metadata,
                                                      params, threads, max_matches)
        capped = (f", {truncated} of them capped at {max_matches:,} matches"
                  if truncated else "")
        check(f"_walk and pair_rows match the kernels on all {pairs} pair(s){capped}",
              mismatches, 0)

        common = dict(source_files=source_files, source_metadata=source_metadata,
                      threads=threads, **(overrides or {}))
        plain, traced = os.path.join(workdir, "plain"), os.path.join(workdir, "traced")
        count = align(output_path=plain, **common)
        check("debug does not change the alignment count",
              align(output_path=traced, debug=True, **common), count)

        matches = merged_away = coalesced = 0
        debug_dir = os.path.join(traced, "debug_output")
        for trace in sorted(os.listdir(debug_dir)) if os.path.isdir(debug_dir) else []:
            with open(os.path.join(debug_dir, trace), encoding="utf8") as handle:
                text = handle.read()
            matches += len(re.findall(r"## MATCH ##", text))
            merged_away += sum(int(n) for n in
                               re.findall(r"^(\d+) passage\(s\) merged", text, re.M))
            coalesced += sum(int(n) for n in
                             re.findall(r"^(\d+) passage\(s\) coalesced", text, re.M))
        check("the trace accounts for every alignment",
              matches - coalesced - merged_away, count)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-files", default="")
    parser.add_argument("--source-metadata", default="")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-matches", type=int, default=1000, metavar="N",
                        help="matches taken from any one pair; 0 for all of them. "
                             "Both implementations get the same list either way")
    args = parser.parse_args(argv)
    failures = []
    # Both settings of flex_gap: it changes how far the matcher links and the allowance
    # the walk down each chain enforces, so the mirror has to hold for each. The shipped
    # sa_config.ini sets it true, the aligner's own DEFAULTS leave it false.
    settings = [{"flex_gap": False}, {"flex_gap": True}]
    if args.source_files:
        base = os.path.basename(args.source_files.rstrip("/")) or "corpus"
        for overrides in settings:
            check_corpus(f"{base} flex_gap={overrides['flex_gap']}", args.source_files,
                         args.source_metadata, args.threads, failures, overrides,
                         args.max_matches)
    else:
        for fixture in FIXTURES:
            root = os.path.join(HERE, "fixtures", fixture)
            for overrides in settings:
                check_corpus(f"{fixture} flex_gap={overrides['flex_gap']}",
                             os.path.join(root, "ngrams"),
                             os.path.join(root, "metadata", "metadata.json"),
                             args.threads, failures, overrides)
    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'all checks PASS'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
