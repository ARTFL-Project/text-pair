#!/usr/bin/env python3
"""Run the aligner's checks and report one line each.

    run_tests.py [--threads N] [--only NAME]... [--fresh-cache]
    run_tests.py --source-files DIR --source-metadata FILE [...]   # also at corpus scale

With no corpus, every check runs against the synthetic fixtures in fixtures/ and the
corpora the checks build for themselves. That is the fast pass and it needs nothing
installed beyond the package.

Given a corpus, the checks that can take one are run against it as well. That pass is
much slower and is the one worth doing before pushing: several of the properties here --
the trace mirror, symmetry under a reversed document order, the ordering invariants
`align_source` relies on -- only fail on corpora dense enough to produce the awkward
cases, which the fixtures are not. The validation corpora are under
/disk1/shared/text-pair-validation/corpora; see README.md.

Exits non-zero if any check fails, and prints each one's output on failure.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# (name, script, fixture-pass arguments, whether a corpus is passed to it)
#
# Two are deliberately fixture-only even when a corpus is given:
#
#   document_order exercises sort-key logic on metadata it builds itself, and it takes
#   --source-files without a metadata argument, so the corpus pair does not fit it.
#
#   aligner_paths tests batching arithmetic, which corpus size does not exercise, and one
#   of its cases asks for more batches than the corpus has documents. Batch counts clamp
#   to the document count, so that case becomes one batch per document and therefore
#   O(documents^2) combinations, each rebuilding the inverted index over the whole corpus:
#   62 documents is already 20 minutes, and frantext's 3,596 would be millions of
#   combinations. Its other claim -- that the binary and JSON indexes give the same
#   alignments -- is checked against real corpora by ngram_binary.
CHECKS = (
    ("ngram_index", "test_ngram_index.py", (), False),
    ("ngram_order", "test_ngram_order.py", (), False),
    ("document_order", "test_document_order.py", (), False),
    ("match_order", "test_match_order.py", (), True),
    ("chunk_order", "test_chunk_order.py", (), True),
    ("aligner_paths", "test_aligner_paths.py", (), False),
    ("banality_parallel", "test_banality_parallel.py", (), False),
    ("writer_descriptors", "test_writer_descriptors.py", (), False),
    ("passage_families", "test_passage_families.py", (), False),
    ("matcher_symmetry", "test_matcher_symmetry.py", (), True),
    ("reference_output", "check_reference_output.py", ("--fixtures",), False),
    # Needs a corpus: there is no fixture-scale binary index to compare against.
    ("ngram_binary", "check_ngram_binary.py", None, True),
)

# Two more are not here at all.
#
#   check_direction_flips.py characterises two output trees that already exist, so it has
#   nothing to run against until someone produces them.
#
#   check_tracing.py checks the --debug trace, not the alignments: that `tracing._walk`
#   still mirrors the matcher, and that tracing changes neither the records nor the count.
#   It is a tool for whoever changes the matcher and has to re-sync the mirror, which is
#   the one moment it earns its keep. As a standing check it was the slowest thing here --
#   a --debug run re-derives every pair, which the walk's match cap cannot bound because a
#   trace has to describe the whole pair -- and it guards a developer diagnostic rather
#   than the output. What the matcher itself produces is covered by matcher_symmetry,
#   reference_output, match_order, chunk_order and aligner_paths.


def run(name, script, argv, threads, env):
    command = [sys.executable, os.path.join(HERE, script), *argv, "--threads", str(threads)]
    started = time.perf_counter()
    done = subprocess.run(command, capture_output=True, text=True, env=env)
    return name, done.returncode == 0, time.perf_counter() - started, done


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-files", default="", help="ngram directory of a corpus")
    parser.add_argument("--source-metadata", default="")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--only", action="append", default=[], metavar="NAME",
                        help="run just these checks; repeatable")
    parser.add_argument("--fresh-cache", action="store_true",
                        help="compile into an empty numba cache directory, and discard it")
    args = parser.parse_args(argv)
    if bool(args.source_files) != bool(args.source_metadata):
        parser.error("--source-files and --source-metadata go together")
    unknown = set(args.only) - {name for name, _, _, _ in CHECKS}
    if unknown:
        parser.error(f"unknown check(s): {', '.join(sorted(unknown))}; "
                     f"choose from {', '.join(name for name, _, _, _ in CHECKS)}")

    env = dict(os.environ)
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    cache = None
    if args.fresh_cache:
        # A cache holding kernels compiled from different source has produced convincing
        # wrong answers more than once; this is the way to rule it out.
        cache = tempfile.mkdtemp(prefix="textpair_numba_")
        env["NUMBA_CACHE_DIR"] = cache
        env["TEXTPAIR_NUMBA_CACHE_DIR"] = cache

    corpus = (("--source-files", args.source_files,
               "--source-metadata", args.source_metadata) if args.source_files else ())
    plan = []
    for name, script, fixture_argv, takes_corpus in CHECKS:
        if args.only and name not in args.only:
            continue
        if corpus and takes_corpus:
            plan.append((name, script, corpus))
        elif fixture_argv is not None:
            plan.append((name, script, fixture_argv))
        else:
            print(f"{'SKIP':<6}{name:<20} needs a corpus", flush=True)

    scope = "corpus" if corpus else "fixtures"
    print(f"{len(plan)} check(s), {scope}, {args.threads} threads"
          + (f", fresh cache {cache}" if cache else ""), flush=True)
    failures = []
    try:
        for name, script, argv_for in plan:
            label, ok, seconds, done = run(name, script, argv_for, args.threads, env)
            print(f"{'PASS' if ok else 'FAIL':<6}{label:<20}{seconds:>8.1f}s", flush=True)
            if not ok:
                failures.append(label)
                print(done.stdout[-4000:] or "(no output)")
                if done.stderr.strip():
                    print(done.stderr[-2000:], file=sys.stderr)
    finally:
        if cache:
            shutil.rmtree(cache, ignore_errors=True)
    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print(f"all {len(plan)} check(s) passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
