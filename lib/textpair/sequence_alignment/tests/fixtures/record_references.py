#!/usr/bin/env python3
"""Regenerate the stored reference output for each fixture corpus.

    record_references.py [NAME ...]

`check_reference_output.py --fixtures` checks the aligner against `<fixture>/reference`.
Those trees are output, not a second implementation, so any deliberate change to what
the matcher emits makes them stale and every fixture fails until they are recorded
again.

Run this only when the change is intended, and say in the commit message why the records
moved. Recording references to hide an unexplained difference defeats the point of having
them.
"""
import argparse
import os
import shutil
import sys

from textpair.sequence_alignment.aligner.runner import align

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from check_reference_output import FIXTURE_THREADS  # noqa: E402

FIXTURES = ("no_byte_range", "non_string_meta", "missing_text", "no_metadata")


def record(name, threads=FIXTURE_THREADS):
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(here, name)
    reference = os.path.join(root, "reference")
    shutil.rmtree(reference, ignore_errors=True)
    cwd = os.getcwd()
    os.chdir(root)                       # fixture metadata paths are relative
    try:
        count = align("ngrams", "metadata/metadata.json", reference, threads=threads)
    finally:
        os.chdir(cwd)
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", default=None,
                        help="fixtures to record; all of them by default")
    # Not configurable: a reference recorded at another thread count could never
    # match, because chunk file names carry the per-thread target split.
    args = parser.parse_args(argv)
    names = args.names or list(FIXTURES)
    unknown = [name for name in names if name not in FIXTURES]
    if unknown:
        parser.error(f"unknown fixture(s): {', '.join(unknown)}")
    for name in names:
        count = record(name)
        print(f"recorded {name}: {count} alignment(s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
