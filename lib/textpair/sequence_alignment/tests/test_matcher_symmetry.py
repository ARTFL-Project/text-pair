#!/usr/bin/env python3
"""Checks that comparing a pair of documents either way round gives mirrored passages.

    test_matcher_symmetry.py [--source-files DIR --source-metadata FILE] [--threads N]

Which document of a pair is the source is decided by `sort_by` order, so it follows from
a metadata field the matcher has no business depending on. The aligner is run twice over
the same corpus, the second time with the document order reversed, which compares every
pair the other way round. Each record is then reduced to an undirected passage pair --
`(docA, docB, startA, endA, startB, endB)` with `docA` the smaller document ID -- so a
passage found with the roles exchanged reduces to the same key.

Three things are asserted:

  1. the two runs report the same number of records, over the same document pairs, with
     the same duplicate set;
  2. they report exactly the same passages;
  3. a phrase occurring once in one document and n times in another is reported n times
     whichever way round the pair is compared.

The third is the case that motivated the current matcher. Its predecessor reserved the
source range of every passage it emitted and reserved nothing in the target, so it
answered n one way and 1 the other; on frantext that left the two directions agreeing on
52% of passages. `MATCHER_SYMMETRY.md` has the measurements.

The second needs one rule that is not symmetric in the two documents' roles. When two
candidate predecessors sit at (di, dj) and (dj, di), or two chain ends at (i, j) and
(j, i), they are exact mirror images, and the choice between them would otherwise follow
which document is the source. The kernels settle those by document identity instead, so
the same choice is made either way round. Without it ecco_clean differed by 48 passages.

With no arguments it uses the synthetic corpora in fixtures/ plus a repeated-phrase
corpus built here.
"""
import argparse
import csv
import glob
import os
import shutil
import sys
import tempfile
from collections import Counter

import lz4.frame
import orjson

from textpair.sequence_alignment.aligner import documents
from textpair.sequence_alignment.aligner.documents import load_metadata
from textpair.sequence_alignment.aligner.runner import align

FIXTURES = ("no_byte_range", "non_string_meta", "missing_text", "no_metadata")
REPEATS = 7          # occurrences of the shared phrase in the repeating document


def reversed_metadata(source_files, source_metadata, into):
    """The corpus metadata with `year` replaced by a rank that reverses the order.

    The values are distinct, so the reversal is total: no pair keeps its direction
    through a tie in the sort field.
    """
    metadata = load_metadata(source_metadata)
    docs = documents.get_files(source_files, metadata, "year")
    total = len(docs)
    flipped = {}
    for rank, (doc_id, _path) in enumerate(docs):
        fields = dict(metadata.get(doc_id, {}))
        fields["year"] = str(total - rank)
        flipped[doc_id] = fields
    os.makedirs(into, exist_ok=True)
    path = os.path.join(into, "metadata.json")
    with open(path, "wb") as handle:
        handle.write(orjson.dumps(flipped))
    return path, [doc_id for doc_id, _ in docs]


def undirected(tree):
    """Every record as an undirected passage pair, with its document pair."""
    keys = Counter()
    for path in glob.glob(os.path.join(tree, "result_batches", "**", "*.lz4"),
                          recursive=True):
        with lz4.frame.open(path, "rb") as handle:
            for line in handle.read().splitlines():
                if not line.strip():
                    continue
                record = orjson.loads(line)
                one = (record["source_doc_id"], record["source_start_byte"],
                       record["source_end_byte"])
                two = (record["target_doc_id"], record["target_start_byte"],
                       record["target_end_byte"])
                low, high = (one, two) if one[0] < two[0] else (two, one)
                keys[(low[0], high[0], low[1], low[2], high[1], high[2])] += 1
    return keys


def duplicates(tree):
    """The duplicate pairs, as unordered pairs with their overlap.

    A row names its two documents as source and target, so the rows themselves differ
    between the two runs even when the same pairs were flagged. What has to match is the
    set of pairs and the percentage recorded for each.
    """
    path = os.path.join(tree, "duplicate_files.csv")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return sorted((tuple(sorted((row["source_filename"], row["target_filename"]))),
                   row["overlap"]) for row in rows)


def run_both(source_files, source_metadata, workdir, threads, params):
    """Align the corpus in document order and in the reverse of it."""
    forward = os.path.join(workdir, "forward")
    reverse = os.path.join(workdir, "reverse")
    flipped, _docs = reversed_metadata(source_files, source_metadata,
                                       os.path.join(workdir, "flipped"))
    align(source_files, source_metadata, forward, threads=threads, **params)
    align(source_files, flipped, reverse, threads=threads, **params)
    return forward, reverse


def check_corpus(name, source_files, source_metadata, threads, failures, params=None):
    workdir = tempfile.mkdtemp(prefix="textpair_symmetry_")

    def check(label, got, want):
        ok = got == want
        print(f"{'PASS' if ok else 'FAIL'} {name}: {label}", flush=True)
        if not ok:
            failures.append(f"{name}: {label}")
            print(f"     got  {got}\n     want {want}")

    try:
        forward, reverse = run_both(source_files, source_metadata, workdir, threads,
                                    params or {})
        ahead, behind = undirected(forward), undirected(reverse)
        check("the two directions report the same number of records",
              sum(behind.values()), sum(ahead.values()))
        check("over the same document pairs",
              sorted({(k[0], k[1]) for k in behind}),
              sorted({(k[0], k[1]) for k in ahead}))
        check("with the same duplicate set", duplicates(reverse), duplicates(forward))
        total = sum(ahead.values())
        for label, missing in (("forward", ahead - behind), ("reverse", behind - ahead)):
            ok = not missing
            print(f"{'PASS' if ok else 'FAIL'} {name}: {sum(missing.values())} of {total} "
                  f"{label}-run passages are missing from the other direction", flush=True)
            if not ok:
                failures.append(f"{name}: {sum(missing.values())} {label}-run passages "
                                "not found the other way")
                print(f"     examples {sorted(missing)[:3]}")
        return ahead
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def build_repeated_phrase(root):
    """A corpus where one document holds a shared phrase once and another holds it
    REPEATS times, far enough apart that they cannot chain into one passage."""
    phrase = [200001, 200002, 200003, 200004, 200005, 200006]
    for sub in ("ngrams", "metadata", "text"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    metadata = {}
    for doc_id, occurrences in (("1", 1), ("2", REPEATS)):
        index = {}
        # Occurrences sit 5,000 ngrams apart, far beyond matching_window_size, so the
        # matcher cannot join two of them; each has to be reported on its own.
        for occurrence in range(occurrences):
            base = occurrence * 5000
            for offset, key in enumerate(phrase):
                position = base + offset
                index.setdefault(str(key), []).append(
                    [position, position * 10, position * 10 + 8])
        for filler in range(40):                 # keeps the shared share under
            position = 100000 + filler           # duplicate_threshold either way round
            index[str(300000 + filler + int(doc_id) * 1000)] = [
                [position, position * 10, position * 10 + 8]]
        with open(os.path.join(root, "ngrams", f"{doc_id}.json"), "wb") as handle:
            handle.write(orjson.dumps(index))
        with open(os.path.join(root, "text", f"{doc_id}.txt"), "w", encoding="utf8") as h:
            h.write("x" * (REPEATS * 5000 * 10 + 200000))
        metadata[doc_id] = {"filename": f"text/{doc_id}.txt", "year": str(1800 + int(doc_id)),
                            "title": f"doc {doc_id}", "author": "", "start_byte": "0",
                            "end_byte": str(REPEATS * 5000 * 10 + 200000)}
    with open(os.path.join(root, "metadata", "metadata.json"), "wb") as handle:
        handle.write(orjson.dumps(metadata))


def check_repeated_phrase(threads, failures):
    """The case the matcher was changed for: one occurrence against REPEATS of them."""
    workdir = tempfile.mkdtemp(prefix="textpair_symmetry_repeat_")
    name = "repeated phrase"
    try:
        root = os.path.join(workdir, "corpus")
        build_repeated_phrase(root)
        cwd = os.getcwd()
        os.chdir(root)
        try:
            found = check_corpus(name, os.path.join(root, "ngrams"),
                                 os.path.join(root, "metadata", "metadata.json"),
                                 threads, failures)
        finally:
            os.chdir(cwd)
        ok = sum(found.values()) == REPEATS
        print(f"{'PASS' if ok else 'FAIL'} {name}: all {REPEATS} occurrences reported",
              flush=True)
        if not ok:
            failures.append(f"{name}: all {REPEATS} occurrences reported")
            print(f"     got  {sum(found.values())}\n     want {REPEATS}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-files", default="")
    parser.add_argument("--source-metadata", default="")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--flex-gap", default="",
                        help="override flex_gap; both settings are checked by default")
    args = parser.parse_args(argv)
    failures = []
    settings = ([{"flex_gap": args.flex_gap}] if args.flex_gap
                else [{"flex_gap": False}, {"flex_gap": True}])
    if args.source_files:
        if not args.source_metadata:
            parser.error("--source-files needs --source-metadata")
        for params in settings:
            label = f"flex_gap={params['flex_gap']}"
            check_corpus(label, args.source_files, args.source_metadata, args.threads,
                         failures, params)
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        cwd = os.getcwd()
        for fixture in FIXTURES:
            root = os.path.join(here, "fixtures", fixture)
            os.chdir(root)                       # fixture metadata paths are relative
            try:
                check_corpus(fixture, os.path.join(root, "ngrams"),
                             os.path.join(root, "metadata", "metadata.json"),
                             args.threads, failures)
            finally:
                os.chdir(cwd)
        check_repeated_phrase(args.threads, failures)
    print()
    if failures:
        print("FAILED:", *failures, sep="\n  ")
        return 1
    print("all checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
