#!/usr/bin/env python3
"""Characterise the difference between two aligner runs over the same corpus.

    check_direction_flips.py BEFORE_DIR AFTER_DIR --source-metadata FILE
        [--target-metadata FILE] [--sort-by FIELD] [--workers N] [--max-differences N]

Changing the document order changes which document of a pair is the source and which is
the target. Every record one run has and the other has not must therefore

  1. involve a document the two orders can place differently, and
  2. belong to a document pair the two runs compare in opposite directions.

Both are required to pass. The matcher is symmetric, so reversing a pair now does
relabel its alignments and nothing more, apart from the tie `test_matcher_symmetry.py`
documents, which can move a passage's start by a few ngrams. How many of the differing
records have an exact mirror in the other run is therefore reported and expected to be
nearly all of them, but is still not required. Exits non-zero, with an example, on any
failure.

Which documents count as movable follows the sort mode: in numeric mode those whose sort
value does not parse as an integer, in string mode those sharing a value with another
document, in document-ID mode those whose ID is not an integer.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import lz4.frame
import orjson

from textpair.sequence_alignment.aligner import docorder
from textpair.sequence_alignment.aligner.gotext import load_metadata

CANON = orjson.OPT_SORT_KEYS


def digest(record):
    return hashlib.sha1(orjson.dumps(record, option=CANON)).digest()


def flip(record):
    """The same alignment with source and target exchanged."""
    out = {}
    for key, value in record.items():
        if key.startswith("source_"):
            out["target_" + key[len("source_"):]] = value
        elif key.startswith("target_"):
            out["source_" + key[len("target_"):]] = value
        else:
            out[key] = value
    return out


def chunks(tree):
    return sorted(glob.glob(os.path.join(tree, "result_batches", "**", "*.lz4"),
                            recursive=True))


def chunk_digests(path):
    with lz4.frame.open(path, "rb") as handle:
        raw = handle.read()
    return Counter(digest(orjson.loads(line)) for line in raw.splitlines() if line.strip())


def chunk_records(job):
    path, wanted = job
    with lz4.frame.open(path, "rb") as handle:
        raw = handle.read()
    out = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        record = orjson.loads(line)
        key = digest(record)
        if key in wanted:
            out.append((key, record))
    return out


def scan(tree, workers):
    counts = Counter()
    paths = chunks(tree)
    if not paths:
        raise SystemExit(f"no result chunks under {tree}")
    if workers > 1 and len(paths) > 1:
        with ProcessPoolExecutor(workers) as pool:
            for part in pool.map(chunk_digests, paths, chunksize=16):
                counts.update(part)
    else:
        for path in paths:
            counts.update(chunk_digests(path))
    return counts


def collect(tree, wanted, workers):
    paths = chunks(tree)
    jobs = [(path, wanted) for path in paths]
    out = []
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(workers) as pool:
            for part in pool.map(chunk_records, jobs, chunksize=16):
                out.extend(part)
    else:
        for job in jobs:
            out.extend(chunk_records(job))
    return out


def trim(records, counts):
    """Keep only as many copies of each digest as the multiset difference calls for."""
    seen = Counter()
    out = []
    for key, record in records:
        if seen[key] < counts[key]:
            seen[key] += 1
            out.append((key, record))
    return out


def movable_documents(metadata, sort_field):
    """The mode, and a predicate for the documents the two orders can place differently.
    A document with no metadata entry has no sort value, so it is always movable."""
    mode = docorder.sort_mode(metadata, sort_field)
    if mode == docorder.DOC_ID:
        movable = {doc for doc in metadata if docorder.parse_int(doc) is None}
    elif mode == docorder.STRING:
        values = Counter(fields.get(sort_field, "") for fields in metadata.values())
        movable = {doc for doc, fields in metadata.items()
                   if values[fields.get(sort_field, "")] > 1}
    else:
        movable = {doc for doc, fields in metadata.items()
                   if docorder.parse_int(fields.get(sort_field, "")) is None}
    return mode, movable, lambda doc: doc in movable or doc not in metadata


def read_text(path):
    try:
        with open(path, encoding="utf8") as handle:
            return handle.read()
    except FileNotFoundError:
        return None


def characterise(before, after, source_movable, target_movable, workers, limit):
    result = {"before": before, "after": after}
    before_counts, after_counts = scan(before, workers), scan(after, workers)
    result["n_records_before"] = sum(before_counts.values())
    result["n_records_after"] = sum(after_counts.values())
    only_before = before_counts - after_counts
    only_after = after_counts - before_counts
    result["n_records_only_before"] = sum(only_before.values())
    result["n_records_only_after"] = sum(only_after.values())
    result["records_identical"] = not only_before and not only_after
    if result["records_identical"]:
        result["PASS"] = True
        return result
    total = sum(only_before.values()) + sum(only_after.values())
    if total > limit:
        result["PASS"] = False
        result["error"] = (f"{total} differing records exceeds --max-differences {limit}; "
                           "this is far more than a direction flip can explain")
        return result

    before_only = trim(collect(before, set(only_before), workers), only_before)
    after_only = trim(collect(after, set(only_after), workers), only_after)
    mirrored = {"before": 0, "after": 0}
    orientations = {"before": {}, "after": {}}
    not_movable = []
    for run, records, other in (("before", before_only, only_after),
                                ("after", after_only, only_before)):
        for _key, record in records:
            source, target = record.get("source_doc_id"), record.get("target_doc_id")
            if not source_movable(source) and not target_movable(target):
                not_movable.append({"run": run, "source_doc_id": source,
                                    "target_doc_id": target})
            orientations[run].setdefault(frozenset((source, target)), set()).add(
                (source, target))
            mirrored[run] += digest(flip(record)) in other

    pairs = set(orientations["before"]) | set(orientations["after"])
    same_direction, ambiguous = [], []
    for pair in pairs:
        seen_before = orientations["before"].get(pair, set())
        seen_after = orientations["after"].get(pair, set())
        if len(seen_before) > 1 or len(seen_after) > 1:
            ambiguous.append(sorted(pair))
        if seen_before & seen_after:
            same_direction.append(sorted(seen_before & seen_after)[0])

    result["n_document_pairs_affected"] = len(pairs)
    result["document_pairs_affected"] = sorted(sorted(pair) for pair in pairs)[:20]
    result["movable_documents_involved"] = sorted(
        {doc for pair in pairs for doc in pair
         if source_movable(doc) or target_movable(doc)})[:50]
    result["n_records_mirrored_in_the_other_run"] = [mirrored["before"], mirrored["after"]]
    result["every_difference_involves_a_movable_document"] = not not_movable
    result["every_difference_is_a_direction_flip"] = not same_direction and not ambiguous
    if not_movable:
        result["differences_involving_no_movable_document"] = not_movable[:10]
    if same_direction:
        result["pairs_differing_without_a_direction_flip"] = same_direction[:10]
    if ambiguous:
        result["pairs_differing_in_both_directions"] = ambiguous[:10]
    result["PASS"] = (result["every_difference_involves_a_movable_document"]
                      and result["every_difference_is_a_direction_flip"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--source-metadata", required=True)
    parser.add_argument("--target-metadata", default="",
                        help="defaults to the source metadata, as a single-corpus run does")
    parser.add_argument("--sort-by", default="year")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-differences", type=int, default=500000)
    args = parser.parse_args(argv)

    source_meta = load_metadata(args.source_metadata)
    target_meta = load_metadata(args.target_metadata) if args.target_metadata else source_meta
    source_mode, source_set, source_movable = movable_documents(source_meta, args.sort_by)
    target_mode, target_set, target_movable = movable_documents(target_meta, args.sort_by)
    result = {"sort_by": args.sort_by,
              "source_sort_mode": source_mode,
              "n_source_documents": len(source_meta),
              "n_source_movable": len(source_set),
              "target_sort_mode": target_mode,
              "n_target_movable": len(target_set)}
    result.update(characterise(args.before, args.after, source_movable, target_movable,
                               args.workers, args.max_differences))
    for name in ("count.txt", "duplicate_files.csv"):
        before_text = read_text(os.path.join(args.before, name))
        after_text = read_text(os.path.join(args.after, name))
        if name == "count.txt":
            result["count_txt"] = [before_text, after_text]
        else:
            result["n_duplicates"] = [len((before_text or "").splitlines()) - 1,
                                      len((after_text or "").splitlines()) - 1]
            result["duplicates_identical_multiset"] = (
                sorted((before_text or "").splitlines()) == sorted((after_text or "").splitlines()))
    print(json.dumps(result, indent=1, ensure_ascii=False))
    return 0 if result["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
