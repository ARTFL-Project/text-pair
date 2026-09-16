#!/usr/bin/env python3
"""Checks for the aligner's document ordering (aligner/docorder.py).

    test_document_order.py

Asserts the exact order of a fixture mixing parseable years, unparseable years, a missing
sort field and non-numeric document IDs, that the key is a strict total order over every
pair, and that get_files does not depend on the directory listing's order. Exits non-zero
on any failure.
"""
import itertools
import os
import shutil
import sys
import tempfile

from textpair.sequence_alignment.aligner import docorder

# Parseable and unparseable years, a document with no year at all, and non-numeric IDs.
MIXED = {
    "10": {"year": "1800"},
    "9": {"year": "1800"},
    "2": {"year": "1750"},
    "100": {"year": "1750"},
    "b": {"year": "1750"},
    "a": {"year": "1900"},
    "7": {"year": ""},
    "8": {"year": "n.d."},
    "3": {"year": "1750s"},
    "z": {"title": "no year field"},
}
MIXED_ORDER = ["2", "100", "b", "9", "10", "a", "3", "7", "8", "z"]

# Only two of the four year values parse, so a majority does not: string mode.
MOSTLY_TEXT = {
    "1": {"year": "n.d."},
    "2": {"year": "1800"},
    "3": {"year": "ca. 1750"},
    "x": {"year": "1800"},
    "4": {"title": "no year field"},
}
MOSTLY_TEXT_ORDER = ["4", "2", "x", "3", "1"]

# The document ID order an empty sort_by, or a sort field no document carries, falls to.
DOC_ID_ORDER = ["2", "3", "7", "8", "9", "10", "100", "a", "b", "z"]

failures = []


def check(name, got, want):
    if got == want:
        print(f"PASS {name}")
    else:
        print(f"FAIL {name}\n  got  {got}\n  want {want}")
        failures.append(name)


def check_total_order(name, doc_ids, key):
    """Strict total order: trichotomy over every pair, transitivity over every triple."""
    keys = {doc: key(doc) for doc in doc_ids}
    problems = []
    for first, second in itertools.combinations(doc_ids, 2):
        less, more = keys[first] < keys[second], keys[second] < keys[first]
        if less == more:
            problems.append(f"{first} vs {second}: neither strictly precedes the other")
    for first, second, third in itertools.permutations(doc_ids, 3):
        if keys[first] < keys[second] < keys[third] and not keys[first] < keys[third]:
            problems.append(f"{first} < {second} < {third} but not {first} < {third}")
    check(name, problems[:5], [])


def main():
    check("mode of the mixed fixture",
          docorder.sort_mode(MIXED, "year"), docorder.NUMERIC)
    check("mode of the mostly-text fixture",
          docorder.sort_mode(MOSTLY_TEXT, "year"), docorder.STRING)
    check("mode with an empty sort field", docorder.sort_mode(MIXED, ""), docorder.DOC_ID)
    check("mode with a sort field no document carries",
          docorder.sort_mode(MIXED, "absent"), docorder.DOC_ID)

    numeric = docorder.sort_key(MIXED, "year")
    check("numeric order", sorted(MIXED, key=numeric), MIXED_ORDER)
    check_total_order("numeric order is total", sorted(MIXED), numeric)

    text = docorder.sort_key(MOSTLY_TEXT, "year")
    check("string order", sorted(MOSTLY_TEXT, key=text), MOSTLY_TEXT_ORDER)
    check_total_order("string order is total", sorted(MOSTLY_TEXT), text)

    by_doc_id = docorder.sort_key(MIXED, "")
    check("document ID order", sorted(MIXED, key=by_doc_id), DOC_ID_ORDER)
    check("a sort field no document carries falls back to document ID order",
          sorted(MIXED, key=docorder.sort_key(MIXED, "absent")), DOC_ID_ORDER)
    check_total_order("document ID order is total", sorted(MIXED), by_doc_id)

    # Every value parses, so the key must agree with compareNgrams' (year, int(docID)).
    clean = {doc: fields for doc, fields in MIXED.items()
             if docorder.parse_int(fields.get("year", "")) is not None and doc.isdigit()}
    check("clean corpora keep compareNgrams' (year, int(docID)) order",
          sorted(clean, key=docorder.sort_key(clean, "year")),
          sorted(clean, key=lambda doc: (int(clean[doc]["year"]), int(doc))))

    directory = tempfile.mkdtemp()
    try:
        for doc in MIXED:
            open(os.path.join(directory, doc + ".json"), "w").close()
        os.mkdir(os.path.join(directory, "subdir"))
        found = docorder.get_files(directory, MIXED, "year")
        check("get_files order", [doc for doc, _ in found], MIXED_ORDER)
        check("get_files paths", [os.path.basename(path) for _, path in found],
              [doc + ".json" for doc in MIXED_ORDER])

        listdir = os.listdir
        docorder.os.listdir = lambda path: list(reversed(sorted(listdir(path))))
        try:
            reversed_listing = docorder.get_files(directory, MIXED, "year")
        finally:
            docorder.os.listdir = listdir
        check("get_files ignores the directory listing's order", reversed_listing, found)
    finally:
        shutil.rmtree(directory, ignore_errors=True)

    check("get_files on an empty path", docorder.get_files("", MIXED, "year"), [])
    check("doc_id_of", [docorder.doc_id_of(name) for name in ("1.json", "1.bin", "a.b")],
          ["1", "1", "a.b"])

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'all checks PASS'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
