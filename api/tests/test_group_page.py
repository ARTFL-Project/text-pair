#!/usr/bin/env python3
"""A passage group's page must list each reuse once, compared with the right alignment.

    test_group_page.py

Each case is a few alignments around a group anchored in document A at bytes 100-400:
spans overlapping in one document making one place, the anchor's own document left out,
an alignment with the passage itself preferred, then one with another reusing document,
and last one with another passage of the anchor's document.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_pair import occurrences  # noqa: E402

FAILURES = []


def check(label, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        FAILURES.append(label)


GROUP = {"source_doc_id": "A", "source_start_byte": 100, "source_end_byte": 400}


def align(rowid, source, s0, s1, target, t0, t1):
    return {"rowid": rowid,
            "source_doc_id": source, "source_filename": f"{source}.xml", "source_start_byte": s0, "source_end_byte": s1,
            "target_doc_id": target, "target_filename": f"{target}.xml", "target_start_byte": t0, "target_end_byte": t1}


def places(rows):
    return {(p["target_doc_id"], p["target_start_byte"]): (p["rowid"], p["source_doc_id"]) for p in occurrences(rows, GROUP)}


def main():
    found = places([align(1, "A", 100, 400, "B", 0, 300), align(2, "B", 0, 250, "C", 0, 250),
                    align(3, "B", 10, 290, "D", 0, 280)])
    check("each document is one place, the anchor's left out", sorted(found) == [("B", 0), ("C", 0), ("D", 0)])
    check("... compared with the passage where it is aligned with it", found[("B", 0)] == (1, "A"))
    check("... and with another reusing document where it is not", found[("C", 0)] == (2, "B"))

    found = places([align(1, "A", 100, 400, "B", 0, 300), align(2, "A", 120, 300, "B", 50, 230)])
    check("the alignment overlapping the passage most stands for the place", found[("B", 0)] == (1, "A"))

    # E is aligned with A only at another passage, 600-700, and with B, a reuse of this one
    found = places([align(1, "A", 100, 400, "B", 0, 300), align(2, "A", 600, 700, "E", 0, 100),
                    align(3, "B", 200, 300, "E", 0, 100)])
    check("an alignment with another passage of the anchor's document comes after one with a reuse",
          found[("E", 0)] == (3, "B"))
    found = places([align(2, "A", 600, 700, "E", 0, 100)])
    check("... and still stands for a place that has no other", found.get(("E", 0)) == (2, "A"))
    found = places([align(2, "A", 400, 500, "E", 0, 100), align(4, "A", 900, 950, "E", 20, 80)])
    check("... the nearest of them first, a span ending where the passage starts counting as apart",
          found[("E", 0)] == (2, "A"))

    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
