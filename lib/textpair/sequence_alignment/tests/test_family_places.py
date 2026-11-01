#!/usr/bin/env python3
"""A family's passage must be projected onto each place that reuses it.

    test_family_places.py

One family, its passage in document A, and the places its alignments give in other
documents: one running on past the passage is cut to it, one holding only part of it is
grown to the whole of what it shares, overlapping spans in a document are one place, the
anchor's own document is no place, and a place that shares nothing with the passage in its
order keeps its span, shared 0 -- even where another place of its document has the passage.
"""
import os
import shutil
import sys
import tempfile

import lz4.frame
import orjson

from textpair.sequence_alignment.family_places import project_places

FAILURES = []
PASSAGE = "il y a donc une profession de foi purement civile dont il appartient au souverain de fixer les articles"
TEXTS = {
    "A": f"Du contrat social. Livre quatre. {PASSAGE}, non pas précisément comme dogmes de religion. Fin du chapitre.",
    "B": f"Blanc écrit ceci : {PASSAGE.replace('donc ', '')}, et il ajoute tout un discours sur la Convention, "
         "les clubs, les sections et la commune de Paris, qui ne doit rien au livre.",
    "C": f"Taine le cite : {PASSAGE.replace('il y a donc ', '')}. Puis il passe aux jacobins et à leur catéchisme.",
    "E": f"Sorel reprend : {PASSAGE}. Le reste du chapitre parle de diplomatie.",
    "F": "Un tout autre livre, sur la marine et le commerce des Indes, qui cite la commune de Paris et les clubs.",
}


def check(label, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        FAILURES.append(label)


def span(doc, words):
    data = TEXTS[doc].encode()
    start = data.index(words.encode())
    return start, start + len(words.encode())


def main():
    root = tempfile.mkdtemp(prefix="places_")
    try:
        path = {}
        for doc, text in TEXTS.items():
            path[doc] = os.path.join(root, f"{doc}.txt")
            with open(path[doc], "w", encoding="utf8") as handle:
                handle.write(text)

        def record(source, source_words, target, target_words):
            s0, s1 = span(source, source_words)
            t0, t1 = span(target, target_words)
            return {"group_id": [7], "source_filename": path[source], "source_doc_id": source, "source_start_byte": s0,
                    "source_end_byte": s1, "target_filename": path[target], "target_doc_id": target,
                    "target_start_byte": t0, "target_end_byte": t1}

        records = [
            record("A", "une profession de foi purement civile", "B", "une profession de foi purement civile dont il "
                   "appartient au souverain de fixer les articles, et il ajoute tout un discours sur la Convention"),
            record("B", "profession de foi purement civile dont", "C", "profession de foi purement civile dont"),
            record("A", "une profession de foi purement", "E", "une profession de foi purement"),
            record("A", "foi purement civile dont il appartient", "E", "foi purement civile dont il appartient"),
            record("B", "la commune de Paris", "F", "la commune de Paris"),
        ]
        results = os.path.join(root, "alignments.jsonl.lz4")
        with lz4.frame.open(results, "wb") as handle:
            handle.write(b"".join(orjson.dumps(r) + b"\n" for r in records))
        g0, g1 = span("A", PASSAGE)
        groups = os.path.join(root, "passage_group_source.jsonl")
        with open(groups, "wb") as handle:
            handle.write(orjson.dumps({"group_id": 7, "source_filename": path["A"], "source_doc_id": "A",
                                       "source_start_byte": g0, "source_end_byte": g1}) + b"\n")

        rows = [orjson.loads(line) for line in open(project_places(results, groups, 1), "rb")]
        by_doc = {os.path.basename(r["filename"])[0]: r for r in sorted(rows, key=lambda r: -r["start_byte"])}
        check("the anchor's document is no place, every other document's alignments are",
              sorted(by_doc) == ["B", "C", "E", "F"] and len(rows) == 5)
        b = sorted((r for r in rows if r["filename"] == path["B"]), key=lambda r: r["start_byte"])
        check("... and a document's second place, sharing nothing, does not take the first's passage",
              (b[1]["passage_start_byte"], b[1]["passage_end_byte"]) == (b[1]["start_byte"], b[1]["end_byte"])
              and b[1]["shared"] == 0.0)
        check("a place running on past the passage is cut to it",
              by_doc["B"]["passage"] == PASSAGE.replace("donc ", "") and by_doc["B"]["shared"] > 0.9)
        check("a place holding part of the passage grows to all it shares of it",
              by_doc["C"]["passage"] == PASSAGE.replace("il y a donc ", ""))
        check("overlapping spans in one document are one place, its passage the whole",
              by_doc["E"]["passage"] == PASSAGE and by_doc["E"]["shared"] == 1.0)
        f = by_doc["F"]
        check("a place sharing nothing with the passage keeps its span",
              (f["passage_start_byte"], f["passage_end_byte"]) == (f["start_byte"], f["end_byte"]) and f["shared"] == 0.0)
        from textpair.sequence_alignment import family_places
        family_places._MAPS, family_places._STATE["reader"] = 1, family_places._Reader()  # each read evicts the last
        again = [orjson.loads(line) for line in open(project_places(results, groups, 1), "rb")]
        family_places._MAPS = 256
        check("a worker short of mappings projects the same", again == rows)
        check("each place comes with its context",
              by_doc["C"]["context_before"].strip().endswith("le cite :") and by_doc["C"]["context_after"].startswith("."))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("all checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
