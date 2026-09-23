#!/usr/bin/env python3
"""The passage grouper must build one family per reused passage.

    test_passage_families.py

Each case is a handful of synthetic alignments between synthetic documents, laid out
so that one rule decides the outcome: overlapping windows of one passage merging, a
window running past two passages not fusing them, the floor on fragments, sliding
windows over a long copy staying bounded, a passage aligned only with a member
joining its family, a family whose earliest occurrence is only ever a target, and a
group's passage moved back up to five words to start its sentence -- never past a comma or
the like, though an apostrophe does not stop it, and on a word, not an opening quote -- unless
that would make it another group's passage exactly. Serial and parallel runs must then write the same files.
"""
import os
import shutil
import sys
import tempfile

import lz4.frame
import orjson

from textpair.sequence_alignment import alignment_merger as am
from textpair.sequence_alignment import banality_finder as bf

FAILURES = []


def check(label, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        FAILURES.append(label)


class Corpus:
    """Documents on disk, and the alignment records between them."""

    def __init__(self, root):
        self.root = root
        self.docs = {}
        self.records = []
        os.makedirs(os.path.join(root, "words"))

    def dump(self, name, sentences):
        """A words_and_philo_ids dump for `name`: a token every 10 bytes of each (start, end) sentence."""
        doc = sorted(self.docs).index(name) + 1
        lines = []
        for number, (start, end) in enumerate(sentences, 1):
            for word, byte in enumerate(range(start, end, 10), 1):
                lines.append(orjson.dumps({"token": "w", "position": f"{doc} 1 1 1 1 {number} {word} {byte} 0",
                                           "start_byte": byte, "end_byte": min(byte + 9, end)}))
        with lz4.frame.open(os.path.join(self.root, "words", f"{name}.lz4"), "wb") as handle:
            handle.write(b"\n".join(lines) + b"\n")

    def doc(self, name, year, size=4000):
        path = os.path.join(self.root, f"{name}.txt")
        text = "".join("abcdefghijklmnopqrstuvwxyz "[(i * 7 + len(self.docs) * 3) % 27] for i in range(size))
        with open(path, "w") as handle:
            handle.write(text)
        self.docs[name] = (path, year, text)

    def align(self, source, s0, s1, target, t0, t1):
        sp, sy, st = self.docs[source]
        tp, ty, tt = self.docs[target]
        self.records.append({
            "source_doc_id": source, "source_filename": sp, "source_start_byte": s0, "source_end_byte": s1,
            "source_title": f"title {source}", "source_year": str(sy), "source_passage": st[s0:s1],
            "source_parsed_filename": os.path.join(self.root, "words", f"{source}.lz4"),
            "target_doc_id": target, "target_filename": tp, "target_start_byte": t0, "target_end_byte": t1,
            "target_title": f"title {target}", "target_year": str(ty), "target_passage": tt[t0:t1],
            "target_parsed_filename": os.path.join(self.root, "words", f"{target}.lz4"),
        })

    def write(self, path, per_frame=3):
        """One lz4 frame per few records, so the output steps can split the file."""
        with open(path, "wb") as handle:
            for i in range(0, len(self.records), per_frame):
                chunk = b"".join(orjson.dumps(r) + b"\n" for r in self.records[i:i + per_frame])
                handle.write(lz4.frame.compress(chunk))


def build(root):
    c = Corpus(root)
    for name, year in (("A", 1600), ("B", 1650), ("C", 1700), ("D", 1750),
                       ("E", 1600), ("F", 1700), ("G", 1700), ("H", 1700), ("I", 1700), ("J", 1700),
                       ("K", 1600), ("L", 1700), ("M", 1700), ("N", 1700), ("O", 1700),
                       ("P", 1600), ("R", 1700), ("S", 1600), ("T", 1700), ("U", 1800),
                       ("V", 1600), ("W", 1700), ("X", 1800), ("Y", 1600), ("Z", 1700), ("Q", 1700), ("Y2", 1600), ("Y3", 1600)):
        c.doc(name, year)
    # one passage in A, reached through three windows of different extents
    c.align("A", 100, 400, "B", 0, 300)
    c.align("A", 120, 380, "C", 0, 260)
    c.align("A", 100, 300, "D", 0, 200)
    # two passages printed one after the other in E, and a window running across both
    c.align("E", 0, 300, "F", 0, 300)
    c.align("E", 10, 300, "G", 0, 290)
    c.align("E", 400, 700, "H", 0, 300)
    c.align("E", 400, 690, "I", 0, 290)
    c.align("E", 0, 700, "J", 0, 700)
    # a long passage in K, with a fragment of 15% of it and one of 5%
    c.align("K", 0, 1000, "L", 0, 1000)
    c.align("K", 0, 990, "M", 0, 990)
    c.align("K", 500, 650, "O", 0, 150)
    c.align("K", 500, 550, "N", 0, 50)
    # a long copy of P in R, aligned in sliding windows
    for start in range(0, 3000, 150):
        c.align("P", start, start + 300, "R", start, start + 300)
    # X is aligned with W alone, and W with V: X belongs with them, not on its own
    c.align("V", 0, 200, "W", 0, 200)
    c.align("W", 0, 200, "X", 0, 200)
    # passages starting inside sentences of Y (a word every 10 bytes): five words in, six
    # words in, and far into a long sentence
    c.dump("Y", [(0, 120), (120, 400), (400, 1500)])
    c.align("Y", 170, 380, "Z", 0, 210)
    c.align("Y", 460, 700, "Q", 0, 240)
    c.align("Y", 1200, 1400, "Q", 300, 500)
    # in Y2 a short passage four long words into its sentence, and a window running from the
    # sentence's start over it: moving the short one back would give it the other's passage
    c.dump("Y2", [(0, 40), (40, 80), (80, 120), (120, 150)] and [(0, 400)])
    words = [0, 40, 80, 120] + list(range(150, 400, 10))
    doc = sorted(c.docs).index("Y2") + 1
    with lz4.frame.open(os.path.join(root, "words", "Y2.lz4"), "wb") as handle:
        handle.write(b"".join(orjson.dumps({"token": "w", "position": f"{doc} 1 1 1 1 1 {i} {b} 0",
                                            "start_byte": b, "end_byte": b + 5}) + b"\n"
                              for i, b in enumerate(words, 1)))
    c.align("Y2", 150, 200, "Z", 400, 450)
    c.align("Y2", 150, 199, "Z", 500, 549)
    c.align("Y2", 151, 200, "Z", 600, 649)
    c.align("Y2", 0, 200, "Q", 600, 800)
    # in Y3 a comma, then an apostrophe, between a passage and its sentence's start, and a
    # sentence opening on a quote
    tokens = [(0, "Or"), (10, ","), (20, "le")] + [(b, "w") for b in range(30, 200, 10)]
    tokens += [(200, "de"), (210, "l"), (220, "'")] + [(b, "w") for b in range(230, 400, 10)]
    tokens += [(400, '"')] + [(b, "w") for b in range(410, 600, 10)]
    doc = sorted(c.docs).index("Y3") + 1
    with lz4.frame.open(os.path.join(root, "words", "Y3.lz4"), "wb") as handle:
        handle.write(b"".join(orjson.dumps({"token": t, "position": f"{doc} 1 1 1 1 {1 + (b >= 200) + (b >= 400)} {i} {b} 0",
                                            "start_byte": b, "end_byte": b + 5,
                                            "philo_type": "punct" if t in ",'\"" else "word"}) + b"\n"
                              for i, (b, t) in enumerate(tokens, 1)))
    c.align("Y3", 30, 190, "Z", 700, 860)
    c.align("Y3", 230, 390, "Q", 900, 1060)
    c.align("Y3", 430, 590, "Z", 1100, 1260)
    # S is the earliest occurrence but only ever a target
    c.align("T", 0, 200, "S", 0, 200)
    c.align("T", 0, 200, "U", 0, 200)
    return c


def run(corpus, root, workers):
    work = os.path.join(root, f"run{workers}")
    os.makedirs(work)
    path = os.path.join(work, "alignments.jsonl.lz4")
    corpus.write(path)
    groups_file = am.merge_alignments(path, len(corpus.records), workers)
    with lz4.frame.open(path) as handle:
        results = [orjson.loads(line) for line in handle]
    with open(groups_file, "rb") as handle:
        groups_raw = handle.read()
    rows = {r["group_id"]: r for r in map(orjson.loads, groups_raw.splitlines())}
    return results, rows, groups_raw


def main():
    root = tempfile.mkdtemp(prefix="families_")
    saved_floor = bf._PARALLEL_FLOOR
    try:
        corpus = build(root)
        results, rows, serial_groups = run(corpus, root, 1)
        path = {name: p for name, (p, _, _) in corpus.docs.items()}

        def families_of(source, s0, s1):
            for r in results:
                if (r["source_filename"], r["source_start_byte"], r["source_end_byte"]) == (path[source], s0, s1):
                    return set(r["group_id"])
            raise KeyError((source, s0, s1))

        def row_at(name):
            return [r for r in rows.values() if r["source_filename"] == path[name]]

        a = families_of("A", 100, 400) & families_of("A", 120, 380) & families_of("A", 100, 300)
        check("windows of one passage share one family", len(a) == 1)
        check("... whose row counts its four documents", rows[a.pop()]["count"] == 4)

        first, second = families_of("E", 0, 300), families_of("E", 400, 700)
        check("two neighbouring passages stay apart", not first & second)
        run_on = families_of("E", 0, 700)
        check("the window across both is listed under each", first <= run_on and second <= run_on)
        check("... and still has a family of its own", len(run_on - first - second) == 1)

        k = families_of("K", 0, 1000)
        check("a fragment 15% of a passage joins it", families_of("K", 500, 650) == k)
        check("a fragment 5% of a passage does not", not families_of("K", 500, 550) & k)

        spans = [r["source_end_byte"] - r["source_start_byte"] for r in row_at("P")]
        check("sliding windows over a long copy make several families", len(spans) >= 3)
        check("... none covering the whole copy", max(spans) < 1000)

        v = families_of("V", 0, 200)
        check("a passage aligned only with a member joins its family", families_of("W", 0, 200) == v)
        check("... which counts all three documents", rows[next(iter(v))]["count"] == 3)

        s_rows = row_at("S")
        check("a family is anchored on its earliest occurrence", len(s_rows) == 1)
        check("... whose metadata is taken from the target side it appears on",
              s_rows and s_rows[0]["source_doc_id"] == "S" and s_rows[0]["source_title"] == "title S")
        y = {r["source_end_byte"]: r["source_start_byte"] for r in row_at("Y")}
        check("a group's passage five words into its sentence starts at the sentence", y.get(380) == 120)
        check("... six words in, it keeps its start", y.get(700) == 460)
        check("... and far into a long sentence, too", y.get(1400) == 1200)
        y2 = sorted(r["source_start_byte"] for r in row_at("Y2") if r["source_end_byte"] == 200)
        check("... and when that would give it another group's passage exactly", y2 == [0, 150])
        y3 = {r["source_end_byte"]: r["source_start_byte"] for r in row_at("Y3")}
        check("... never back past a comma", y3.get(190) == 20)
        check("... though an apostrophe does not stop it", y3.get(390) == 200)
        check("... and it starts on a word, not on an opening quote", y3.get(590) == 410)
        check("... while the reuses keep their own spans",
              any(r["source_filename"] == path["Y"] and r["source_start_byte"] == 170 for r in results))
        check("every alignment is in a family", all(r["group_id"] for r in results))
        check("every row has the same fields in the same order", len({tuple(r) for r in rows.values()}) == 1)

        bf._PARALLEL_FLOOR = 0  # the file is tiny: make the output steps split it anyway
        parallel_results, _, parallel_groups = run(corpus, root, 4)
        check("parallel output matches serial",
              parallel_results == results and sorted(parallel_groups.splitlines()) == sorted(serial_groups.splitlines()))
    finally:
        bf._PARALLEL_FLOOR = saved_floor
        shutil.rmtree(root, ignore_errors=True)
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
