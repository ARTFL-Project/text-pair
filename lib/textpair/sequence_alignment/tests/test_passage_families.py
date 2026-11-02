#!/usr/bin/env python3
"""The passage grouper must build one family per reused passage.

    test_passage_families.py

Each case is a handful of synthetic alignments between synthetic documents, laid out
so that one rule decides the outcome: overlapping windows of one passage merging, a
window running past two passages not fusing them (even when each has only one window) nor
anchoring a family of its own, a passage quoted inside another's quotation going to its own
family, the floor on fragments, sliding windows over a long copy staying bounded, a passage
aligned only with a member joining its family, a passage that most of its partners put in one
family not being taken into another by a longer window, a family whose earliest occurrence is
only ever a target, a window counting for every document behind it (six identical quotations
of a passage outweighing two of part of it), a group's passage being its anchor's, from
where one of its reuses starts when that is up to five words before, a document in a
family only through a partner staying there only if it holds half its passage or more, and
an alignment that leaves every family it had joining one whose passage both its ends hold,
or else forming a family of its own. Serial and parallel runs must then write the same files.
"""
import os
import random
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

    def doc(self, name, year, size=4000, text=None):
        path = os.path.join(self.root, f"{name}.txt")
        text = text or "".join("abcdefghijklmnopqrstuvwxyz "[(i * 7 + len(self.docs) * 3) % 27] for i in range(size))
        with open(path, "w") as handle:
            handle.write(text)
        self.docs[name] = (path, year, text)

    def align(self, source, s0, s1, target, t0, t1):
        sp, sy, st = self.docs[source]
        tp, ty, tt = self.docs[target]
        self.records.append({
            "source_doc_id": source, "source_filename": sp, "source_start_byte": s0, "source_end_byte": s1,
            "source_title": f"title {source}", "source_year": str(sy), "source_passage": st[s0:s1],
            "target_doc_id": target, "target_filename": tp, "target_start_byte": t0, "target_end_byte": t1,
            "target_title": f"title {target}", "target_year": str(ty), "target_passage": tt[t0:t1],
        })

    def write(self, path, per_frame=3):
        """One lz4 frame per few records, so the output steps can split the file."""
        with open(path, "wb") as handle:
            for i in range(0, len(self.records), per_frame):
                chunk = b"".join(orjson.dumps(r) + b"\n" for r in self.records[i:i + per_frame])
                handle.write(lz4.frame.compress(chunk))


def prose(size, seed):
    """Words that differ, for the cases where what a text holds is compared."""
    rng = random.Random(seed)
    words = []
    while sum(len(w) + 1 for w in words) < size:
        words.append("".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 9))))
    return " ".join(words)[:size]


# QA's two passages; QT quotes the first with the second inside it, as Taine quotes Rousseau
P1, P2, FILLER = prose(200, 1), prose(200, 2), prose(300, 3)
HALF = P1.index(" ", 100) + 1
# a phrase inside two passages, one in each of two families, the M ones with a family of its own too
T, X, Y = prose(120, 5), prose(140, 6), prose(140, 7)
X, Y = f"{X} {T} {prose(140, 8)}", f"{Y} {T} {prose(140, 9)}"


def build(root):
    c = Corpus(root)
    fb = prose(4000, 4)
    texts = {"FB": fb, "FC": fb, "FD": fb[100:], "FE": fb,
             "QA": P1 + FILLER + P2 + FILLER, "QT": P1[:HALF] + P2 + P1[HALF:] + FILLER,
             "QM1": P1 + FILLER, "QM2": P1 + FILLER, "QS": P1 + P2 + FILLER, "QB1": P2 + FILLER, "QB2": P2 + FILLER}
    for prefix in "LM":
        texts.update({prefix + name: text + " " + FILLER for name, text in
                      (("X", X), ("B", X), ("C", X), ("D", Y), ("E", Y), ("F", Y))})
    texts.update({name: T + " " + FILLER for name in ("MA", "MG", "MH")})
    for name, year in (("A", 1600), ("B", 1650), ("C", 1700), ("D", 1750),
                       ("E", 1600), ("F", 1700), ("G", 1700), ("H", 1700), ("I", 1700), ("J", 1700),
                       ("K", 1600), ("L", 1700), ("M", 1700), ("N", 1700), ("O", 1700),
                       ("P", 1600), ("R", 1700), ("S", 1600), ("T", 1700), ("U", 1800),
                       ("V", 1600), ("W", 1700), ("X", 1800), ("Y", 1600), ("Z", 1700), ("YA", 1600), ("YB", 1600), ("ZA", 1700), ("RO", 1600), ("R1", 1700), ("R2", 1700),
                       ("R3", 1700), ("R4", 1700), ("R5", 1700), ("R6", 1700), ("R7", 1700), ("R8", 1700),
                       ("FB", 1700), ("FC", 1600), ("FD", 1650), ("FE", 1650),
                       ("DD", 1600), ("P1", 1700), ("P2", 1700), ("P3", 1700), ("P4", 1700), ("P5", 1700),
                       ("GG", 1750), ("BR", 1800),
                       ("RR", 1600), ("RA1", 1700), ("RA2", 1700), ("RB1", 1700), ("RB2", 1700), ("RS", 1750),
                       ("NR", 1600), ("NA1", 1700), ("NA2", 1700), ("NB1", 1700), ("NB2", 1700), ("NT", 1700),
                       ("QA", 1600), ("QT", 1700), ("QM1", 1700), ("QM2", 1700), ("QS", 1700), ("QB1", 1700),
                       ("QB2", 1700), ("LX", 1600), ("LB", 1700), ("LC", 1700), ("LD", 1650), ("LE", 1700),
                       ("LF", 1700), ("MA", 1550), ("MG", 1700), ("MH", 1700), ("MX", 1600), ("MB", 1700),
                       ("MC", 1700), ("MD", 1650), ("ME", 1700), ("MF", 1700)):
        c.doc(name, year, text=texts.get(name))
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
    # a passage that starts some words into its sentence in Y
    c.align("Y", 170, 380, "Z", 0, 210)
    # RO's sentence, 0-176, quoted alone by two documents in two windows, and with its
    # continuation, 0-515, by six whose spans are all the same window
    c.align("RO", 0, 176, "R1", 0, 176)
    c.align("RO", 3, 130, "R2", 0, 127)
    for other in ("R3", "R4", "R5", "R6", "R7", "R8"):
        c.align("RO", 0, 515, other, 0, 515)
    # passages of YA and YB whose cores two reuses share, and a third reuse starting three words
    # earlier in YA and eight in YB (the texts have a word every 27 bytes)
    c.align("YA", 300, 600, "ZA", 0, 300)
    c.align("YA", 302, 600, "ZA", 400, 698)
    c.align("YA", 250, 600, "ZA", 800, 1150)
    c.align("YB", 300, 600, "ZA", 1300, 1600)
    c.align("YB", 302, 600, "ZA", 1700, 1998)
    c.align("YB", 100, 600, "ZA", 2100, 2600)
    # two passages in FB with a window each, and a longer window over both: it must not fuse them
    c.align("FB", 0, 50, "FC", 0, 50)
    c.align("FB", 100, 174, "FD", 0, 74)
    c.align("FB", 0, 365, "FE", 0, 365)
    # DD's opening is quoted by four documents, once in a slightly longer window and once in a much
    # longer one; GG also quotes the opening, and is aligned with DD only through the long window;
    # BR reuses a stretch of the long window alone
    for other in ("P1", "P2", "P3", "P4"):
        c.align("DD", 172, 221, other, 0, 49)
        c.align("GG", 323, 370, other, 0, 49)
    c.align("DD", 172, 265, "P5", 0, 93)
    c.align("DD", 172, 730, "GG", 323, 417)
    c.align("DD", 235, 447, "BR", 0, 212)
    # RR has two passages with text between them. RS skips that text, so its one window over both
    # runs on in RR, and RS's own copy is aligned with the second passage's reusers too
    c.align("RR", 0, 300, "RA1", 0, 300)
    c.align("RR", 10, 290, "RA2", 0, 280)
    c.align("RR", 600, 900, "RB1", 0, 300)
    c.align("RR", 610, 890, "RB2", 0, 280)
    c.align("RR", 0, 900, "RS", 0, 400)
    c.align("RB1", 0, 300, "RS", 200, 400)
    c.align("RB2", 0, 280, "RS", 200, 400)
    # NT quotes NR's second passage inside its quotation of the first, which NR's window encloses
    c.align("NR", 0, 300, "NA1", 0, 300)
    c.align("NR", 5, 295, "NA2", 0, 290)
    c.align("NR", 10, 290, "NT", 0, 500)
    c.align("NR", 600, 900, "NB1", 0, 300)
    c.align("NR", 605, 895, "NB2", 0, 290)
    c.align("NR", 610, 890, "NT", 150, 350)
    c.align("NB1", 0, 300, "NT", 150, 350)
    # QA's two passages, each with its reuses; QT's quotation, the second inside the first, is aligned
    # with QA at the second only, and as a whole with QM1 and QM2, which hold the first, and QS, both
    n1, n2, q2 = len(P1), len(P2), len(P1) + len(FILLER)
    for other in ("QM1", "QM2", "QS"):
        c.align("QA", 0, n1, other, 0, n1)
    c.align("QA", q2, q2 + n2, "QT", HALF, HALF + n2)
    for other in ("QB1", "QB2"):
        c.align("QA", q2, q2 + n2, other, 0, n2)
    for other in ("QM1", "QM2"):
        c.align(other, 0, n1, "QT", 0, n1 + n2)
    c.align("QS", 0, n1 + n2, "QT", 0, n1 + n2)
    # B and E reuse two families' passages, both holding one phrase, and are aligned at that phrase
    # alone; in the M documents the phrase has a family of its own, which B is in through G
    x, y, t = len(X), len(Y), len(T)
    tx, ty = X.index(T), Y.index(T)
    for prefix in "LM":
        for other in "BC":
            c.align(prefix + "X", 0, x, prefix + other, 0, x)
        for other in "EF":
            c.align(prefix + "D", 0, y, prefix + other, 0, y)
        c.align(prefix + "B", 0, x, prefix + "C", 0, x)
        c.align(prefix + "E", 0, y, prefix + "F", 0, y)
        c.align(prefix + "B", tx, tx + t, prefix + "E", ty, ty + t)
    for other in ("MG", "MH"):
        c.align("MA", 0, t, other, 0, t)
    c.align("MG", 0, t, "MB", tx, tx + t)
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
    with open(os.path.join(work, "passage_group_places.jsonl"), "rb") as handle:
        places = [orjson.loads(line) for line in handle]
    return results, rows, groups_raw, places


def main():
    root = tempfile.mkdtemp(prefix="families_")
    saved_floor = bf._PARALLEL_FLOOR
    try:
        corpus = build(root)
        results, rows, serial_groups, places = run(corpus, root, 1)
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
        check("... and has no family of its own", run_on == first | second)

        ra, rb = families_of("RR", 0, 300), families_of("RR", 600, 900)
        check("a window running on over two passages and the text between them anchors no family",
              families_of("RR", 0, 900) == ra | rb and not [r for r in row_at("RR") if r["source_end_byte"] == 900
                                                           and r["source_start_byte"] < 600])
        check("... its document counts under each passage",
              rows[next(iter(ra))]["count"] == 4 and rows[next(iter(rb))]["count"] == 4)
        check("... and the second passage keeps its reuses", families_of("RB1", 0, 300) == rb)

        na, nb = families_of("NR", 0, 300), families_of("NR", 600, 900)
        check("a passage quoted inside another's quotation goes to its own family",
              families_of("NR", 610, 890) == nb and families_of("NB1", 0, 300) == nb)
        check("... which leaves the other's count", rows[next(iter(na))]["count"] == 4)

        fb_a, fb_b, across = families_of("FB", 0, 50), families_of("FB", 100, 174), families_of("FB", 0, 365)
        check("passages with one window each are not fused by a window over both", not fb_a & fb_b)
        check("... which is listed under each", fb_a <= across and fb_b <= across)

        dd, long_dd, stretch = families_of("DD", 172, 221), families_of("DD", 172, 730), families_of("DD", 235, 447)
        check("a longer window over a passage's opening is listed under it", dd <= long_dd)
        check("... and a quoting document aligned only through it stays in its own family",
              families_of("GG", 323, 370) == dd)
        check("... while a stretch only the long window shares is a family of its own",
              len(stretch - dd) == 1 and not stretch & dd)

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
        ro = row_at("RO")
        check("a window counts for every document behind it: six quoting the whole outweigh two quoting part",
              len(ro) == 1 and (ro[0]["source_start_byte"], ro[0]["source_end_byte"], ro[0]["count"]) == (0, 515, 9))
        check("a group's passage starts where its anchor's does",
              y.get(380) == 170)
        check("... or where a reuse of it starts, a few words before",
              {r["source_end_byte"]: r["source_start_byte"] for r in row_at("YA")}.get(600) == 250)
        check("... but not many words before",
              {r["source_end_byte"]: r["source_start_byte"] for r in row_at("YB")}.get(600) == 300)
        first, second = families_of("QA", 0, len(P1)), families_of("QA", len(P1) + len(FILLER), len(P1) + len(FILLER) + len(P2))
        check("a document in a family only through a partner, holding none of its passage, leaves it",
              families_of("QM1", 0, len(P1)) == first and len(first) == 1 and not first & second)
        check("... one holding it stays", families_of("QS", 0, len(P1) + len(P2)) == first | second)
        check("... and the family counts the documents left",
              rows[next(iter(second))]["count"] == 5 and rows[next(iter(first))]["count"] == 5)
        held = {os.path.basename(p["filename"]) for p in places if p["group_id"] == next(iter(second))}
        check("... and has a place for each, the anchor's document apart", held == {"QT.txt", "QS.txt", "QB1.txt", "QB2.txt"})
        lx, ld, lt = families_of("LX", 0, len(X)), families_of("LD", 0, len(Y)), families_of("LB", X.index(T), X.index(T) + len(T))
        check("an alignment no family of its ends fits forms a family of its own",
              len(lt) == 1 and not lt & (lx | ld) and rows[next(iter(lt))]["count"] == 2
              and rows[next(iter(lt))]["source_filename"] == path["LB"])
        check("... which leaves the others their own documents",
              rows[next(iter(lx))]["count"] == 3 and rows[next(iter(ld))]["count"] == 3)
        mz, mt = families_of("MA", 0, len(T)), families_of("MB", X.index(T), X.index(T) + len(T))
        check("... or joins a family whose passage both its ends hold", mt == mz and rows[next(iter(mz))]["count"] == 5)
        check("every alignment is in a family", all(r["group_id"] for r in results))
        check("every row has the same fields in the same order", len({tuple(r) for r in rows.values()}) == 1)

        bf._PARALLEL_FLOOR = 0  # the file is tiny: make the output steps split it anyway
        parallel_results, _, parallel_groups, parallel_places = run(corpus, root, 4)
        check("parallel output matches serial",
              parallel_results == results and sorted(parallel_groups.splitlines()) == sorted(serial_groups.splitlines())
              and parallel_places == places)
    finally:
        bf._PARALLEL_FLOOR = saved_floor
        shutil.rmtree(root, ignore_errors=True)
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
