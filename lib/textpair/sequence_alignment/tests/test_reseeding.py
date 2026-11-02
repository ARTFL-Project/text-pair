#!/usr/bin/env python3
"""A passage's ends must be re-seeded from the ngrams beside them that skip a word.

    test_reseeding.py [--threads N]

Each case is a pair of synthetic documents sharing a passage of ten words and, beside it,
a few words the two do not hold in the same ngrams. Re-seeding joins them to the passage
when an inserted word ("par le sort, dit Montesquieu, est de la nature") or a changed word
leaves them two ngrams that skip it, and not when only one does, nor when they lie further
than max_gap words from the passage, nor toward another passage within twice max_gap words
in either document, which merging is left to decide about. A document whose words file is
missing, or does not fit its ngrams, is not re-seeded; and comparing the corpus in the
reverse order gives the same passages.
"""
import argparse
import os
import shutil
import sys
import tempfile

import orjson
from mmh3 import hash64

from textpair.sequence_alignment import ngram_binary
from test_matcher_symmetry import run_both, undirected

FAILURES = []
WIDTH = 10  # bytes a word takes: eight letters and two spaces
PARAMS = dict(minimum_matching_ngrams=3, minimum_matching_ngrams_in_window=3,
              minimum_matching_ngrams_in_docs=3, max_gap=15, flex_gap=True)


def check(label, ok, got=None):
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        FAILURES.append(label)
        if got is not None:
            print(f"     got {got}")


def words(case, name, count):
    """`count` words of their own for a case."""
    return [f"{case}{name}{k}" for k in range(count)]


def case(c, before_source, before_target, after_source, after_target):
    """A case's two documents: words of each one's own, the words around the passage, and
    the passage of ten words both share."""
    passage = words(c, "y", 10)
    return (words(c, "s", 20) + before_source + passage + after_source + words(c, "t", 20),
            words(c, "u", 20) + before_target + passage + after_target + words(c, "v", 20))


CASES = {
    # a fragment of four words, a word inserted inside it in the target
    "insert": case("i", ["ix0", "ix1", "ix2", "ix3"] + words("i", "m", 3),
                   ["ix0", "ix1", "iins", "ix2", "ix3"] + words("i", "n", 2), [], []),
    # a word inserted after the first of three: one ngram skips it
    "once": case("o", ["ox0", "ox1", "ox2"] + words("o", "m", 3),
                 ["ox0", "oins", "ox1", "ox2"] + words("o", "n", 2), [], []),
    # the fragment of "insert", but further than max_gap words from the passage
    "far": case("f", ["fx0", "fx1", "fx2", "fx3"] + words("f", "m", 17),
                ["fx0", "fx1", "fins", "fx2", "fx3"] + words("f", "n", 17), [], []),
    # the fragment of "insert" before the passage, and before it another passage, near the
    # fragment in the source and too far from it in the target to be merged with
    "near": case("n", words("n", "p", 10) + words("n", "m", 3) + ["nx0", "nx1", "nx2", "nx3"]
                 + words("n", "k", 3),
                 words("n", "p", 10) + words("n", "r", 35) + ["nx0", "nx1", "nins", "nx2", "nx3"]
                 + words("n", "l", 2), [], []),
    # after the passage and a few words of each one's own, a word changed in the target
    "change": case("c", [], [], words("c", "m", 3) + ["cw0", "cw1", "cw2", "cw3", "cw4"],
                   words("c", "n", 2) + ["cw0", "cw1", "cchanged", "cw3", "cw4"]),
}


def letters(word):
    """Eight letters of its own for a word, so the letter extension stops where words differ."""
    value = hash64(word)[0] & 0xFFFFFFFFFFFF
    out = ""
    for _ in range(8):
        out += "abcdefghijklmnopqrstuvwxyz"[value % 26]
        value //= 26
    return out


def write_doc(root, doc_id, words, year, word_file=True, broken=False):
    """A document's text, ngram index and words, word i at byte i * WIDTH."""
    text = "".join(letters(word) + "  " for word in words)
    with open(os.path.join(root, "text", f"{doc_id}.txt"), "w") as handle:
        handle.write(text)
    keys = [hash64("_".join(words[i:i + 3]))[0] for i in range(len(words) - 2)]
    starts = [i * WIDTH for i in range(len(keys))]
    ends = [(i + 2) * WIDTH + 8 for i in range(len(keys))]
    ngram_binary.write_positions(os.path.join(root, "ngrams", f"{doc_id}.bin"), keys, starts, ends)
    if word_file:
        word_keys = [hash64(word)[0] for word in words]
        ngram_binary.write_words(os.path.join(root, "words", f"{doc_id}.bin"),
                                 word_keys[:-1] if broken else word_keys)
    return {"filename": f"text/{doc_id}.txt", "year": str(year), "title": doc_id, "author": "",
            "start_byte": "0", "end_byte": str(len(text))}


def build(root, word_file=True, broken=()):
    for sub in ("ngrams", "words", "metadata", "text"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    metadata = {}
    for k, (name, (source, target)) in enumerate(CASES.items()):
        for doc_id, words, year in ((f"{k}1", source, 1700), (f"{k}2", target, 1800)):
            metadata[doc_id] = write_doc(root, doc_id, words, year + k, word_file, doc_id in broken)
    with open(os.path.join(root, "metadata", "metadata.json"), "wb") as handle:
        handle.write(orjson.dumps(metadata))


def passages(root, workdir, threads):
    """Each case's passages, undirected, from the corpus compared both ways; and whether
    the two ways agree."""
    cwd = os.getcwd()
    os.chdir(root)
    try:
        forward, reverse = run_both(os.path.join(root, "ngrams"),
                                    os.path.join(root, "metadata", "metadata.json"),
                                    workdir, threads, PARAMS)
    finally:
        os.chdir(cwd)
    found, back = undirected(forward), undirected(reverse)
    by_case = {name: sorted((key[2], key[3], key[4], key[5]) for key in found
                            if key[0] == f"{k}1") for k, name in enumerate(CASES)}
    return by_case, found == back


def span(name, first, last):
    """The passage from word `first` to word `last` in both of a case's documents."""
    source, target = CASES[name]
    return [(source.index(first) * WIDTH, source.index(last) * WIDTH + 8,
             target.index(first) * WIDTH, target.index(last) * WIDTH + 8)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    workdir = tempfile.mkdtemp(prefix="textpair_reseeding_")
    try:
        root = os.path.join(workdir, "corpus")
        build(root)
        found, mirrored = passages(root, os.path.join(workdir, "run"), args.threads)
        check("a fragment that two ngrams skipping an inserted word share joins the passage",
              found["insert"] == span("insert", "ix0", "iy9"), found["insert"])
        check("... not one that only one such ngram does",
              found["once"] == span("once", "oy0", "oy9"), found["once"])
        check("... nor one further than max_gap words from the passage",
              found["far"] == span("far", "fy0", "fy9"), found["far"])
        check("... nor one beside which another passage lies within twice max_gap words",
              found["near"] == sorted(span("near", "np0", "np9") + span("near", "ny0", "ny9")),
              found["near"])
        check("a passage's end is moved past a word the two documents spell otherwise",
              found["change"] == span("change", "cy0", "cw4"), found["change"])
        check("comparing the corpus the other way round gives the same passages", mirrored)

        bare = os.path.join(workdir, "bare")
        build(bare, word_file=False)
        without, _ = passages(bare, os.path.join(workdir, "run_bare"), args.threads)
        check("without words files no passage is re-seeded",
              without["insert"] == span("insert", "iy0", "iy9")
              and without["change"] == span("change", "cy0", "cy9"), without)
        broken = os.path.join(workdir, "broken")
        build(broken, broken=("02",))
        ignored, _ = passages(broken, os.path.join(workdir, "run_broken"), args.threads)
        check("... nor with a words file that does not fit its document's ngrams",
              ignored["insert"] == without["insert"] and ignored["change"] == found["change"],
              ignored["insert"])
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("all checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
