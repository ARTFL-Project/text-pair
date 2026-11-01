#!/usr/bin/env python3
"""A passage must be extended past its matching ngrams as far as both documents share it.

    test_passage_extension.py

Each case is two small documents and a passage the matcher could have found in them: the
words its ngrams leave out at either end are taken, the extension stops where the texts
part, a last word the two do not share is left out, markup is stepped over, a document's
limits are kept to, and comparing the pair the other way round extends it the same way.
Passages of one pair that overlap by half in both documents once extended become one.
"""
import random
import sys

import numpy as np

from textpair.sequence_alignment.aligner import extension

FAILURES = []


def check(label, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        FAILURES.append(label)


def passage(text, words):
    """The byte range of `words` in text."""
    data = text.encode()
    start = data.index(words.encode())
    return data, start, start + len(words.encode())


def extended(source, source_words, target, target_words, source_limits=None, target_limits=None):
    a, s0, s1 = passage(source, source_words)
    b, t0, t1 = passage(target, target_words)
    found = extension.extend(a, source_limits or (0, len(a)), b, target_limits or (0, len(b)), s0, s1, t0, t1)
    return a[found[0]:found[1]].decode(), b[found[2]:found[3]].decode(), found, (a, b, s0, s1, t0, t1)


def word_codes(text):
    codes, kinds, _, _ = extension._text(np.frombuffer(text.encode(), np.uint8), extension.FOLD, extension.KIND, False)
    return codes, kinds


def main():
    source = "Ainsi parla le roi : le peuple est las de vengeance, et il veut la paix. Puis il partit."
    target = "On lit : «Et le peuple est las de vengeance, et il veut la guerre» dit-il."
    a, b, found, (da, db, s0, s1, t0, t1) = extended(source, "peuple est las de vengeance", target,
                                                     "peuple est las de vengeance")
    check("the words the ngrams leave out are taken at both ends", a == b == "le peuple est las de vengeance, et il veut la")
    mirror = extension.extend(db, (0, len(db)), da, (0, len(da)), t0, t1, s0, s1)
    check("comparing the pair the other way round extends it the same way", mirror == (found[2], found[3], found[0], found[1]))

    a, b, _, _ = extended("Il dit que Thou ſhalt not couet thy neighbours houſe. Fin", "not couet",
                          "Nous lisons: thou shalt not covet thy neighbours house! Autre", "not covet")
    check("spellings the folding makes alike do not stop it", a == "Thou ſhalt not couet thy neighbours houſe" and b == "thou shalt not covet thy neighbours house")

    a, b, _, _ = extended("aux peuples de la terre, en 1727 parut le livre", "peuples de la terre",
                          "aux peuples de la terre, en 1728 vint la guerre", "peuples de la terre")
    check("a word the two spell otherwise at the end is left out", a == b == "aux peuples de la terre, en")

    words = "and the faithfull knowe howe they eate the fleshe"
    codes_a, kinds_a = word_codes(words)
    codes_b, kinds_b = word_codes("and the faithful know how they eat the flesh")
    i, j = extension._trim(codes_a, kinds_a, len(codes_a), codes_b, kinds_b, len(codes_b))
    check("a last word spelt otherwise but alike is kept", (i, j) == (len(codes_a), len(codes_b)))
    codes_c, kinds_c = word_codes("Act. 4. 27")
    codes_d, kinds_d = word_codes("Act. 4. 28")
    i, j = extension._trim(codes_c, kinds_c, len(codes_c), codes_d, kinds_d, len(codes_d))
    check("... a last word that is not, cut back to the one before", (i, j) == (6, 6))

    a, b, _, _ = extended("<p>dit-il: <hi>le</hi> peuple est las de vengeance</hi>, et rien</p>", "peuple est las de vengeance",
                          "<l>Il le dit:</l> <l>le peuple est las de vengeance, et rien de plus</l>", "peuple est las de vengeance")
    check("markup is stepped over, and ends land on words", a == "le</hi> peuple est las de vengeance</hi>, et rien" and b == "le peuple est las de vengeance, et rien")

    source = "un texte, puis le peuple est las de vengeance et la suite"
    target = "un texte, puis le peuple est las de vengeance et la suite"
    da = source.encode()
    a, b, _, _ = extended(source, "peuple est las", target, "peuple est las",
                          source_limits=(da.index(b"le peuple"), da.index(b" et la")))
    check("an extension stops at the limits of a document's text", a == "le peuple est las de vengeance" and b == a)

    a, b, _, _ = extended("Tout autre chose ici. le peuple est las de vengeance Rien ne suit.", "le peuple est las de vengeance",
                          "Nul rapport là-bas! le peuple est las de vengeance Quoi encore?", "le peuple est las de vengeance")
    check("where the texts part at once, nothing is added", a == b == "le peuple est las de vengeance")

    a, b, _, _ = extended("&amp; le peuple est las", "peuple est las", "&amp; le peuple est las", "peuple est las")
    check("a character reference is one character", a == b == "le peuple est las")

    spans = [(0, 10, 100, 110), (2, 20, 102, 120), (9, 25, 500, 510), (40, 50, 140, 150), (44, 60, 144, 160)]
    once = extension.merged(spans)
    check("passages that overlap by half in both documents become one, not those overlapping in one",
          once == [(0, 20, 100, 120), (9, 25, 500, 510), (40, 60, 140, 160)])
    shuffled = spans[:]
    random.Random(3).shuffle(shuffled)
    check("... whatever order they come in", extension.merged(shuffled) == once)
    repeated = [(0, 30, 100, 130), (28, 58, 100, 130), (56, 86, 100, 130)]
    check("copies of a phrase that only touch once extended stay apart", extension.merged(repeated) == repeated)
    chained = [(0, 10, 0, 10), (5, 15, 5, 15), (10, 20, 10, 20)]
    check("... and passages each one with the next are one", extension.merged(chained) == [(0, 20, 0, 20)])

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("all checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
