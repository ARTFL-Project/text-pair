#!/usr/bin/env python3
"""What two texts share, letter by letter: the group page's shading and "Show differences".

    test_shared_text.py

A group page's passage must say, for each character, how many other documents reuse it:
shared runs found in any order, each word used once, short runs only in a chain, spellings
compared alike, each document counted once, and characters of a script written without
spaces needing a longer run. "Show differences" must cut both passages into what they share,
spelt alike or not, in the same order or not, and what each has alone.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_text import REUSE_WORD, differences, reuse_depth  # noqa: E402

FAILURES = []


def check(label, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        FAILURES.append(label)


def align(document, text, start=0):
    return {"source_doc_id": "A", "source_filename": "A.xml", "source_start_byte": 0, "source_end_byte": 10,
            "source_passage": "", "target_doc_id": document, "target_filename": f"{document}.xml",
            "target_start_byte": start, "target_end_byte": start + len(text), "target_passage": text}


def by_word(text, values):
    """Each word with the value all its characters have, or ? when they differ."""
    out = []
    for m in REUSE_WORD.finditer(text):
        found = set(values[m.start():m.end()])
        out.append(f"{m.group()}{found.pop() if len(found) == 1 else '?'}")
    return " ".join(out)


def depth(passage, rows):
    counts = [count for text, count in reuse_depth(passage, rows, "A")["pieces"] for _ in text]
    return by_word(passage, counts)


def marks(text, pieces):
    kinds = [kind[0] for piece, kind in pieces for _ in piece]  # s(ame), r(espelt), m(oved), d(ifferent)
    return by_word(text, kinds)


def main():
    passage = "le peuple est las de vengeance. Je vous annonce que ce soir le tocsin sonnera"
    found = reuse_depth(passage, [align("B", "ce soir le tocsin sonnera; le peuple est las de vengeance")], "A")
    check("the pieces are the passage", "".join(text for text, _ in found["pieces"]) == passage)
    check("runs count in any order",
          depth(passage, [align("B", "ce soir le tocsin sonnera; le peuple est las de vengeance")])
          == "le1 peuple1 est1 las1 de1 vengeance1 Je0 vous0 annonce0 que0 ce1 soir1 le1 tocsin1 sonnera1")

    passage = "craignez qu'il ne se venge. Nous voulons qu'il soit jugé"
    check("a word of the other text is used once",
          depth(passage, [align("B", "craignez qu'il ne se venge")])
          == "craignez1 qu1 il1 ne1 se1 venge1 Nous0 voulons0 qu0 il0 soit0 jugé0")

    passage = "abandonné par Charles IX, par Henri III, par Louis XIII"
    check("a two-word run alone does not count",
          depth(passage, [align("B", "sous Charles IX on vit")]) == "abandonné0 par0 Charles0 IX0 par0 Henri0 III0 par0 Louis0 XIII0")
    check("... two following each other in both texts do, not the word between them the other lacks",
          depth(passage, [align("B", "Charles IX, Henri III")]) == "abandonné0 par0 Charles1 IX1 par0 Henri1 III1 par0 Louis0 XIII0")

    passage = "le peuple est las de n'être pas vengé. Craignez qu'il"
    check("a word changed inside a reused stretch does not count, the words around it do",
          depth(passage, [align("B", "le peuple est las de n'être point vengé: craignez qu'il")])
          in ("le1 peuple1 est1 las1 de1 n1 être1 pas0 vengé1 Craignez1 qu1 il1", "le1 peuple1 est1 las1 de1 n1 être1 pas? vengé1 Craignez1 qu1 il1"))
    check("... and a short word kept between changed ones counts",
          " et1 " in depth("le peuple est las de sa longue et cruelle vengeance, il est temps",
                           [align("B", "le peuple est las de ta grande et terrible vengeance, il est temps")]))
    check("accents, case, u and v, i and y, long s are compared alike",
          depth("Thou ſhalt not couet thy neighbours house", [align("B", "thou shalt not covet thy neighbours house")])
          == "Thou1 ſhalt1 not1 couet1 thy1 neighbours1 house1")
    check("letters of a word cut otherwise count, what is not in the other text does not",
          depth("thou shalt not covet thy neighbours house", [align("B", "thou shalt not covet thy neighbour's house")])
          == "thou1 shalt1 not1 covet1 thy1 neighbours1 house1")

    passage = "le peuple est las de vengeance"
    rows = [align("B", "le peuple est las"), align("B", "est las de vengeance", 100), align("C", "le peuple est las"),
            {**align("A", "le peuple est las de vengeance"), "target_filename": "A.xml"}]
    found = reuse_depth(passage, rows, "A")
    check("a document counts once however many of its spans reuse a word, the anchor's not at all",
          depth(passage, rows) == "le2 peuple2 est2 las2 de1 vengeance1" and found["documents"] == 2)

    passage = "天下有道則禮樂征伐自天子出"
    check("characters written without spaces need four in a row",
          depth(passage, [align("B", "於是有道則")]) == " ".join(f"{c}0" for c in passage)
          and depth(passage, [align("B", "其有道則禮也")]) == " ".join(f"{c}{int(2 <= i <= 5)}" for i, c in enumerate(passage)))

    check("an empty passage has nothing to count", reuse_depth("", [align("B", "le peuple")], "A")["pieces"] == [])

    source = "Thou ſhalt not couet thy neighbours house, nor his wife. And God spake all these words"
    target = "And God spake all these words: thou shalt not covet thy neighbours house nor his maid"
    found = differences(source, target)
    check("the differences' pieces are the two passages",
          "".join(p for p, _ in found["source"]) == source and "".join(p for p, _ in found["target"]) == target)
    check("a word spelt otherwise is shared, its other letters respelt, a long s not",
          marks(source, found["source"]).startswith("Thou? ſhalts nots")
          and [p for p, kind in found["source"] if kind == "respelt"][:2] == ["T", "u"]
          and [p for p, kind in found["target"] if kind == "respelt"][:2] == ["t", "v"])
    check("text shared in another order is moved, on both sides",
          marks(source, found["source"]).endswith("Andm Godm spakem allm thesem wordsm")
          and marks(target, found["target"]).startswith("Andm Godm spakem allm thesem wordsm"))
    check("what one passage has alone is different",
          "wifed" in marks(source, found["source"]) and "maidd" in marks(target, found["target"]))

    reuse = "le peuple est las de vengeance"
    longer = "Il est temps que le roi parle enfin. " * 30 + reuse + ". Les jours passent et rien ne vient. " * 30
    check("a passage inside a much longer text is found",
          marks(reuse, differences(reuse, longer)["source"]) == "les peuples ests lass des vengeances")
    check("unrelated passages share nothing",
          {kind for _, kind in differences("le peuple est las de vengeance", "the quality of mercy is not strained")["source"]}
          == {"different"})

    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
