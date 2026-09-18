#!/usr/bin/env python3
"""Checks the normalization chain's order of operations.

    test_normalization.py

The chain is a pure function of a surface form, which is what lets it be memoized
per form -- the single largest speed win in this package. These cases pin the
order down, because several steps only differ in their result when reordered:
min_word_length is applied after stemming, ascii after that, and stopwords are
tested both before and after lemmatization.
"""
import sys
import tempfile

from textpair.preprocessing.config import PreprocessConfig
from textpair.preprocessing.normalize import Normalizer

FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


def normalizer(**kwargs):
    return Normalizer(PreprocessConfig.from_kwargs(**kwargs))


def write(lines):
    handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    handle.write("\n".join(lines) + "\n")
    handle.close()
    return handle.name


def test_basics():
    plain = normalizer(language="french", stemmer=False, lowercase=True)
    check("lowercase", plain("MAISON"), "maison")
    check("strip whitespace", plain("  maison  "), "maison")
    check("punctuation removed", plain("mai.son"), "maison")
    check("min length drops short", plain("a"), "")
    check("numbers dropped", plain("1789"), "")
    check("mixed digits dropped", plain("x1789"), "")

    kept = normalizer(language="french", stemmer=False, strip_numbers=False)
    check("numbers kept when asked", kept("1789"), "1789")


def test_min_word_length_follows_stemming():
    """A form long enough before stemming can be too short after it."""
    stemmed = normalizer(language="french", stemmer=True, min_word_length=4)
    check("stem then measure", stemmed("eue"), "")
    unstemmed = normalizer(language="french", stemmer=False, min_word_length=4)
    check("measure without stemming", unstemmed("eue"), "")
    # 'avions' stems to 'avion', which clears a length-5 floor; 'avio' would not.
    floor5 = normalizer(language="french", stemmer=True, min_word_length=5)
    check("stem clears the floor", floor5("avions"), "avion")


def test_ascii_applies_last():
    """unidecode runs after the length test, so it cannot rescue a short form."""
    folded = normalizer(language="french", stemmer=False, ascii=True)
    check("ascii folds", folded("étranger"), "etranger")
    short = normalizer(language="french", stemmer=False, ascii=True, min_word_length=9)
    check("length tested before folding", short("étranger"), "")


def test_punctuation_passthrough():
    """With strip_punctuation off, a lone punctuation mark survives verbatim."""
    keep = normalizer(language="french", stemmer=False, strip_punctuation=False)
    check("single mark kept", keep("."), ".")
    check("single mark bypasses min length", keep("!"), "!")
    check("word still normalized", keep("Maison"), "maison")
    strip = normalizer(language="french", stemmer=False, strip_punctuation=True)
    check("mark removed when stripping", strip("."), "")


def test_stopwords_checked_twice():
    """Once against the raw form, once after lemmatization and lowercasing.

    Each list isolates one of the two checks: a capitalised entry can only ever
    match before lowercasing, and a lowercase entry only after.
    """
    capitalised = normalizer(language="french", stemmer=False, lowercase=True,
                             stopwords=write(["Maison"]))
    check("raw form matched", capitalised("Maison"), "")
    check("other casing survives the raw check", capitalised("MAISON"), "maison")

    lowercased = normalizer(language="french", stemmer=False, lowercase=True,
                            stopwords=write(["maison"]))
    check("form matched after lowercasing", lowercased("MAISON"), "")
    check("unrelated form survives", lowercased("cheval"), "cheval")


def test_lemmatizer_file():
    path = write(["chevaux\tcheval", "maisons\tmaison"])
    lemmatized = normalizer(language="french", stemmer=False, lemmatizer=path)
    check("lemma applied", lemmatized("chevaux"), "cheval")
    check("unmapped form untouched", lemmatized("chien"), "chien")


def test_modernize_precedes_the_chain():
    modern = normalizer(language="french", modernize=True, stemmer=False)
    check("modernized", modern.modernize("estoit"), "était")
    check("unmapped untouched", modern.modernize("maison"), "maison")
    plain = normalizer(language="french", modernize=False, stemmer=False)
    check("not modernized when off", plain.modernize("estoit"), "estoit")


def test_memo_matches_uncached():
    memo = normalizer(language="french", stemmer=True, modernize=True, lowercase=True, ascii=True)
    forms = ["Maison", "MAISON", "1789", "a", "étranger", "mai.son", "", "  ", "chevaux"]
    for form in forms:
        check(f"memo({form!r})", memo(form), memo.normalize(form))
    # second lookup comes from the cache
    for form in forms:
        check(f"memo repeat({form!r})", memo(form), memo.normalize(form))


def test_whitespace_forms_survive_when_long_enough():
    """Punctuation separated by spaces normalizes to a space run.

    Not a nicety: these forms reach the n-gram index, and the shell pipeline that
    used to build it stripped their leading spaces, conflating distinct n-grams.
    """
    plain = normalizer(language="french", stemmer=False, min_word_length=2)
    # One space is shorter than min_word_length, two clear it.
    check("two marks and a space", plain("- -"), "")
    check("three marks and two spaces", plain("- - -"), "  ")


def main():
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    if FAILURES:
        print(f"test_normalization: {len(FAILURES)} failure(s)")
        for failure in FAILURES:
            print(f"  {failure}")
        return 1
    print("test_normalization: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
