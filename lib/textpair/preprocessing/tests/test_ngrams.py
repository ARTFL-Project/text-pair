#!/usr/bin/env python3
"""Checks n-gram generation, including the skipgram byte ranges.

    test_ngrams.py

The gap > 0 path is the reason this file exists. text_preprocessing took a
reference to the leading token's `ext` dict and mutated `end_byte` on it, so
every combination sharing a leading token ended up with whichever range was
written last -- half the rows on a real corpus. It also re-emitted every
combination that did not involve the newly arrived token, roughly doubling the
index. Both are checked here directly rather than against the old output.
"""
import sys
from itertools import combinations

from textpair.preprocessing import ngrams
from textpair.preprocessing.tokens import TextObject

FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


def ok(label, condition, detail=""):
    if not condition:
        FAILURES.append(f"{label}{': ' + detail if detail else ''}")


def words(count, prefix="w"):
    """A text object of `count` tokens, token i spanning bytes [10i, 10i+5]."""
    return TextObject(
        forms=[f"{prefix}{i}" for i in range(count)],
        start_bytes=[i * 10 for i in range(count)],
        end_bytes=[i * 10 + 5 for i in range(count)],
        metadata={},
    )


def rows(result):
    return list(zip(result.forms, result.start_bytes, result.end_bytes))


def test_contiguous():
    result = ngrams.generate(words(5), size=3, window=3, word_order=True)
    check("contiguous 3-grams", rows(result), [
        ("w0_w1_w2", 0, 25),
        ("w1_w2_w3", 10, 35),
        ("w2_w3_w4", 20, 45),
    ])


def test_too_short():
    result = ngrams.generate(words(2), size=3, window=3, word_order=True)
    check("fewer tokens than the n-gram size", rows(result), [])


def test_word_order_sorts_the_text_not_the_span():
    result = ngrams.generate(words(3, prefix="z"), size=3, window=3, word_order=False)
    check("forms sorted", result.forms, ["z0_z1_z2"])
    reversed_forms = TextObject(forms=["c", "b", "a"], start_bytes=[0, 10, 20],
                                end_bytes=[5, 15, 25], metadata={})
    result = ngrams.generate(reversed_forms, size=3, window=3, word_order=False)
    check("sorted text", result.forms, ["a_b_c"])
    check("span still first..last", (result.start_bytes[0], result.end_bytes[0]), (0, 25))


def test_skipgram_spans_are_its_own_tokens():
    """Each row's byte range must cover exactly its first and last token."""
    text_object = words(6)
    starts = dict(zip(text_object.forms, text_object.start_bytes))
    ends = dict(zip(text_object.forms, text_object.end_bytes))
    result = ngrams.generate(words(6), size=3, window=5, word_order=True)
    for form, start, end in rows(result):
        parts = form.split("_")
        ok("skipgram start", start == starts[parts[0]], f"{form} start={start}")
        ok("skipgram end", end == ends[parts[-1]], f"{form} end={end}")


def test_skipgram_emits_each_combination_once():
    result = ngrams.generate(words(6), size=3, window=5, word_order=True)
    emitted = rows(result)
    ok("no duplicate rows", len(emitted) == len(set(emitted)),
       f"{len(emitted)} rows, {len(set(emitted))} distinct")
    # Ground truth: every combination of 3 of the 6 tokens whose first and last
    # lie within one window of 5.
    expected = set()
    for group in combinations(range(6), 3):
        if group[-1] - group[0] <= 4:
            expected.add("_".join(f"w{i}" for i in group))
    check("skipgram set", {form for form, _, _ in emitted}, expected)


def test_skipgram_does_not_share_byte_ranges():
    """The old implementation gave every row with the same leading token one range."""
    result = ngrams.generate(words(6), size=3, window=5, word_order=True)
    by_start = {}
    for form, start, end in rows(result):
        by_start.setdefault(start, set()).add(end)
    ok("rows with a shared start have distinct ends",
       any(len(ends) > 1 for ends in by_start.values()),
       "every leading token produced a single end byte, as the old bug did")


def test_gap_zero_matches_contiguous():
    """window == size must take the contiguous path and agree with it."""
    contiguous = ngrams.generate(words(8), size=3, window=3, word_order=True)
    windowed = ngrams.generate(words(8), size=3, window=3, word_order=True)
    check("same rows", rows(contiguous), rows(windowed))


def test_purges_filtered_tokens_first():
    """Empty forms are placeholders; n-grams must be built over survivors only."""
    text_object = TextObject(
        forms=["a", "", "b", "", "c"],
        start_bytes=[0, 10, 20, 30, 40],
        end_bytes=[5, 15, 25, 35, 45],
        metadata={},
    )
    result = ngrams.generate(text_object, size=3, window=3, word_order=True)
    check("filtered tokens skipped", rows(result), [("a_b_c", 0, 45)])


def test_duplicate_forms_in_window_collapse():
    """Distinct combinations that yield the same n-gram over the same bytes are one."""
    text_object = TextObject(
        forms=["ou", "le", "violon", "le", "son"],
        start_bytes=[0, 10, 20, 30, 40],
        end_bytes=[5, 15, 25, 35, 45],
        metadata={},
    )
    result = ngrams.generate(text_object, size=3, window=5, word_order=True)
    emitted = rows(result)
    ok("no duplicate rows", len(emitted) == len(set(emitted)),
       f"{[r for r in emitted if emitted.count(r) > 1]}")
    ok("the repeated form appears once", [f for f, _, _ in emitted].count("ou_le_son") == 1,
       f"{[f for f, _, _ in emitted]}")


def main():
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    if FAILURES:
        print(f"test_ngrams: {len(FAILURES)} failure(s)")
        for failure in FAILURES:
            print(f"  {failure}")
        return 1
    print("test_ngrams: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
