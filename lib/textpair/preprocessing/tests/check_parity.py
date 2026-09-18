#!/usr/bin/env python3
"""Byte-exact parity harness against the text_preprocessing library.

Runs both implementations over a PhiloLogic database under many config
permutations and compares every (form, start_byte, end_byte) triple. Requires
the old library to be importable, so it is a migration tool rather than a
permanent test; see test_preprocessing.py for the tests that outlive it.

Usage: parity.py <philo_db_path> [n_files]
"""

from __future__ import annotations

import glob
import itertools
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from textpair.preprocessing import PreProcessor  # noqa: E402

# Config keys are named as the old library expected them; the mapping to the new
# names happens in PreprocessConfig.from_kwargs.
BASE = dict(language="french", is_philo_db=True, workers=1, text_object_type="doc")

FIXTURES = os.environ.get("PARITY_FIXTURES", "")
STOPWORDS = os.path.join(FIXTURES, "stopwords.txt")
LEMMAS = os.path.join(FIXTURES, "lemmas.tsv")


def old_run(files, kwargs, keep_all):
    """Collect (form, start_byte, end_byte) from text_preprocessing.

    Whitespace pseudo-tokens are dropped. The old library emitted one between
    every pair of words because spaCy's Doc carries a `whitespace_` attribute;
    with n-grams enabled its own purge() removed them again, so they only ever
    reached a caller on the no-n-gram path. They are exactly equivalent to
    joining the surface forms with a space, which is what VSA now does.
    """
    from text_preprocessing import PreProcessor as OldPreProcessor
    from text_preprocessing.preprocessor import TextFetcher

    captured = []

    def post(tokens):
        captured.append(
            [
                (str(t), t.ext["start_byte"], t.ext["end_byte"])
                for t in tokens
                if str(t) != " "
            ]
        )
        return {}

    OldPreProcessor(**kwargs, post_processing_function=post)
    # Bypass the pool so post_func results stay in this process.
    for path in files:
        list(TextFetcher._TextFetcher__local_process((path, False, keep_all, post)))
    return captured


def new_run(files, kwargs, keep_all):
    preproc = PreProcessor(**kwargs)
    return [
        list(zip(obj.forms, obj.start_bytes, obj.end_bytes))
        for obj in preproc.process_texts(files, keep_all=keep_all)
    ]


# gap > 0 is deliberately NOT byte-exact: the old implementation double-counted
# and shared one mutable byte range across every combination with the same
# leading token. Checked by verify_skipgrams() instead.
SKIPGRAM = dict(stemmer=True, modernize=True, ngrams=3, ngram_gap=2)

PERMUTATIONS = [
    ("default 3-grams", dict(stemmer=True, modernize=True, ngrams=3)),
    ("no stemmer", dict(stemmer=False, modernize=True, ngrams=3)),
    ("no modernize", dict(stemmer=True, modernize=False, ngrams=3)),
    ("ascii", dict(stemmer=True, modernize=True, ngrams=3, ascii=True)),
    ("no lowercase", dict(stemmer=False, modernize=True, ngrams=3, lowercase=False)),
    ("min_word_length 4", dict(stemmer=True, modernize=True, ngrams=3, min_word_length=4)),
    ("keep numbers", dict(stemmer=True, modernize=True, ngrams=3, strip_numbers=False)),
    ("keep punctuation", dict(stemmer=False, modernize=True, ngrams=3, strip_punctuation=False)),
    ("5-grams", dict(stemmer=True, modernize=True, ngrams=5)),
    ("no ngrams", dict(stemmer=True, modernize=True, ngrams=0)),
    ("2-grams english stemmer", dict(stemmer=True, modernize=False, ngrams=2, language="english")),
    ("sent objects", dict(stemmer=True, modernize=True, ngrams=0, text_object_type="sent")),
    ("para objects, 3-grams", dict(stemmer=True, modernize=True, ngrams=3, text_object_type="para")),
    ("div1 objects", dict(stemmer=True, modernize=True, ngrams=3, text_object_type="div1")),
    ("stopwords", dict(stemmer=True, modernize=True, ngrams=3, stopwords=STOPWORDS)),
    ("stopwords, no stemmer", dict(stemmer=False, modernize=True, ngrams=3, stopwords=STOPWORDS)),
    ("lemmatizer file", dict(stemmer=False, modernize=True, ngrams=3, lemmatizer=LEMMAS)),
    ("lemmatizer + stemmer", dict(stemmer=True, modernize=True, ngrams=3, lemmatizer=LEMMAS)),
    ("lemmatizer + stopwords", dict(stemmer=False, modernize=True, ngrams=3, lemmatizer=LEMMAS, stopwords=STOPWORDS)),
    ("no word order", dict(stemmer=True, modernize=True, ngrams=3, ngram_word_order=False)),
    ("everything", dict(stemmer=True, modernize=True, ngrams=3, ascii=True, stopwords=STOPWORDS,
                        lemmatizer=LEMMAS, min_word_length=3, text_object_type="para")),
]


def verify_skipgrams(files):
    """Check the gap > 0 fix directly rather than against the old output.

    Asserts that every n-gram the old implementation produced still appears,
    that none is emitted twice, and that each byte range really spans its own
    tokens -- the property the shared `ext` dict destroyed.
    """
    from text_preprocessing import PreProcessor as OldPreProcessor
    from text_preprocessing.preprocessor import TextFetcher

    kwargs = {**BASE, **SKIPGRAM}
    old_rows = []

    def post(tokens):
        old_rows.extend((str(t), t.ext["start_byte"], t.ext["end_byte"]) for t in tokens)
        return {}

    OldPreProcessor(**kwargs, post_processing_function=post)
    for path in files:
        list(TextFetcher._TextFetcher__local_process((path, False, False, post)))

    # Rebuild the new output, keeping the token stream so byte ranges can be
    # checked against the tokens each n-gram is actually made of.
    plain = PreProcessor(**{**kwargs, "ngrams": 0})
    preproc = PreProcessor(**kwargs)
    new_rows = []
    spans_ok = True
    token_objects = list(plain.process_texts(files))
    ngram_objects = list(preproc.process_texts(files))
    for tokens, grams in zip(token_objects, ngram_objects):
        by_start = {start: end for start, end in zip(tokens.start_bytes, tokens.end_bytes)}
        seen = set()
        for form, start, end in zip(grams.forms, grams.start_bytes, grams.end_bytes):
            new_rows.append((form, start, end))
            seen.add((form, start, end))
            if start not in by_start:
                spans_ok = False
                print(f"          n-gram start is not a token start: {(form, start, end)}")
            elif end < by_start[start]:
                spans_ok = False
                print(f"          n-gram ends before its own first token: {(form, start, end)}")

    old_forms = {row[0] for row in old_rows}
    new_forms = {row[0] for row in new_rows}
    old_bad = sum(1 for form, start, end in set(old_rows) if end < start)
    print(f"  gap>0: old emitted {len(old_rows):,} rows ({len(old_forms):,} distinct n-grams)")
    print(f"         new emitted {len(new_rows):,} rows ({len(new_forms):,} distinct n-grams)")
    print(f"         duplicate rows in old output: {len(old_rows) - len(set(old_rows)):,}")
    ok = True
    if old_forms != new_forms:
        missing = old_forms - new_forms
        extra = new_forms - old_forms
        print(f"         FAIL n-gram sets differ (missing {len(missing)}, extra {len(extra)})")
        ok = False
    else:
        print("         ok   same set of n-grams as the old implementation")
    if len(new_rows) != len(set(new_rows)):
        print("         FAIL new output contains duplicates")
        ok = False
    else:
        print("         ok   no duplicates in new output")
    if spans_ok:
        print("         ok   every byte range spans its own tokens")
    else:
        print("         FAIL byte ranges inconsistent")
        ok = False
    # Every n-gram sharing a leading token shared one dict, so they all ended up
    # with whichever end byte was written last -- uniformly wrong rather than
    # merely inconsistent. Quantify against the corrected ranges.
    corrected = {(form, start): end for form, start, end in new_rows}
    wrong = sum(
        1 for form, start, end in set(old_rows) if corrected.get((form, start), end) != end
    )
    total = len(set(old_rows))
    print(f"         old rows carrying the wrong byte range: {wrong:,} of {total:,} "
          f"({wrong / total * 100:.1f}%)")
    return ok


# The new implementation deliberately disagrees with two order-dependent
# behaviours in the old one. Both are recognised by shape rather than by field
# name, so a genuinely new difference in any field still fails.
EXPECTED_DIFFS = {
    "cache-poisoned field": (
        lambda field, old, new: old == "<absent>" and new == "",
        "old omitted a field whose value is empty at the object's own level. Its "
        "per-level metadata cache stores only the fields assigned when the entry "
        "was built, so a sibling with a non-empty value poisons the parent entry "
        "for siblings whose value is empty. Which fields vanish depends on "
        "processing order; the new reader caches whole rows and reports the true "
        "toms.db value.",
    ),
    "inherited sentence word_count": (
        lambda field, old, new: field == "word_count" and isinstance(old, str),
        "old set word_count from the token count for every sentence except the "
        "last in each file, which kept the string inherited from the enclosing "
        "paragraph. The new reader is consistent.",
    ),
}


def classify(field, old, new):
    for name, (matches, _) in EXPECTED_DIFFS.items():
        if matches(field, old, new):
            return name
    return None


def compare_metadata(files):
    """Compare text-object metadata field by field, across object types."""
    from text_preprocessing import PreProcessor as OldPreProcessor
    from text_preprocessing.preprocessor import TextFetcher

    ok = True
    for object_type in ("doc", "div1", "div2", "para", "sent"):
        kwargs = {**BASE, "text_object_type": object_type,
                  "stemmer": True, "modernize": True, "ngrams": 0}
        old: list[dict] = []

        def post(tokens):
            old.append(dict(tokens.metadata))
            return {}

        OldPreProcessor(**kwargs, post_processing_function=post)
        for path in files:
            list(TextFetcher._TextFetcher__local_process((path, False, False, post)))
        new = [dict(obj.metadata) for obj in PreProcessor(**kwargs).process_texts(files)]

        if len(old) != len(new):
            print(f"  FAIL  {object_type:5s} object count old={len(old)} new={len(new)}")
            ok = False
            continue
        expected: dict[str, int] = {}
        unexpected: dict[str, tuple] = {}
        for old_meta, new_meta in zip(old, new):
            for key in set(old_meta) | set(new_meta):
                old_value = old_meta.get(key, "<absent>")
                new_value = new_meta.get(key, "<absent>")
                if old_value == new_value:
                    continue
                reason = classify(key, old_value, new_value)
                if reason is None:
                    unexpected.setdefault(key, (old_value, new_value))
                else:
                    expected[reason] = expected.get(reason, 0) + 1
        summary = ", ".join(f"{k} x{v}" for k, v in sorted(expected.items())) or "identical"
        if unexpected:
            ok = False
            print(f"  FAIL  {object_type:5s} {len(new):>6,} objects, unexpected differences:")
            for key, (old_value, new_value) in sorted(unexpected.items()):
                print(f"          {key}: old={old_value!r} new={new_value!r}")
        else:
            print(f"  ok    {object_type:5s} {len(new):>6,} objects, {len(new[0])} fields ({summary})")
    return ok


def main():
    db = sys.argv[1]
    n_files = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    files = sorted(glob.glob(os.path.join(db, "data/words_and_philo_ids/*")))[:n_files]
    if not files:
        sys.exit(f"no words_and_philo_ids files under {db}")

    failures = 0
    for label, overrides in PERMUTATIONS:
        for keep_all in (False, True):
            kwargs = {**BASE, **overrides}
            try:
                expected = old_run(files, kwargs, keep_all)
                actual = new_run(files, kwargs, keep_all)
            except Exception as error:  # noqa: BLE001
                print(f"  ERROR {label} keep_all={keep_all}: {type(error).__name__}: {error}")
                failures += 1
                continue
            flat_expected = list(itertools.chain.from_iterable(expected))
            flat_actual = list(itertools.chain.from_iterable(actual))
            if flat_expected == flat_actual:
                print(f"  ok    {label:28s} keep_all={keep_all!s:5s} {len(flat_actual):>9,} tokens")
                continue
            failures += 1
            print(
                f"  FAIL  {label:28s} keep_all={keep_all!s:5s} "
                f"old={len(flat_expected):,} new={len(flat_actual):,}"
            )
            shown = 0
            for index, (want, got) in enumerate(zip(flat_expected, flat_actual)):
                if want != got and shown < 5:
                    print(f"          [{index}] old={want} new={got}")
                    shown += 1
    print()
    print("metadata:")
    if not compare_metadata(files):
        failures += 1
    print()
    if not verify_skipgrams(files):
        failures += 1
    print()
    print("all checks pass" if not failures else f"{failures} check(s) failed")
    if not failures:
        print()
        print("Deliberate metadata differences, both fixing order-dependent old behaviour:")
        for name, (_, reason) in EXPECTED_DIFFS.items():
            print(f"  {name}: {reason}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
