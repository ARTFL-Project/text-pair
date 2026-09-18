#!/usr/bin/env python3
"""Checks that configuration is per-instance and that config-file names are honoured.

    test_config.py

text_preprocessing stored configuration on TextFetcher as class variables so
forked workers would inherit it copy-on-write. The cost was that a second
PreProcessor silently overwrote the first, which VSA builds routinely -- so
source_text_object_type, target_text_object_type and target_language did not
actually work. Modernizer had the same shape and aliased its dictionary.

It also accepted only its own parameter names, so `numbers` and
`minimum_word_length` -- the names in sa_config.ini and vsa_config.ini, passed
straight through by VSA -- were silently dropped.
"""
import sys

from textpair.preprocessing import PreProcessor
from textpair.preprocessing.config import PreprocessConfig
from textpair.preprocessing.modernize import load_modernizer

FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


def test_config_file_names_are_honoured():
    """The [PREPROCESSING] names, not just the internal ones."""
    config = PreprocessConfig.from_kwargs(
        numbers=False, minimum_word_length=5, ngram=4, gap=2, word_order=False,
        spacy_model="", language="french",
    )
    check("numbers -> strip_numbers", config.strip_numbers, False)
    check("minimum_word_length -> min_word_length", config.min_word_length, 5)
    check("ngram -> ngrams", config.ngrams, 4)
    check("gap -> ngram_gap", config.ngram_gap, 2)
    check("word_order -> ngram_word_order", config.ngram_word_order, False)
    check("window is size + gap", config.ngram_window, 6)


def test_unknown_keys_are_ignored():
    """VSA passes its whole config section through, vectorizer settings included."""
    config = PreprocessConfig.from_kwargs(
        language="french", vectorization="tfidf", min_freq=0.05, max_freq=0.9,
        embedding_model="some/model", n_chunk=5, min_text_object_length=10,
    )
    check("language still read", config.language, "french")


def test_empty_strings_mean_unset():
    """configparser hands back "" and sometimes the literal "False"."""
    config = PreprocessConfig.from_kwargs(
        language="french", lemmatizer="", stopwords="False", spacy_model="",
    )
    check("empty lemmatizer", config.lemmatizer, "")
    check("'False' stopwords", config.stopwords, "")
    check("no spacy model", config.language_model, "")
    check("needs_spacy false", config.needs_spacy, False)


def test_pos_to_keep_from_a_string():
    config = PreprocessConfig.from_kwargs(language="french", pos_to_keep="NOUN, VERB ,ADJ",
                                          spacy_model="/nonexistent/model")
    check("split and stripped", config.pos_to_keep, ("NOUN", "VERB", "ADJ"))


def test_spacy_only_settings_require_a_model():
    for kwargs, label in (
        ({"lemmatizer": "spacy"}, "lemmatizer = spacy"),
        ({"pos_to_keep": "NOUN"}, "pos_to_keep"),
        ({"ents_to_keep": "PER"}, "ents_to_keep"),
    ):
        try:
            PreprocessConfig.from_kwargs(language="french", **kwargs)
        except ValueError:
            continue
        FAILURES.append(f"{label} without spacy_model was accepted")


def test_missing_files_are_reported():
    for key in ("stopwords", "lemmatizer"):
        try:
            PreprocessConfig.from_kwargs(language="french", **{key: "/nonexistent/file"})
        except FileNotFoundError:
            continue
        FAILURES.append(f"a missing {key} file was accepted")


def test_two_preprocessors_stay_independent():
    """The bug that made source_text_object_type and target_language inoperative."""
    source = PreProcessor(language="french", modernize=True, text_object_type="doc",
                          ngrams=3, workers=1, stemmer=True)
    target = PreProcessor(language="english", modernize=False, text_object_type="sent",
                          ngrams=0, workers=1, stemmer=True)
    check("source language", source.config.language, "french")
    check("source text_object_type", source.config.text_object_type, "doc")
    check("source ngrams", source.config.ngrams, 3)
    check("source modernize", source.config.modernize, True)
    check("target language", target.config.language, "english")
    check("target text_object_type", target.config.text_object_type, "sent")
    check("target ngrams", target.config.ngrams, 0)


def test_modernizers_do_not_alias():
    """Constructing an English modernizer must not disable the French one."""
    french = PreProcessor(language="french", modernize=True, workers=1)
    check("french before", french.normalizer.modernize("estoit"), "était")
    english = PreProcessor(language="english", modernize=True, workers=1)
    check("english unaffected by french", english.normalizer.modernize("estoit"), "estoit")
    check("french after", french.normalizer.modernize("estoit"), "était")
    if french.normalizer.modernizer is english.normalizer.modernizer:
        FAILURES.append("french and english share one map")


def test_same_language_shares_one_map():
    """Two French preprocessors must not hold two copies: forked workers rely on it."""
    first = PreProcessor(language="french", modernize=True, workers=1)
    second = PreProcessor(language="french", modernize=True, workers=1)
    if first.normalizer.modernizer is not second.normalizer.modernizer:
        FAILURES.append("two french preprocessors built separate maps")
    if load_modernizer("french") is not first.normalizer.modernizer:
        FAILURES.append("the module cache is not what instances use")


def test_unsupported_language_has_no_map():
    latin = PreProcessor(language="latin", modernize=True, workers=1)
    check("passes through", latin.normalizer.modernize("quidem"), "quidem")


def main():
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    if FAILURES:
        print(f"test_config: {len(FAILURES)} failure(s)")
        for failure in FAILURES:
            print(f"  {failure}")
        return 1
    print("test_config: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
