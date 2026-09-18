"""Token normalization.

The chain is deliberately a function of a plain string, so its result can be
memoized per surface form -- surface forms are Zipfian, so most tokens cost a
dict lookup. The order of the steps is load-bearing and several steps only differ
once reordered; tests/test_normalization.py pins it down.
"""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from html import unescape as unescape_html
from xml.sax.saxutils import unescape as unescape_xml

from Stemmer import Stemmer
from unidecode import unidecode

from .config import PreprocessConfig
from .modernize import load_modernizer

NUMBERS = re.compile(r"\d")

_PUNCT_MAP: dict[int, None] | None = None


def punctuation_map() -> dict[int, None]:
    """str.translate table that deletes every Unicode punctuation character."""
    global _PUNCT_MAP
    if _PUNCT_MAP is None:
        _PUNCT_MAP = dict.fromkeys(
            i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
        )
    return _PUNCT_MAP


def is_punctuation(text: str) -> bool:
    """Whether text is a single punctuation character."""
    return len(text) == 1 and unicodedata.category(text).startswith("P")


# Keyed by (path, mtime), so every Normalizer in a process shares one map and an
# edited file is still picked up. A lemma map runs to a few hundred thousand
# entries; without this each worker would build its own, and under fork the
# parent's copy is shared copy-on-write instead.
_STOPWORDS: dict[tuple[str, float], frozenset[str]] = {}
_LEMMAS: dict[tuple[str, float], dict[str, str]] = {}


def load_stopwords(path: str) -> frozenset[str]:
    if not path:
        return frozenset()
    key = (path, os.path.getmtime(path))
    cached = _STOPWORDS.get(key)
    if cached is None:
        with open(path, encoding="utf-8") as stopword_file:
            cached = frozenset(line.strip() for line in stopword_file if line.strip())
        _STOPWORDS[key] = cached
    return cached


def load_lemmatizer(path: str) -> dict[str, str]:
    if not path or path == "spacy":
        return {}
    key = (path, os.path.getmtime(path))
    cached = _LEMMAS.get(key)
    if cached is None:
        cached = {}
        with open(path, encoding="utf-8") as input_file:
            for line in input_file:
                word, _, lemma = line.strip().partition("\t")
                if lemma:
                    cached[word] = lemma
        _LEMMAS[key] = cached
    return cached


class Normalizer:
    """Applies a configured normalization chain to surface forms.

    Returns "" for a token that should be dropped. Callers decide whether to
    remove it or keep an empty placeholder (the old `keep_all`).
    """

    __slots__ = (
        "config", "modernizer", "stopwords", "lemmas", "stem", "punct_map",
        "lowercase", "strip_punctuation", "strip_numbers", "min_word_length",
        "ascii", "convert_entities", "spacy_lemmatizer", "pos_to_keep",
        "ents_to_keep", "_memo",
    )

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.modernizer = load_modernizer(config.language) if config.modernize else None
        self.stopwords = load_stopwords(config.stopwords)
        self.lemmas = load_lemmatizer(config.lemmatizer)
        if config.stemmer:
            stemmer = Stemmer(config.language)
            stemmer.maxCacheSize = 50000
            self.stem = stemmer.stemWord
        else:
            self.stem = None
        self.punct_map = punctuation_map() if config.strip_punctuation else None
        self.lowercase = config.lowercase
        self.strip_punctuation = config.strip_punctuation
        self.strip_numbers = config.strip_numbers
        self.min_word_length = config.min_word_length
        self.ascii = config.ascii
        self.convert_entities = config.convert_entities
        self.spacy_lemmatizer = config.lemmatizer == "spacy"
        self.pos_to_keep = frozenset(config.pos_to_keep)
        self.ents_to_keep = frozenset(config.ents_to_keep)
        self._memo: dict[str, str] = {}

    def modernize(self, token: str) -> str:
        """Modernize a raw surface form. Applied before everything else."""
        if self.modernizer is None:
            return token
        return self.modernizer.get(token, token)

    def __call__(self, token: str) -> str:
        """Normalize a surface form, memoized. Not valid when spaCy supplies lemmas."""
        try:
            return self._memo[token]
        except KeyError:
            result = self.normalize(token)
            self._memo[token] = result
            return result

    def normalize(self, token: str, pos: str = "", ent_type: str = "", lemma: str = "") -> str:
        """The chain itself. Returns "" for a dropped token."""
        if token in self.stopwords:
            return ""
        # An entity verdict is final: matching ents_to_keep bypasses the POS filter.
        if self.ents_to_keep and ent_type:
            if ent_type not in self.ents_to_keep:
                return ""
        elif self.pos_to_keep and pos not in self.pos_to_keep:
            return ""

        text = token.strip()
        if self.convert_entities:
            text = unescape_xml(unescape_html(text))
        if self.spacy_lemmatizer:
            if lemma:
                text = lemma
        elif self.lemmas:
            text = self.lemmas.get(text, text)
        if self.lowercase:
            text = text.lower()
        if text in self.stopwords:
            return ""
        if self.strip_punctuation:
            text = text.translate(self.punct_map)
        elif is_punctuation(text):
            return text
        if self.strip_numbers and NUMBERS.search(text):
            return ""
        if self.stem is not None:
            text = self.stem(text)
        if len(text) < self.min_word_length:
            return ""
        if self.ascii:
            text = unidecode(text)
        return text
