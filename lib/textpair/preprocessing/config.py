"""Preprocessing configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any

# Settings whose [PREPROCESSING] name in the config files differs from the name
# used here. VSA passes the config section through as **kwargs, so without these
# aliases `numbers` and `minimum_word_length` silently fall back to defaults.
ALIASES: dict[str, str] = {
    "numbers": "strip_numbers",
    "minimum_word_length": "min_word_length",
    "ngram": "ngrams",
    "gap": "ngram_gap",
    "word_order": "ngram_word_order",
    "spacy_model": "language_model",
}

DEFAULT_WORD_REGEX = r"[\p{L}\p{M}\p{N}]+|'"


@dataclass(slots=True)
class PreprocessConfig:
    """Everything the preprocessor needs, resolved from a config file section."""

    language: str = "french"
    text_object_type: str = "doc"

    # Normalization, applied in the order listed in normalize.py
    modernize: bool = False
    convert_entities: bool = False
    lemmatizer: str = ""  # path to a tab-separated file, or "spacy"
    lowercase: bool = True
    stopwords: str = ""  # path to a newline-separated file
    strip_punctuation: bool = True
    strip_numbers: bool = True
    stemmer: bool = False
    min_word_length: int = 2
    ascii: bool = False

    # Token filtering, requires language_model
    pos_to_keep: tuple[str, ...] = ()
    ents_to_keep: tuple[str, ...] = ()

    # spaCy pipeline for lemmatization and POS/entity tagging
    language_model: str = ""
    # None means "GPU if the CUDA extra is installed"; False and True force it.
    use_gpu: bool | None = None

    # N-grams. ngrams=0 disables the stage.
    ngrams: int = 0
    # Whether the n-gram strings themselves are wanted. When they are not, and
    # the reader interned its tokens, keys are hashed from the forms directly
    # and no n-gram string is ever built.
    keep_ngram_text: bool = True
    ngram_gap: int = 0
    ngram_word_order: bool = True

    # Only used by process_string, which tokenizes raw text rather than reading
    # a PhiloLogic database.
    word_regex: str = DEFAULT_WORD_REGEX
    sentence_boundaries: tuple[str, ...] = (".", "!", "?")
    strip_tags: bool = False

    def __post_init__(self):
        self.pos_to_keep = _as_tuple(self.pos_to_keep)
        self.ents_to_keep = _as_tuple(self.ents_to_keep)
        self.sentence_boundaries = _as_tuple(self.sentence_boundaries)
        self.lemmatizer = _as_path(self.lemmatizer)
        self.stopwords = _as_path(self.stopwords)
        self.language_model = _as_path(self.language_model)
        self.language = (self.language or "french").lower()
        self.ngrams = int(self.ngrams or 0)
        self.ngram_gap = int(self.ngram_gap or 0)
        self.min_word_length = int(self.min_word_length or 0)
        if self.stopwords and not os.path.isfile(self.stopwords):
            raise FileNotFoundError(f"stopwords file not found: {self.stopwords}")
        if self.lemmatizer not in ("", "spacy") and not os.path.isfile(self.lemmatizer):
            raise FileNotFoundError(f"lemmatizer file not found: {self.lemmatizer}")
        if self.lemmatizer == "spacy" and not self.language_model:
            raise ValueError("lemmatizer = spacy requires spacy_model to be set")
        if self.pos_to_keep and not self.language_model:
            raise ValueError("pos_to_keep requires spacy_model to be set")
        if self.ents_to_keep and not self.language_model:
            raise ValueError("ents_to_keep requires spacy_model to be set")

    @property
    def ngram_window(self) -> int:
        return self.ngrams + self.ngram_gap

    @property
    def needs_spacy(self) -> bool:
        """Whether a spaCy pipeline has to run at all."""
        return bool(self.language_model) and bool(
            self.lemmatizer == "spacy" or self.pos_to_keep or self.ents_to_keep
        )

    @property
    def memoizable(self) -> bool:
        """Whether normalization is a pure function of the surface form.

        False once spaCy supplies a per-occurrence lemma, since the same form can
        then normalize differently depending on context.
        """
        return self.lemmatizer != "spacy"

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "PreprocessConfig":
        """Build from a config-file section, ignoring keys meant for other stages."""
        known = {f.name for f in fields(cls)}
        resolved: dict[str, Any] = {}
        for key, value in kwargs.items():
            key = ALIASES.get(key, key)
            if key in known and value is not None:
                resolved[key] = value
        return cls(**resolved)


def _as_tuple(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        return tuple(i.strip() for i in value.split(",") if i.strip())
    return tuple(value)


def _as_path(value: Any) -> str:
    # configparser hands back the literal strings "False"/"None" for some of these.
    if not value or value in ("False", "None", "false", "none"):
        return ""
    return str(value)
