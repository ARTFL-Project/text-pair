"""Optional spaCy stage for lemmatization and POS/entity filtering.

Only the per-token attributes that are read -- pos, lemma, entity type -- are
pulled out of each Doc, which is then discarded. Nothing is stitched back
together, so long documents need no Doc.from_docs round trip.
"""

from __future__ import annotations

import os
from typing import Iterable, Iterator

from .config import PreprocessConfig
from .normalize import Normalizer
from .tokens import TextObject

# Pipes nothing downstream reads. Sentence boundaries come from PhiloLogic, so a
# parser or senter would only cost time.
ALWAYS_EXCLUDED = ("parser", "senter", "textcat", "textcat_multilabel")
LEMMATIZER_PIPES = ("lemmatizer", "trainable_lemmatizer")

# Tokens per Doc handed to the pipeline. Segmentation exists only so one very long
# text object cannot exhaust VRAM, and is not free: a transformer lemmatizer's
# output depends on its context window, so a boundary shifts the lemmas near it.
# Hence a threshold high enough that ordinary documents are never split.
SEGMENT_TOKENS = 20000
# Tokens in flight per pipeline call. Bounds VRAM; throughput is insensitive to
# it, since spacy-transformers windows each document internally anyway.
GPU_BATCH_TOKENS = 20000
CPU_BATCH_TOKENS = 4000

# Loaded pipelines, keyed by what makes them different. VSA builds a source and a
# target preprocessor, usually against the same model; without this they would
# each put a copy of the transformer weights on the GPU.
_PIPELINES: dict[tuple, object] = {}


def gpu_available() -> bool:
    """Whether a usable CUDA device is present.

    Probes cupy rather than calling thinc's prefer_gpu(): cupy is only installed
    by the `cuda` extra, so its absence is the signal that this is a CPU install.
    """
    if os.environ.get("TEXTPAIR_DISABLE_GPU"):
        return False
    try:
        import cupy
    except ImportError:
        return False
    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def excluded_pipes(config: PreprocessConfig) -> list[str]:
    """Pipes to leave out of the loaded pipeline.

    Deliberately conservative: tagger and morphologizer stay even when POS is not
    wanted, because trainable lemmatizers take their features from them and
    dropping them degrades lemmas silently. The transformer dominates the cost
    either way.
    """
    excluded = list(ALWAYS_EXCLUDED)
    if not config.ents_to_keep:
        excluded.append("ner")
    if config.lemmatizer != "spacy":
        excluded.extend(LEMMATIZER_PIPES)
    return excluded


class SpacyStage:
    """Runs a spaCy pipeline over deferred text objects and normalizes the result."""

    def __init__(self, config: PreprocessConfig, normalizer: Normalizer, keep_all: bool = False):
        self.keep_all = keep_all
        import spacy
        from spacy.tokens import Doc
        from thinc.api import require_gpu, set_gpu_allocator

        self.config = config
        self.normalizer = normalizer
        self.Doc = Doc

        want_gpu = config.use_gpu if config.use_gpu is not None else gpu_available()
        self.using_gpu = False
        if want_gpu:
            if not gpu_available() and config.use_gpu is True:
                raise RuntimeError(
                    "use_gpu was requested but no CUDA device is usable. Install with "
                    "install.sh -c, or leave use_gpu unset to fall back to CPU."
                )
            if gpu_available():
                # pytorch allocator: the transformer weights live in torch, and
                # letting cupy and torch run separate pools fragments VRAM.
                set_gpu_allocator("pytorch")
                require_gpu()
                self.using_gpu = True

        excluded = excluded_pipes(config)
        cache_key = (config.language_model, tuple(excluded), self.using_gpu)
        if cache_key in _PIPELINES:
            self.nlp = _PIPELINES[cache_key]
        else:
            try:
                self.nlp = spacy.load(config.language_model, exclude=excluded)
            except OSError as error:
                raise RuntimeError(
                    f"Could not load the spaCy model {config.language_model!r}. "
                    "See https://spacy.io/models for installation instructions."
                ) from error
            _PIPELINES[cache_key] = self.nlp

        if config.ents_to_keep and "ner" not in self.nlp.pipe_names:
            raise RuntimeError(
                f"ents_to_keep is set but the model {config.language_model!r} has no NER pipe."
            )
        if config.lemmatizer == "spacy" and not any(
            pipe in self.nlp.pipe_names for pipe in LEMMATIZER_PIPES
        ):
            raise RuntimeError(
                f"lemmatizer = spacy but the model {config.language_model!r} has no lemmatizer pipe."
            )
        self.batch_tokens = GPU_BATCH_TOKENS if self.using_gpu else CPU_BATCH_TOKENS

    def __call__(self, text_objects: Iterable[TextObject]) -> Iterator[TextObject]:
        """Tag, normalize and filter each text object, in order."""
        for batch in self._batched(text_objects):
            segments: list = []
            owners: list[int] = []
            for index, text_object in enumerate(batch):
                for segment in self._segment(text_object):
                    segments.append(segment)
                    owners.append(index)
            tagged: list[list[tuple[str, str, str]]] = [[] for _ in batch]
            for owner, doc in zip(owners, self.nlp.pipe(segments, batch_size=len(segments) or 1)):
                tagged[owner].extend(
                    (token.pos_, token.ent_type_, token.lemma_) for token in doc
                )
            for text_object, attributes in zip(batch, tagged):
                yield self._apply(text_object, attributes)

    def _batched(self, text_objects: Iterable[TextObject]) -> Iterator[list[TextObject]]:
        batch: list[TextObject] = []
        tokens = 0
        for text_object in text_objects:
            batch.append(text_object)
            tokens += len(text_object.forms)
            if tokens >= self.batch_tokens:
                yield batch
                batch = []
                tokens = 0
        if batch:
            yield batch

    def _segment(self, text_object: TextObject) -> list:
        """Split one text object into Docs of at most SEGMENT_TOKENS tokens.

        Splits land on sentence starts where possible. Attributes are read back
        per token and concatenated, so no Doc ever has to be merged.
        """
        words = text_object.forms
        sent_starts = text_object.sent_starts or [False] * len(words)
        if len(words) <= SEGMENT_TOKENS:
            return [self._make_doc(words, sent_starts)]
        docs = []
        start = 0
        while start < len(words):
            end = min(start + SEGMENT_TOKENS, len(words))
            if end < len(words):
                boundary = end
                # Walk back to a sentence start, but never past half a segment.
                while boundary > start + SEGMENT_TOKENS // 2 and not sent_starts[boundary]:
                    boundary -= 1
                if sent_starts[boundary]:
                    end = boundary
            docs.append(self._make_doc(words[start:end], sent_starts[start:end]))
            start = end
        return docs

    def _make_doc(self, words: list[str], sent_starts: list[bool]):
        if not words:
            return self.Doc(self.nlp.vocab, [])
        # The first token must open a sentence or spaCy rejects the Doc.
        starts = list(sent_starts)
        starts[0] = True
        return self.Doc(self.nlp.vocab, words, sent_starts=starts)

    def normalize_words(self, words: list[str]) -> list[str]:
        """Tag and normalize a bare word list, for process_string."""
        if not words:
            return []
        normalize = self.normalizer.normalize
        normalized: list[str] = []
        for doc in self.nlp.pipe(self._segment(TextObject(forms=words))):
            normalized.extend(
                normalize(token.text, token.pos_, token.ent_type_, token.lemma_) for token in doc
            )
        return normalized

    def _apply(
        self, text_object: TextObject, attributes: list[tuple[str, str, str]]
    ) -> TextObject:
        """Normalize with spaCy's per-token output and drop filtered tokens."""
        normalize = self.normalizer.normalize
        keep_all = self.keep_all
        forms = text_object.forms
        kept: list[int] = []
        normalized: list[str] = []
        for index, form in enumerate(forms):
            if index < len(attributes):
                pos, ent_type, lemma = attributes[index]
            else:
                pos, ent_type, lemma = "", "", ""
            result = normalize(form, pos, ent_type, lemma)
            if not result and not keep_all:
                continue
            kept.append(index)
            normalized.append(result)
        if len(kept) == len(forms):
            text_object.forms = normalized
        else:
            text_object.forms = normalized
            text_object.start_bytes = [text_object.start_bytes[i] for i in kept]
            text_object.end_bytes = [text_object.end_bytes[i] for i in kept]
            if text_object.surface_forms:
                text_object.surface_forms = [text_object.surface_forms[i] for i in kept]
            if text_object.positions:
                text_object.positions = [text_object.positions[i] for i in kept]
            if text_object.sent_starts:
                text_object.sent_starts = [text_object.sent_starts[i] for i in kept]
        return text_object
