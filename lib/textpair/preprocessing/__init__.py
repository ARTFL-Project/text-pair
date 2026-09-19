"""TextPAIR text preprocessing.

Reads a PhiloLogic database, normalizes, optionally tags with spaCy, optionally
shingles into n-grams. See PREPROCESSING_REWRITE.md for the design and
tests/README.md for what the checks pin down.
"""

from __future__ import annotations

import os
import platform
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import get_context
from typing import Any, Callable, Iterable, Iterator

import regex as re

from . import ngrams as ngram_stage
from .config import PreprocessConfig
from .metadata import PHILO_LEVELS, MetadataLookup
from .normalize import Normalizer
from .philo_reader import sent_metadata, split_text_objects
from .tokens import TextObject

__all__ = [
    "PreProcessor",
    "PreprocessConfig",
    "TextObject",
    "PHILO_LEVELS",
    "worker_start_method",
]

TAGS = re.compile(r"<[^>]+>")

# Per-worker state. Built once by the pool initializer so the modernization map
# and the normalization memo are not rebuilt or pickled per file.
_WORKER: dict[str, Any] = {}


def worker_start_method() -> str:
    """How to start worker processes.

    fork() on macOS deadlocks when a pool is created right after an earlier
    pool -- reliably, in the PhiloLogic-parse-then-preprocess sequence
    (bpo-33725). spawn is the documented fix and costs one interpreter startup
    per worker, which the pool initializer already amortizes.
    """
    override = os.environ.get("TEXTPAIR_START_METHOD")
    if override:
        return override
    return "spawn" if platform.system() == "Darwin" else "fork"


def _init_worker(
    config: PreprocessConfig,
    options: dict[str, Any],
    post_func: Callable[[TextObject], Any] | None,
    finish: bool,
) -> None:
    _WORKER["config"] = config
    _WORKER["normalizer"] = Normalizer(config)
    _WORKER["options"] = options
    _WORKER["post_func"] = post_func
    _WORKER["finish"] = finish


def _process_file(path: str) -> list[Any]:
    """Read one file, and finish it here when there is no spaCy stage.

    Doing the n-gram stage and the caller's post-processing in the worker is what
    keeps the parallelism useful: SA's post-processing collapses each text object
    to a small metadata dict and writes its n-grams straight to disk, so almost
    nothing crosses back.
    """
    config: PreprocessConfig = _WORKER["config"]
    objects = read_and_normalize(path, config, _WORKER["normalizer"], **_WORKER["options"])
    if not _WORKER["finish"]:
        return objects
    return [_finish_object(obj, config, _WORKER["post_func"]) for obj in objects]


def _finish_object(
    text_object: TextObject,
    config: PreprocessConfig,
    post_func: Callable[[TextObject], Any] | None,
) -> Any:
    if config.ngrams:
        text_object = ngram_stage.generate(
            text_object, config.ngrams, config.ngram_window, config.ngram_word_order,
            want_text=config.keep_ngram_text,
        )
    if post_func is not None:
        return post_func(text_object)
    return text_object


def read_and_normalize(
    path: str,
    config: PreprocessConfig,
    normalizer: Normalizer,
    keep_all: bool = False,
    keep_surface: bool = False,
    defer_normalization: bool = False,
    with_metadata: bool = True,
) -> list[TextObject]:
    """Read one PhiloLogic words file into normalized text objects."""
    lookup = MetadataLookup(path) if with_metadata else None
    is_sent = config.text_object_type == "sent"
    results: list[TextObject] = []
    for text_object, object_id in split_text_objects(
        path,
        config.text_object_type,
        normalizer,
        keep_all=keep_all,
        keep_surface=keep_surface,
        defer_normalization=defer_normalization,
    ):
        if lookup is not None:
            text_object.metadata = lookup(object_id, config.text_object_type)
            if is_sent:
                text_object.metadata.update(sent_metadata(text_object))
        else:
            text_object.metadata = {"parsed_filename": path}
        results.append(text_object)
    return results


class PreProcessor:
    """Turns PhiloLogic databases into normalized text objects.

    All configuration is per-instance, so a source and a target preprocessor can
    differ in language, text object type and n-gram settings without interfering.
    """

    def __init__(
        self,
        workers: int | None = None,
        post_processing_function: Callable[[TextObject], Any] | None = None,
        **kwargs: Any,
    ):
        self.config = PreprocessConfig.from_kwargs(**kwargs)
        self.normalizer = Normalizer(self.config)
        self.post_func = post_processing_function
        if workers is None:
            workers = max((os.cpu_count() or 2) - 1, 1)
        self.workers = max(int(workers), 1)
        self.spacy_stage = None
        self.using_gpu = False
        if self.config.needs_spacy:
            from .spacy_stage import SpacyStage

            self.spacy_stage = SpacyStage(self.config, self.normalizer)
            self.using_gpu = self.spacy_stage.using_gpu
        self._string_tokenizer: re.Pattern | None = None

    # ------------------------------------------------------------------
    # Corpus processing
    # ------------------------------------------------------------------

    def process_texts(
        self,
        texts: Iterable[str],
        keep_all: bool = False,
        keep_surface: bool | None = None,
        progress: bool = False,
    ) -> Iterator[Any]:
        """Process a corpus, yielding one result per text object.

        Results are TextObjects, or whatever post_processing_function returns.
        With a spaCy pipeline, reading and normalizing stay in worker processes
        while tagging runs here, so decoding overlaps inference instead of
        alternating with it.
        """
        if keep_surface is None:
            keep_surface = keep_all
        options = {
            "keep_all": keep_all,
            "keep_surface": keep_surface,
            "defer_normalization": self.spacy_stage is not None,
        }
        if self.spacy_stage is not None:
            self.spacy_stage.keep_all = keep_all
        paths = list(texts)
        if not paths:
            return

        # With a spaCy stage the workers stop after reading, because tagging has
        # to happen in the process that owns the model and the GPU.
        finish_in_worker = self.spacy_stage is None
        results = self._read_corpus(paths, options, finish_in_worker)
        if self.spacy_stage is not None:
            for text_object in self.spacy_stage(results):
                yield self._finish(text_object)
        else:
            yield from results

    def _read_corpus(
        self, paths: list[str], options: dict[str, Any], finish_in_worker: bool
    ) -> Iterator[Any]:
        """Yield results from every file, in parallel when it is worth it."""
        if self.workers == 1 or len(paths) == 1:
            for path in paths:
                objects = read_and_normalize(path, self.config, self.normalizer, **options)
                for text_object in objects:
                    if finish_in_worker:
                        yield _finish_object(text_object, self.config, self.post_func)
                    else:
                        yield text_object
            return

        context = get_context(worker_start_method())
        with ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(self.config, options, self.post_func, finish_in_worker),
        ) as executor:
            # One result per file rather than per text object: the pickling cost
            # is per message, and a sentence-level corpus has tens of thousands
            # of text objects per handful of files.
            pending: list[Future] = []
            queue = iter(paths)
            window = self.workers * 2
            for path in queue:
                pending.append(executor.submit(_process_file, path))
                if len(pending) >= window:
                    break
            while pending:
                done = pending.pop(0)
                next_path = next(queue, None)
                if next_path is not None:
                    pending.append(executor.submit(_process_file, next_path))
                yield from done.result()

    def _finish(self, text_object: TextObject) -> Any:
        return _finish_object(text_object, self.config, self.post_func)

    # ------------------------------------------------------------------
    # Single string processing, for passage highlighting
    # ------------------------------------------------------------------

    def _tokenize_string(self, text: str) -> list[str]:
        """Split raw text into surface forms, words and sentence boundaries alike."""
        if self.config.strip_tags:
            text = remove_tags(text)
        if self._string_tokenizer is None:
            boundaries = "".join(self.config.sentence_boundaries)
            self._string_tokenizer = re.compile(
                rf"({self.config.word_regex})|([{re.escape(boundaries)}])"
            )
        return [match[0] for match in self._string_tokenizer.finditer(text)]

    def process_string(self, text: str) -> list[tuple[str, str]]:
        """Tokenize and normalize a raw passage.

        Returns (normalized, surface_form) pairs, with a single-space pair
        between words, so joining the surface forms reconstructs the passage.
        Filtered tokens keep an empty normalized form so callers can mark them.
        """
        surface_forms = self._tokenize_string(text)
        if not surface_forms:
            return []
        modernized = [self.normalizer.modernize(form) for form in surface_forms]
        if self.spacy_stage is not None:
            normalized = self.spacy_stage.normalize_words(modernized)
        elif self.config.memoizable:
            normalize = self.normalizer.__call__
            normalized = [normalize(form) for form in modernized]
        else:
            normalize = self.normalizer.normalize
            normalized = [normalize(form) for form in modernized]

        tokens: list[tuple[str, str]] = []
        last = len(surface_forms) - 1
        for index, (form, surface) in enumerate(zip(normalized, surface_forms)):
            tokens.append((form, surface))
            if index < last:
                tokens.append((" ", " "))
        return tokens

    def process_strings(self, texts: Iterable[str]) -> Iterator[list[str]]:
        """Normalize many raw passages, yielding the surviving forms of each.

        Batches the spaCy pipeline across passages instead of running one Doc
        per call, which is what makes this worth having over a loop around
        process_string: the pipeline is the cost, and it is only efficient when
        fed a whole batch. Surface forms are dropped, since a caller with
        thousands of passages wants a bag of words rather than an alignment
        back to the source text.
        """
        modernize = self.normalizer.modernize
        objects = (
            TextObject(forms=[modernize(form) for form in self._tokenize_string(text)])
            for text in texts
        )
        if self.spacy_stage is not None:
            self.spacy_stage.keep_all = False
            for text_object in self.spacy_stage(objects):
                yield text_object.forms
            return
        normalize = self.normalizer.__call__ if self.config.memoizable else self.normalizer.normalize
        for text_object in objects:
            yield [form for form in map(normalize, text_object.forms) if form]


def remove_tags(text: str) -> str:
    """Strip XML tags, and any TEI header, from raw text."""
    end_header_index: int = text.rfind("</teiHeader>")
    if end_header_index != -1:
        text = text[end_header_index + 12 :]
    return TAGS.sub("", text)
