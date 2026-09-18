"""Global imports for main textpair function.

Four entry points are resolved on first use rather than at import. Each pulls in
a stack the sequence-alignment path never touches -- philologic for parsing,
torch and sentence-transformers for VSA and classification, psycopg2 for the web
loader -- and under `spawn`, which macOS defaults to, every worker process would
otherwise pay for all of it.
"""

from importlib import import_module

from .parse_config import get_config
from .sequence_alignment import (
    Ngrams,
    banality_auto_detect,
    merge_alignments,
    phrase_matcher,
)
from .utils import get_text

_DEFERRED = {
    "parse_files": "textpair.text_parser",
    "run_vsa": "textpair.vector_space_alignment",
    "classify_passages": "textpair.passage_classifier",
    "create_web_app": "textpair.web_loader",
}

__all__ = [
    "Ngrams",
    "banality_auto_detect",
    "get_config",
    "get_text",
    "merge_alignments",
    "phrase_matcher",
    *_DEFERRED,
]


def __getattr__(name: str):
    module = _DEFERRED.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value  # only resolved once
    return value


def __dir__():
    return sorted(__all__)
