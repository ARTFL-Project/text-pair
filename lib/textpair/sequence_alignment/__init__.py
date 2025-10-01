"""Sequence alignment submodule containing alignment merger, banality filter, and ngram generation."""

from .alignment_merger import merge_alignments
from .banality_finder import banality_auto_detect, phrase_matcher
from .generate_ngrams import Ngrams

__all__ = [
    'merge_alignments',
    'banality_auto_detect', 
    'phrase_matcher',
    'Ngrams'
]