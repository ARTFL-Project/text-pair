"""N-gram generation.

Every n-gram carries its own byte range, and with gap > 0 each combination is
emitted exactly once. Both are easy to get wrong by sharing state between
combinations that share a leading token; tests/test_ngrams.py checks them.
"""

from __future__ import annotations

from itertools import combinations

from .tokens import TextObject


def generate(text_object: TextObject, size: int, window: int, word_order: bool) -> TextObject:
    """Build n-grams over the object's non-empty tokens.

    `window` is size + gap: with a gap, every combination of `size` tokens drawn
    from each window of `window` tokens is emitted, once.
    """
    text_object.purge()
    forms = text_object.forms
    start_bytes = text_object.start_bytes
    end_bytes = text_object.end_bytes
    count = len(forms)

    ngram_forms: list[str] = []
    ngram_starts: list[int] = []
    ngram_ends: list[int] = []

    if size <= 0 or count < size:
        return TextObject(ngram_forms, ngram_starts, ngram_ends, metadata=text_object.metadata)

    if window <= size:
        # Contiguous n-grams: the common case, and worth not routing through
        # combinations() for a single answer per position.
        for index in range(count - size + 1):
            group = forms[index : index + size]
            ngram_forms.append("_".join(group if word_order else sorted(group)))
            ngram_starts.append(start_bytes[index])
            ngram_ends.append(end_bytes[index + size - 1])
        return TextObject(ngram_forms, ngram_starts, ngram_ends, metadata=text_object.metadata)

    # Skipgrams, anchored on the leading token so the sliding window emits each
    # combination once. `seen` is still needed: a form repeating inside a window
    # gives the same n-gram over the same span twice ("ou le violon le son"
    # builds "ou_le_son" two ways), which is one occurrence.
    span = window - 1
    seen: set[tuple[str, int]] = set()
    for index in range(count):
        limit = min(index + span, count - 1)
        if limit - index < size - 1:
            break
        start_byte = start_bytes[index]
        for rest in combinations(range(index + 1, limit + 1), size - 1):
            positions = (index, *rest)
            if word_order:
                form = "_".join(forms[position] for position in positions)
            else:
                form = "_".join(sorted(forms[position] for position in positions))
            end_byte = end_bytes[positions[-1]]
            if (form, end_byte) in seen:
                continue
            seen.add((form, end_byte))
            ngram_forms.append(form)
            ngram_starts.append(start_byte)
            ngram_ends.append(end_byte)
        seen.clear()

    return TextObject(ngram_forms, ngram_starts, ngram_ends, metadata=text_object.metadata)
