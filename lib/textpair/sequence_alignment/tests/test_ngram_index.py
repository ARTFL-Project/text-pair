#!/usr/bin/env python3
"""Checks the corpus n-gram index against counts computed independently.

The frequencies are documents per key, for the keys in at least MIN_DOCUMENTS of them.

    test_ngram_index.py

Frequencies come from the per-document binary indexes, so a fixture here writes
both what a worker writes: `ngrams/{id}.bin` (the CSR the counts are read from)
and `temp/{id}` (the n-gram text, which only index.tab needs).

index.tab is written only when asked for, since its only reader is the aligner's
--debug tracer. Its cases run twice: once as GNU sort does it, passing the whole
file list through --files0-from, and once as BSD sort must, merging batches of
file arguments. The batch is shrunk to three so the rounds are exercised on a
handful of files.
"""
import collections
import os
import sys
import tempfile

import numpy as np

from textpair.sequence_alignment import ngram_binary, ngram_index

FAILURES = []
MODE = "gnu"


def configure(mode):
    """Force the GNU or the BSD merge route."""
    global MODE
    MODE = mode
    program, supports = ngram_index.sort_program()
    ngram_index._SORT = (program, mode == "gnu" and supports)
    ngram_index.MAX_BATCH = 3 if mode == "bsd" else 1024


def check(label, got, want):
    if got != want:
        FAILURES.append(f"[{MODE}] {label}: got {got!r}, want {want!r}")


def write_corpus(root, documents):
    """documents: list of lists of (ngram, key), in document order, repeats kept."""
    os.makedirs(os.path.join(root, "temp"), exist_ok=True)
    os.makedirs(os.path.join(root, "ngrams"), exist_ok=True)
    os.makedirs(os.path.join(root, "index"), exist_ok=True)
    for number, entries in enumerate(documents):
        keys = [key for _, key in entries]
        starts = [index * 10 for index in range(len(entries))]
        ends = [start + 5 for start in starts]
        ngram_binary.write_positions(
            os.path.join(root, "ngrams", f"{number}.bin"), keys, starts, ends)
        lines = sorted(f"{ngram}\t{key}" for ngram, key in entries)
        with open(os.path.join(root, "temp", str(number)), "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
            if lines:
                handle.write("\n")


def read(root, want_index=True):
    documents, keys, frequencies = ngram_index.read_document_frequencies(
        os.path.join(root, "index", ngram_index.DOCUMENT_FREQUENCIES))
    index = None
    if want_index:
        index = [line.rstrip("\n") for line
                 in open(os.path.join(root, "index", "index.tab"), encoding="utf-8")
                 if line.strip()]
    return index, documents, keys.tolist(), dict(zip(keys.tolist(), frequencies.tolist()))


def document_counts(documents):
    """Documents holding each key, which is what the banality filter reads."""
    counts = collections.Counter()
    for entries in documents:
        for key in {key for _, key in entries}:
            counts[key] += 1
    return counts


def expected(documents):
    """The keys in at least MIN_DOCUMENTS documents, and how many hold each."""
    return {key: count for key, count in document_counts(documents).items()
            if count >= ngram_index.MIN_DOCUMENTS}


def test_counts_come_from_the_binary_indexes():
    documents = [
        [("a_b_c", 1), ("b_c_d", 2), ("a_b_c", 1)],
        [("a_b_c", 1), ("x_y_z", 3)],
        [("b_c_d", 2), ("x_y_z", 3), ("x_y_z", 3)],
        [("m_n_o", 4)],
        [("b_c_d", 2), ("m_n_o", 4)],
        [("z_z_z", 5)],
        [("a_b_c", 1)],
        [("x_y_z", 3)],
    ]
    with tempfile.TemporaryDirectory() as root:
        write_corpus(root, documents)
        distinct = ngram_index.build(root, write_index_tab=True)
        index, count, keys, frequencies = read(root)
        check("distinct keys", distinct, 5)
        check("documents", count, len(documents))
        # a_b_c 3 documents, x_y_z 3, b_c_d 3, m_n_o 2, z_z_z 1
        check("frequencies", frequencies, {1: 3, 2: 3, 3: 3})
        check("frequencies against a recount", frequencies, expected(documents))
        check("ascending keys", keys, sorted(keys))
        check("index.tab is the distinct ngram texts", sorted(set(index)), sorted(index))
        check("index.tab entries", len(index), 5)


def test_repeats_inside_a_document_count_once():
    """A key repeated in one document is that document's, not the corpus's."""
    documents = [[("refrain", 7)] * 100, [("shared", 8)], [("shared", 8)], [("shared", 8)]]
    with tempfile.TemporaryDirectory() as root:
        write_corpus(root, documents)
        ngram_index.build(root)
        check("only the key in three documents", read(root, want_index=False)[3], {8: 3})


def test_colliding_keys_are_counted_once_per_document():
    """Two ngrams, one key: the aligner sees one key, so the index reports one."""
    documents = [
        [("qu_il_le", 172795159)] * 5,
        [("la_liaison_fut", 172795159)] * 2,
        [("qu_il_le", 172795159), ("la_liaison_fut", 172795159)],
        [("other_n_gram", 99)],
    ]
    with tempfile.TemporaryDirectory() as root:
        write_corpus(root, documents)
        distinct = ngram_index.build(root, write_index_tab=True)
        index, _, _, frequencies = read(root)
        check("distinct keys", distinct, 2)
        check("the colliding key, in three documents", frequencies, {172795159: 3})
        # index.tab still carries both texts, which is what the tracer needs.
        check("both texts in index.tab", sorted(index),
              ["la_liaison_fut\t172795159", "other_n_gram\t99", "qu_il_le\t172795159"])


def test_index_tab_is_only_written_when_asked():
    documents = [[("a_b_c", 1), ("b_c_d", 2)]]
    with tempfile.TemporaryDirectory() as root:
        write_corpus(root, documents)
        ngram_index.build(root, write_index_tab=False)
        path = os.path.join(root, "index", "index.tab")
        check("no index.tab by default", os.path.exists(path), False)
        _, count, keys, _ = read(root, want_index=False)
        check("frequencies still written", (count, keys), (1, []))


def test_a_stale_index_tab_is_removed():
    """It would name the wrong ngrams for the keys now in the index."""
    documents = [[("a_b_c", 1)]]
    with tempfile.TemporaryDirectory() as root:
        write_corpus(root, documents)
        path = os.path.join(root, "index", "index.tab")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("stale_n_gram\t999\n")
        ngram_index.build(root, write_index_tab=False)
        check("stale index.tab removed", os.path.exists(path), False)


def test_rebuilding_frequencies_leaves_index_tab_alone():
    """For a finished tree, whose n-gram text is gone, as banality_finder rebuilds it."""
    documents = [[("a_b_c", 1)]] * 3
    with tempfile.TemporaryDirectory() as root:
        write_corpus(root, documents)
        path = os.path.join(root, "index", "index.tab")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("a_b_c\t1\n")
        stale = os.path.join(root, "index", "most_common_ngrams.bin")
        np.arange(3, dtype=np.int64).tofile(stale)
        ngram_index.build(root, write_index_tab=None)
        check("index.tab kept", open(path, encoding="utf-8").read(), "a_b_c\t1\n")
        check("the old frequency order removed", os.path.exists(stale), False)
        check("frequencies", read(root, want_index=False)[3], {1: 3})


def test_key_range_spread():
    """Keys are bucketed by range during aggregation; span the int64 space."""
    keys = [-(1 << 63), -(1 << 62), -1128609534, -1, 0, 1, 1 << 40, (1 << 63) - 1]
    documents = [[(f"ngram_{i}", key) for i, key in enumerate(keys)]] * 3
    with tempfile.TemporaryDirectory() as root:
        write_corpus(root, documents)
        distinct = ngram_index.build(root, write_index_tab=False)
        _, _, written, frequencies = read(root, want_index=False)
        check("every key survives bucketing", distinct, len(keys))
        check("ascending across the buckets", written, sorted(keys))
        check("all in three documents", set(frequencies.values()), {3})


def test_empty_corpus():
    with tempfile.TemporaryDirectory() as root:
        write_corpus(root, [[]])
        distinct = ngram_index.build(root, write_index_tab=True)
        index, count, keys, _ = read(root)
        check("no distinct keys", distinct, 0)
        check("empty index", index, [])
        check("no frequencies", (count, keys), (1, []))


def test_index_tab_preserves_leading_whitespace():
    """Punctuation separated by spaces normalizes to a space run."""
    documents = [[("  _idem_avec", 42), ("_idem_avec", 43), ("   ", 44)]]
    with tempfile.TemporaryDirectory() as root:
        write_corpus(root, documents)
        ngram_index.build(root, write_index_tab=True)
        index = read(root)[0]
        check("both forms kept", set(index),
              {"  _idem_avec\t42", "_idem_avec\t43", "   \t44"})


def test_index_tab_ngrams_containing_spaces():
    """A form can contain a space, so splitting on whitespace would lose the key."""
    documents = [[("a b_c d_e f", 7), ("a b_c d_e f", 7), ("plain_n_gram", 8)]]
    with tempfile.TemporaryDirectory() as root:
        write_corpus(root, documents)
        ngram_index.build(root, write_index_tab=True)
        index = read(root)[0]
        check("spaces preserved", set(index), {"a b_c d_e f\t7", "plain_n_gram\t8"})


def test_index_tab_dedupes_across_documents():
    documents = [[("same_n_gram", 5)], [("same_n_gram", 5)], [("other_one", 6)]]
    with tempfile.TemporaryDirectory() as root:
        write_corpus(root, documents)
        ngram_index.build(root, write_index_tab=True)
        index = read(root)[0]
        check("one line per distinct ngram", sorted(index),
              ["other_one\t6", "same_n_gram\t5"])


def test_the_order_documents_arrive_in_changes_nothing():
    """The n-gram stage spills documents as they come back, not sorted, so the same
    corpus handed over in any order has to give the same file, byte for byte."""
    documents = [
        [("a_b_c", 11), ("b_c_d", 12)],
        [("b_c_d", 12), ("x_y_z", 13)],
        [("x_y_z", 13), ("a_b_c", 11)],
        [("m_n_o", 14), ("a_b_c", 11)],
        [("z_z_z", 15), ("b_c_d", 12)],
    ]
    written = {}
    for label, reverse in (("forward", False), ("reverse", True)):
        with tempfile.TemporaryDirectory() as root:
            write_corpus(root, documents)
            directory = os.path.join(root, "ngrams")
            paths = sorted(os.path.join(directory, name)
                           for name in os.listdir(directory))
            index = ngram_index.IncrementalIndex(root)
            for path in reversed(paths) if reverse else paths:
                index.add(path)
            check(f"distinct keys, {label}", index.finish(), 5)
            with open(os.path.join(root, "index", ngram_index.DOCUMENT_FREQUENCIES),
                      "rb") as handle:
                written[label] = handle.read()
            check(f"frequencies, {label}", read(root, want_index=False)[3],
                  expected(documents))
    check("the same file whichever order they arrive in",
          written["forward"], written["reverse"])


def main():
    for mode in ("gnu", "bsd"):
        configure(mode)
        for name, function in sorted(globals().items()):
            if name.startswith("test_") and callable(function):
                function()
    if FAILURES:
        print(f"test_ngram_index: {len(FAILURES)} failure(s)")
        for failure in FAILURES:
            print(f"  {failure}")
        return 1
    print("test_ngram_index: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
