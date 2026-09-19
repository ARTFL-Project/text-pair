#!/usr/bin/env python3
"""Checks the corpus n-gram index against counts computed independently.

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
    common = [str(key) for key in
              np.fromfile(os.path.join(root, "index", ngram_index.COMMON_NGRAMS),
                          dtype=np.int64).tolist()]
    index = None
    if want_index:
        index = [line.rstrip("\n") for line
                 in open(os.path.join(root, "index", "index.tab"), encoding="utf-8")
                 if line.strip()]
    return index, common


def key_totals(documents):
    """Occurrences per key, which is what the aligner matches on."""
    totals = collections.Counter()
    for entries in documents:
        for _, key in entries:
            totals[key] += 1
    return totals


def check_frequency_order(label, written, totals):
    """The contract: every key once, counts never increasing down the file.

    Ties are not ordered. Keys are hashes, so there is no meaningful order among
    equally frequent ones, and the index does not spend time imposing one.
    """
    check(f"{label}: every key once", sorted(written), sorted(str(k) for k in totals))
    counts = [totals[int(key)] for key in written]
    if any(a < b for a, b in zip(counts, counts[1:])):
        FAILURES.append(f"[{MODE}] {label}: counts increase somewhere in the file")


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
        index, common = read(root)
        totals = key_totals(documents)
        check("distinct keys", distinct, len(totals))
        check("most_common lines", len(common), len(totals))
        check_frequency_order("most_common", common, totals)
        # a_b_c 4, x_y_z 4, b_c_d 3, m_n_o 2, z_z_z 1
        check("counts", dict(totals), {1: 4, 3: 4, 2: 3, 4: 2, 5: 1})
        check("index.tab is the distinct ngram texts", sorted(set(index)), sorted(index))
        check("index.tab entries", len(index), 5)


def test_colliding_keys_get_one_line_with_the_summed_count():
    """Two ngrams, one key: the aligner sees one key, so the index reports one."""
    documents = [
        [("qu_il_le", 172795159)] * 5,
        [("la_liaison_fut", 172795159)] * 2,
        [("other_n_gram", 99)],
    ]
    with tempfile.TemporaryDirectory() as root:
        write_corpus(root, documents)
        distinct = ngram_index.build(root, write_index_tab=True)
        index, common = read(root)
        check("distinct keys", distinct, 2)
        check("the colliding key is listed once", common.count("172795159"), 1)
        check("and ranked by the summed count", common[0], "172795159")
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
        _, common = read(root, want_index=False)
        check("most_common still written", sorted(common), ["1", "2"])


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


def test_negative_keys():
    """mmh3 returns signed int32, so about half the keys are negative."""
    documents = [[("a_b_c", -1128609534), ("d_e_f", 1092826535), ("a_b_c", -1128609534)]]
    with tempfile.TemporaryDirectory() as root:
        write_corpus(root, documents)
        ngram_index.build(root, write_index_tab=True)
        _, common = read(root)
        check("negative key first, it is more frequent", common[0], "-1128609534")
        check("both keys present", sorted(common), sorted(["-1128609534", "1092826535"]))


def test_key_range_spread():
    """Keys are bucketed by range during aggregation; span the int32 space."""
    keys = [-2147483648, -1073741824, -1, 0, 1, 1073741824, 2147483647]
    documents = [[(f"ngram_{i}", key)] * (i + 1) for i, key in enumerate(keys)]
    with tempfile.TemporaryDirectory() as root:
        write_corpus(root, documents)
        distinct = ngram_index.build(root, write_index_tab=False)
        _, common = read(root, want_index=False)
        check("every key survives bucketing", distinct, len(keys))
        check("most frequent first", common[0], str(keys[-1]))
        check("least frequent last", common[-1], str(keys[0]))


def test_high_counts_use_the_log_bands():
    """Counts above EXACT_MAX share power-of-two bands and are sorted within them."""
    counts = {10: 1, 11: 255, 12: 256, 13: 700, 14: 5000, 15: 70000}
    documents = [[(f"n_{key}", key)] * count for key, count in counts.items()]
    with tempfile.TemporaryDirectory() as root:
        write_corpus(root, documents)
        ngram_index.build(root, write_index_tab=False)
        _, common = read(root, want_index=False)
        check_frequency_order("log bands", common, counts)


def test_empty_corpus():
    with tempfile.TemporaryDirectory() as root:
        write_corpus(root, [[]])
        distinct = ngram_index.build(root, write_index_tab=True)
        index, common = read(root)
        check("no distinct keys", distinct, 0)
        check("empty index", index, [])
        check("empty most_common", common, [])


def test_index_tab_preserves_leading_whitespace():
    """Punctuation separated by spaces normalizes to a space run."""
    documents = [[("  _idem_avec", 42), ("_idem_avec", 43), ("   ", 44)]]
    with tempfile.TemporaryDirectory() as root:
        write_corpus(root, documents)
        ngram_index.build(root, write_index_tab=True)
        index, _ = read(root)
        check("both forms kept", set(index),
              {"  _idem_avec\t42", "_idem_avec\t43", "   \t44"})


def test_index_tab_ngrams_containing_spaces():
    """A form can contain a space, so splitting on whitespace would lose the key."""
    documents = [[("a b_c d_e f", 7), ("a b_c d_e f", 7), ("plain_n_gram", 8)]]
    with tempfile.TemporaryDirectory() as root:
        write_corpus(root, documents)
        ngram_index.build(root, write_index_tab=True)
        index, common = read(root)
        check("spaces preserved", set(index), {"a b_c d_e f\t7", "plain_n_gram\t8"})
        check("more frequent first", common, ["7", "8"])


def test_index_tab_dedupes_across_documents():
    documents = [[("same_n_gram", 5)], [("same_n_gram", 5)], [("other_one", 6)]]
    with tempfile.TemporaryDirectory() as root:
        write_corpus(root, documents)
        ngram_index.build(root, write_index_tab=True)
        index, common = read(root)
        check("one line per distinct ngram", sorted(index),
              ["other_one\t6", "same_n_gram\t5"])
        check("both keys present", sorted(common), ["5", "6"])


def test_the_order_documents_arrive_in_only_moves_tied_keys():
    """The n-gram stage spills documents as they come back, not sorted.

    So the same corpus handed over in a different order has to give the same
    keys and the same counts, and may only disagree about which of two equally
    frequent keys is written first. Repeating one order has to be reproducible.
    """
    documents = [
        [("a_b_c", 11), ("b_c_d", 12)],
        [("b_c_d", 12), ("x_y_z", 13)],
        [("x_y_z", 13), ("a_b_c", 11)],
        [("m_n_o", 14), ("a_b_c", 11)],
        [("z_z_z", 15)],
    ]
    totals = key_totals(documents)
    written = {}
    for label, reverse in (("forward", False), ("reverse", True), ("again", False)):
        with tempfile.TemporaryDirectory() as root:
            write_corpus(root, documents)
            directory = os.path.join(root, "ngrams")
            paths = sorted(os.path.join(directory, name)
                           for name in os.listdir(directory))
            index = ngram_index.IncrementalIndex(root)
            for path in reversed(paths) if reverse else paths:
                index.add(path)
            check(f"distinct keys, {label}", index.finish(), len(totals))
            _, common = read(root, want_index=False)
            check_frequency_order(f"most_common, {label}", common, totals)
            written[label] = common
    check("the same keys whichever order they arrive in",
          sorted(written["forward"]), sorted(written["reverse"]))
    check("the same order every time for one arrival order",
          written["forward"], written["again"])


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
