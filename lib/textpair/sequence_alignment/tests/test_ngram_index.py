#!/usr/bin/env python3
"""Checks the corpus n-gram index against counts computed independently.

    test_ngram_index.py

Replaces a shell pipeline that had two defects of its own, both checked here:

  `cat` over temp files that did not end in a newline welded the last n-gram of
  each document to the first of the next, so every file boundary produced one
  entry whose key field was a key with an n-gram glued to it. That is why
  banality_finder and tracing both skip non-numeric keys.

  The awk that stripped `uniq -c`'s count used a greedy whitespace class, so it
  also ate any leading whitespace belonging to the n-gram. Forms of punctuation
  separated by spaces normalize to space runs and do reach the index.

Every case runs twice: once as GNU sort does it, with the whole file list passed
through --files0-from, and once as BSD sort must, with the list merged in batches
of file arguments. The batch is shrunk to three so the rounds are exercised on a
handful of files rather than needing 100k.
"""
import collections
import os
import sys
import tempfile

from textpair.sequence_alignment import ngram_index

FAILURES = []

# Set per case by main(): (supports --files0-from, batch ceiling).
MODE = "gnu"


def configure(mode):
    """Force the GNU or the BSD merge path."""
    global MODE
    MODE = mode
    program, supports = ngram_index.sort_program()
    if mode == "bsd":
        ngram_index._SORT = (program, False)
        ngram_index.MAX_BATCH = 3
    else:
        ngram_index._SORT = (program, supports)
        ngram_index.MAX_BATCH = 1024


def check(label, got, want):
    if got != want:
        FAILURES.append(f"[{MODE}] {label}: got {got!r}, want {want!r}")


def ok(label, condition, detail=""):
    if not condition:
        FAILURES.append(f"[{MODE}] {label}{': ' + detail if detail else ''}")


def write_corpus(root, documents, trailing_newline=True):
    """documents: list of lists of (ngram, key). Written sorted, as a worker does."""
    os.makedirs(os.path.join(root, "temp"), exist_ok=True)
    os.makedirs(os.path.join(root, "index"), exist_ok=True)
    for number, entries in enumerate(documents):
        lines = sorted(f"{ngram}\t{key}" for ngram, key in entries)
        with open(os.path.join(root, "temp", str(number)), "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
            if trailing_newline:
                handle.write("\n")
    return {f"{ngram}\t{key}": None for entries in documents for ngram, key in entries}


def read(root):
    index = [line.rstrip("\n") for line
             in open(os.path.join(root, "index", "index.tab"), encoding="utf-8") if line.strip()]
    common = [line.strip() for line
              in open(os.path.join(root, "index", "most_common_ngrams.txt"), encoding="utf-8")
              if line.strip()]
    return index, common


def ground_truth(documents):
    counts = collections.Counter()
    for entries in documents:
        for ngram, key in entries:
            counts[f"{ngram}\t{key}"] += 1
    return counts


def test_counts_and_order():
    # Eight documents, so the BSD path needs three merge rounds at a batch of 3.
    documents = [
        [("a_b_c", 1), ("b_c_d", 2), ("a_b_c", 1)],
        [("a_b_c", 1), ("x_y_z", 3)],
        [("b_c_d", 2), ("x_y_z", 3), ("x_y_z", 3)],
        [("a_b_c", 1)],
        [("m_n_o", 4)],
        [("x_y_z", 3)],
        [("b_c_d", 2), ("m_n_o", 4)],
        [("z_z_z", 5)],
    ]
    with tempfile.TemporaryDirectory() as root:
        write_corpus(root, documents)
        distinct = ngram_index.build(root)
        index, common = read(root)
        truth = ground_truth(documents)
        check("distinct count", distinct, len(truth))
        check("index lines", len(index), len(truth))
        check("index set", set(index), set(truth))
        check("index is lexicographic", index, sorted(index))
        # a_b_c 4, x_y_z 4, b_c_d 3, m_n_o 2, z_z_z 1 -- ties lexicographic.
        check("frequency order", common, ["1", "3", "2", "4", "5"])
        check("one line per distinct n-gram", len(common), len(truth))


def test_file_boundaries_are_not_welded():
    """One entry per boundary was corrupted when files lacked a trailing newline."""
    documents = [
        [("zzz_last_ngram", 111)],
        [("aaa_first_ngram", 222)],
        [("mmm_middle", 333)],
    ]
    for trailing in (True, False):
        with tempfile.TemporaryDirectory() as root:
            write_corpus(root, documents, trailing_newline=trailing)
            ngram_index.build(root)
            index, common = read(root)
            label = "with" if trailing else "without"
            check(f"{label} trailing newline: entries", len(index), 3)
            bad = [line for line in index if len(line.split("\t")) != 2
                   or not line.split("\t")[1].lstrip("-").isdigit()]
            check(f"{label} trailing newline: corrupt entries", bad, [])
            nonnumeric = [key for key in common if not key.lstrip("-").isdigit()]
            check(f"{label} trailing newline: non-numeric keys", nonnumeric, [])


def test_leading_whitespace_is_preserved():
    """Space-run n-grams are distinct from their stripped form."""
    documents = [[("  _idem_avec", 42), ("_idem_avec", 43), ("   ", 44)]]
    with tempfile.TemporaryDirectory() as root:
        write_corpus(root, documents)
        ngram_index.build(root)
        index, _ = read(root)
        check("both forms kept", set(index),
              {"  _idem_avec\t42", "_idem_avec\t43", "   \t44"})


def test_negative_keys():
    """mmh3 returns signed int32, so roughly half the keys are negative."""
    documents = [[("a_b_c", -1128609534), ("d_e_f", 1092826535)]]
    with tempfile.TemporaryDirectory() as root:
        write_corpus(root, documents)
        ngram_index.build(root)
        index, common = read(root)
        check("negative key round-trips", set(common), {"-1128609534", "1092826535"})
        check("index entries", len(index), 2)


def test_empty_corpus():
    with tempfile.TemporaryDirectory() as root:
        write_corpus(root, [[]])
        distinct = ngram_index.build(root)
        index, common = read(root)
        check("no distinct n-grams", distinct, 0)
        check("empty index", index, [])
        check("empty most_common", common, [])


def test_ngrams_containing_spaces():
    """A form can contain a space, so splitting on whitespace would lose the key."""
    documents = [[("a b_c d_e f", 7), ("a b_c d_e f", 7), ("plain_n_gram", 8)]]
    with tempfile.TemporaryDirectory() as root:
        write_corpus(root, documents)
        ngram_index.build(root)
        index, common = read(root)
        check("spaces preserved", set(index), {"a b_c d_e f\t7", "plain_n_gram\t8"})
        check("more frequent first", common, ["7", "8"])


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
