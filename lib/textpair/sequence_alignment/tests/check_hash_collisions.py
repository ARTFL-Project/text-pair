#!/usr/bin/env python3
"""Measures how often hash collisions fabricate a passage.

    check_hash_collisions.py --philo-db DIR --work DIR [--docs N] [--workers N]
                             [--seeds A B] [--keep]

The aligner matches on a hash of each n-gram, so distinct n-grams that hash
alike are indistinguishable to it. This indexes the same corpus twice under
different mmh3 seeds and diffs the two alignments: collisions under one seed are
almost surely not collisions under the other, so any disagreement is caused by
them. The aligner is checked for determinism first, since a nondeterministic
aligner would make the diff meaningless.

Needs disk: two n-gram indexes and two alignments of the corpus. --docs takes a
prefix of it, which is how the rate was measured against corpus size.

See NGRAM_KEY_COLLISIONS.md for what this found on frantext, and for the
questions it does not answer.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

# Compare passages on what identifies the alignment, not on anything key-derived.
FIELDS = ("source_doc_id", "target_doc_id", "source_start_byte", "source_end_byte",
          "target_start_byte", "target_end_byte")

INDEX_CHILD = r'''
import sys, mmh3
seed = int(sys.argv[3])
import textpair.sequence_alignment.generate_ngrams as gn
# generate_ngrams calls hash64(form)[0]; reseed it in place, keeping the tuple shape.
gn.hash64 = lambda s, _s=seed: mmh3.hash64(s, _s)
from textpair.sequence_alignment.generate_ngrams import Ngrams
Ngrams(text_object_type="doc", ngram=3, gap=0, stemmer=True, lemmatizer="",
       stopwords=False, numbers=True, language="french", lowercase=True,
       minimum_word_length=2, word_order=True, modernize=True, ascii=False,
       pos_to_keep=[], language_model=""
       ).generate(sys.argv[1] + "/data/words_and_philo_ids", sys.argv[2], int(sys.argv[4]))
'''


def subset(philo_db, docs, destination):
    """A PhiloLogic-shaped tree of symlinks, so metadata lookup still finds toms.db."""
    if not docs:
        return philo_db
    words = os.path.join(destination, "data", "words_and_philo_ids")
    shutil.rmtree(destination, ignore_errors=True)
    os.makedirs(words)
    for name in ("toms.db", "TEXT"):
        os.symlink(os.path.join(philo_db, "data", name),
                   os.path.join(destination, "data", name))
    for name in sorted(os.listdir(os.path.join(philo_db, "data", "words_and_philo_ids")))[:docs]:
        os.symlink(os.path.join(philo_db, "data", "words_and_philo_ids", name),
                   os.path.join(words, name))
    return destination


def index(philo_db, out, seed, workers):
    shutil.rmtree(out, ignore_errors=True)
    subprocess.run([sys.executable, "-c", INDEX_CHILD, philo_db, out, str(seed), str(workers)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def align(index_dir, out, threads, extra=()):
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    subprocess.run([sys.executable, "-m", "textpair.sequence_alignment.aligner",
                    f"--source_files={index_dir}/ngrams",
                    f"--source_metadata={index_dir}/metadata/metadata.json",
                    f"--output_path={out}", f"--threads={threads}", *extra],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def passages(root):
    import lz4.frame
    import orjson
    rows = set()
    for path in sorted(glob.glob(f"{root}/result_batches/**/*.lz4", recursive=True)):
        with lz4.frame.open(path) as handle:
            for line in handle:
                record = orjson.loads(line)
                rows.add(tuple(str(record.get(field, "")) for field in FIELDS))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--philo-db", required=True)
    parser.add_argument("--work", required=True, help="scratch directory; needs room for two alignments")
    parser.add_argument("--docs", type=int, default=0, help="use only the first N documents")
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    parser.add_argument("--seeds", type=int, nargs=2, default=(0, 42))
    parser.add_argument("--keep", action="store_true", help="leave the indexes and alignments in place")
    args = parser.parse_args()

    os.makedirs(args.work, exist_ok=True)
    corpus = subset(args.philo_db, args.docs, os.path.join(args.work, "corpus"))
    n_docs = len(os.listdir(os.path.join(corpus, "data", "words_and_philo_ids")))
    print(f"corpus: {n_docs:,} documents from {args.philo_db}")

    runs = {}
    for seed in args.seeds:
        started = time.perf_counter()
        index_dir = os.path.join(args.work, f"index_{seed}")
        align_dir = os.path.join(args.work, f"align_{seed}")
        index(corpus, index_dir, seed, args.workers)
        align(index_dir, align_dir, args.workers)
        runs[seed] = align_dir
        ngrams = sum(1 for _ in open(os.path.join(index_dir, "index", "most_common_ngrams.txt")))
        print(f"  seed {seed}: {ngrams:,} distinct keys, indexed and aligned in "
              f"{time.perf_counter() - started:.0f}s")

    first = args.seeds[0]
    repeat = os.path.join(args.work, "align_repeat")
    align(os.path.join(args.work, f"index_{first}"), repeat, max(args.workers // 2, 1))
    if passages(runs[first]) != passages(repeat):
        print("FAIL the aligner is not deterministic: same input, different threads, "
              "different passages. The seed comparison below would be meaningless.")
        return 1
    print("  aligner is deterministic across thread counts")

    a, b = (passages(runs[seed]) for seed in args.seeds)
    total = len(a | b)
    discordant = len(a - b) + len(b - a)
    print()
    print(f"  seed {args.seeds[0]}: {len(a):,} passages")
    print(f"  seed {args.seeds[1]}: {len(b):,} passages")
    if not discordant:
        print(f"  IDENTICAL over {total:,} passages: no collision changed an alignment, "
              f"so the rate is below {1 / max(total, 1):.1e}")
    else:
        print(f"  discordant: {len(a - b):,} only in {args.seeds[0]}, "
              f"{len(b - a):,} only in {args.seeds[1]}")
        print(f"  rate: {discordant / total:.3e} of {total:,} passages")
        for row in sorted(a - b)[:3]:
            print(f"    only in {args.seeds[0]}: {row}")
    if not args.keep:
        for seed in args.seeds:
            shutil.rmtree(os.path.join(args.work, f"index_{seed}"), ignore_errors=True)
            shutil.rmtree(runs[seed], ignore_errors=True)
        shutil.rmtree(repeat, ignore_errors=True)
        if args.docs:
            shutil.rmtree(corpus, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
