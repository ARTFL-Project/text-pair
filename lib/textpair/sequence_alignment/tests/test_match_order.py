#!/usr/bin/env python3
"""Checks the two index properties align_source relies on to order matches cheaply.

    test_match_order.py [NGRAMS_DIR METADATA_JSON] [--threads N]

`align_source` must hand `match_passage` the matches ordered by (source index, target
index). It does not sort the cross-product to get there. Instead it sorts the pair's
source positions and expands each one's target block in place, which is only equivalent
if both of these hold of the loaded corpus:

  1. a source index is unique within its document, so no two source positions tie and
     the expansion order is fully determined;
  2. a key's positions are ascending by index, so expanding one target block emits
     ascending target indices.

Both come from the writers sorting keys stably and so keeping each key's positions in the
order generation collected them (`ngram_binary.write_positions`, `ngram_loader.build_csr_sorted`).
Neither is enforced at load time, so this asserts them, and then asserts that the order
align_source produces really is what a full sort of the cross-product would give.

With no arguments it uses the fixtures.
"""
import argparse
import os
import sys

import numpy as np

from textpair.sequence_alignment.aligner import ngram_loader
from textpair.sequence_alignment.aligner.documents import get_files
from textpair.sequence_alignment.aligner.documents import load_metadata

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ngrams_dir", nargs="?", default="")
    parser.add_argument("metadata", nargs="?", default="")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--documents", type=int, default=250,
                        help="how many documents to load")
    args = parser.parse_args(argv)
    if not args.ngrams_dir:
        root = os.path.join(HERE, "fixtures", "no_byte_range")
        args.ngrams_dir = os.path.join(root, "ngrams")
        args.metadata = os.path.join(root, "metadata", "metadata.json")

    docs = get_files(args.ngrams_dir, load_metadata(args.metadata), "year")[:args.documents]
    (key_offsets, ngram_keys, position_offsets, ngram_indices, start_bytes,
     end_bytes) = ngram_loader.load_corpus(
        [path for _, path in docs], args.threads)
    failures = []

    def check(label, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'} {label}")
        if not ok:
            failures.append(label)
            if detail:
                print(f"     {detail}")

    unique = ascending = True
    blocks = multi = 0
    for doc in range(len(docs)):
        lo_slot, hi_slot = int(key_offsets[doc]), int(key_offsets[doc + 1])
        seen = []
        for slot in range(lo_slot, hi_slot):
            lo, hi = int(position_offsets[slot + doc]), int(position_offsets[slot + doc + 1])
            blocks += 1
            block = ngram_indices[lo:hi]
            if hi - lo > 1:
                multi += 1
                if not np.all(block[:-1] < block[1:]):
                    ascending = False
            seen.append(block)
        if seen:
            allidx = np.concatenate(seen)
            if np.unique(allidx).shape[0] != allidx.shape[0]:
                unique = False

    check(f"source indices are unique within a document ({len(docs)} documents)", unique)
    check(f"key positions ascend by index ({blocks:,} blocks, {multi:,} multi-position)",
          ascending)

    # The order align_source builds, against a full sort of the cross-product.
    rng = np.random.default_rng(0)
    checked = mismatched = 0
    for _ in range(400):
        s, t = rng.integers(0, len(docs), 2)
        if s == t:
            continue
        s_keys = ngram_keys[key_offsets[s]:key_offsets[s + 1]]
        t_keys = ngram_keys[key_offsets[t]:key_offsets[t + 1]]
        shared = np.intersect1d(s_keys, t_keys, assume_unique=True)
        if shared.shape[0] < 2:
            continue
        cheap, full = [], []
        for key in shared:
            a = int(key_offsets[s]) + int(np.searchsorted(s_keys, key))
            b = int(key_offsets[t]) + int(np.searchsorted(t_keys, key))
            sp = range(int(position_offsets[a + s]), int(position_offsets[a + s + 1]))
            tp = range(int(position_offsets[b + t]), int(position_offsets[b + t + 1]))
            for p in sp:
                cheap.append((int(ngram_indices[p]), [int(ngram_indices[r]) for r in tp]))
                for r in tp:
                    full.append((int(ngram_indices[p]), int(ngram_indices[r])))
        cheap.sort(key=lambda e: e[0])            # what align_source does
        expanded = [(i, r) for i, rs in cheap for r in rs]
        full.sort()                               # what a cross-product sort gives
        checked += 1
        if expanded != full:
            mismatched += 1
    check(f"expansion order equals a full cross-product sort ({checked} pairs)",
          mismatched == 0, f"{mismatched} pair(s) differed")

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'all checks PASS'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
