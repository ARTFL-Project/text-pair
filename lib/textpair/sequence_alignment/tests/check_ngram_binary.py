#!/usr/bin/env python3
"""Round-trip check for the binary ngram index.

    check_ngram_binary.py NGRAMS_DIR METADATA_JSON [--binary-dir DIR] [--threads N]
        [--sort-by FIELD] [--keep]

Converts a JSON `ngrams/` directory to binary, loads both with ngram_loader.load_corpus over
the same document sequence, and compares all six corpus arrays element for element.
That covers the key ordering the writer has to reproduce, since the loader sorts the
JSON side itself. Exits non-zero on any difference.

The document sequence comes from the JSON directory alone; `doc_order_identical`
reports separately whether the binary directory orders the same documents the same way.
"""
import argparse
import os
import shutil
import sys
import tempfile

import numpy as np

from textpair.sequence_alignment import ngram_binary
from textpair.sequence_alignment.aligner import ngram_loader
from textpair.sequence_alignment.aligner.documents import get_files
from textpair.sequence_alignment.aligner.documents import load_metadata

ARRAYS = ("key_offsets", "ngram_keys", "position_offsets", "ngram_indices",
          "start_bytes", "end_bytes")


def compare(ngrams_dir, metadata_path, binary_dir, threads, sort_by):
    count, json_bytes, binary_bytes = ngram_binary.convert_directory(
        ngrams_dir, binary_dir, threads)
    metadata = load_metadata(metadata_path)
    json_docs = get_files(ngrams_dir, metadata, sort_by)
    json_paths = [path for _, path in json_docs]
    binary_paths = [os.path.join(binary_dir, doc + ".bin") for doc, _ in json_docs]
    result = {
        "n_documents": count,
        "json_bytes": json_bytes,
        "binary_bytes": binary_bytes,
        "binary_over_json": round(binary_bytes / json_bytes, 4) if json_bytes else None,
        "doc_order_identical": [doc for doc, _ in json_docs]
                               == [doc for doc, _ in get_files(binary_dir, metadata, sort_by)],
    }
    ngram_loader.warmup()
    from_json = ngram_loader.load_corpus(json_paths, threads)
    from_binary = ngram_loader.load_corpus(binary_paths, threads)
    differences = []
    for name, left, right in zip(ARRAYS, from_json, from_binary):
        if left.shape != right.shape:
            differences.append(f"{name}: shape {left.shape} vs {right.shape}")
        elif not np.array_equal(left, right):
            first = int(np.flatnonzero(left != right)[0])
            differences.append(f"{name}: differs at {first}, {left[first]} vs {right[first]}")
        result[f"{name}_len"] = int(left.shape[0])
    result["arrays_identical"] = not differences
    if differences:
        result["differences"] = differences
    result["PASS"] = result["arrays_identical"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ngrams_dir")
    parser.add_argument("metadata")
    parser.add_argument("--binary-dir", default="", help="default: a temporary directory")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--sort-by", default="year")
    parser.add_argument("--keep", action="store_true", help="keep the converted directory")
    args = parser.parse_args(argv)

    binary_dir = args.binary_dir or tempfile.mkdtemp(prefix="ngram_binary_roundtrip_")
    try:
        result = compare(args.ngrams_dir, args.metadata, binary_dir, args.threads,
                         args.sort_by)
    finally:
        if not args.keep and not args.binary_dir:
            shutil.rmtree(binary_dir, ignore_errors=True)
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0 if result["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
