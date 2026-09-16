#!/usr/bin/env python3
"""Convert an ngrams/ directory of .json files to the binary index, for validation.

    convert_ngrams.py SRC_DIR DST_DIR [--workers N]

DST_DIR may not be SRC_DIR: this never converts in place. Generation writes binary
directly, so this exists to check the reader against corpora built before it did.
"""
import argparse
import sys

from textpair.sequence_alignment.ngram_binary import convert_directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_dir", help="an ngrams/ directory of .json files")
    parser.add_argument("target_dir", help="where the .bin files go; never source_dir")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        count, json_bytes, binary_bytes = convert_directory(
            args.source_dir, args.target_dir, args.workers)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    print(f"{count} files: {json_bytes / 2 ** 30:.3f} GiB json -> "
          f"{binary_bytes / 2 ** 30:.3f} GiB binary "
          f"({binary_bytes / json_bytes:.3f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
