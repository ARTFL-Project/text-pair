"""Command line wrapper around align().

python -m textpair.sequence_alignment.aligner --source_files=DIR \
    --source_metadata=FILE --output_path=DIR [--threads=N] [...]
"""
import argparse
import sys

from .runner import DEFAULTS, align


def main(argv=None):
    parser = argparse.ArgumentParser(prog="textpair-aligner", description=__doc__)
    parser.add_argument("--output_path", default="./output")
    parser.add_argument("--source_files", default="")
    parser.add_argument("--target_files", default="")
    parser.add_argument("--source_metadata", default="")
    parser.add_argument("--target_metadata", default="")
    parser.add_argument("--output_workers", type=int, default=0)
    parser.add_argument("--lz4_level", type=int, default=3)
    for name, value in DEFAULTS.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{name}", default=str(value).lower())
        else:
            parser.add_argument(f"--{name}", type=type(value), default=value)
    args = vars(parser.parse_args(argv))
    fixed = {key: args.pop(key) for key in ("output_path", "source_files", "target_files",
                                            "source_metadata", "target_metadata",
                                            "output_workers", "lz4_level")}
    try:
        align(fixed["source_files"], fixed["source_metadata"], fixed["output_path"],
              target_files=fixed["target_files"], target_metadata=fixed["target_metadata"],
              output_workers=fixed["output_workers"], lz4_level=fixed["lz4_level"], **args)
    except (ValueError, TypeError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
