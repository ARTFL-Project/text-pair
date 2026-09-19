#!/usr/bin/env python3
"""Run the preprocessing checks and report one line each.

    run_tests.py [--only NAME]...

The checks build whatever they need, so they depend on nothing but the package,
and finish in seconds.

Exits non-zero if any check fails, and prints its output.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

CHECKS = (
    ("normalization", "test_normalization.py"),
    ("ngrams", "test_ngrams.py"),
    ("ngram_kernels", "test_ngram_kernels.py"),
    ("config", "test_config.py"),
    ("reader", "test_reader.py"),
    ("worker_imports", "test_worker_imports.py"),
)


def run(name, script):
    started = time.perf_counter()
    result = subprocess.run([sys.executable, os.path.join(HERE, script)],
                            capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    status = "ok  " if result.returncode == 0 else "FAIL"
    print(f"  {status} {name:14s} {elapsed:5.1f}s")
    if result.returncode != 0:
        for line in (result.stdout + result.stderr).splitlines():
            print(f"        {line}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()

    selected = [check for check in CHECKS if not args.only or check[0] in args.only]
    failures = 0
    print("preprocessing checks:")
    for name, script in selected:
        failures += bool(run(name, script))

    print("all checks pass" if not failures else f"{failures} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
