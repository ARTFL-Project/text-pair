#!/usr/bin/env python3
"""Run the preprocessing checks and report one line each.

    run_tests.py [--only NAME]...
    run_tests.py --philo-db PATH [--files N]   # also run the parity check

With no database, the four property checks run. They build whatever they need,
so they depend on nothing but the package, and finish in seconds.

Given a PhiloLogic database, check_parity.py also runs: it compares this package
against the text_preprocessing library token for token across many
configurations. That needs the old library installed, so it is a migration tool
rather than a permanent check -- once text_preprocessing is gone from the
environment, the property checks are what remain.

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
    ("config", "test_config.py"),
    ("reader", "test_reader.py"),
    ("worker_imports", "test_worker_imports.py"),
)


def run(name, script, arguments):
    started = time.perf_counter()
    result = subprocess.run([sys.executable, os.path.join(HERE, script), *arguments],
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
    parser.add_argument("--philo-db", default="")
    parser.add_argument("--files", type=int, default=4)
    parser.add_argument("--fixtures", default=os.environ.get("PARITY_FIXTURES", ""))
    args = parser.parse_args()

    selected = [check for check in CHECKS if not args.only or check[0] in args.only]
    failures = 0
    print("preprocessing checks:")
    for name, script in selected:
        failures += bool(run(name, script, ()))

    if args.philo_db and (not args.only or "parity" in args.only):
        environment = dict(os.environ)
        if args.fixtures:
            environment["PARITY_FIXTURES"] = args.fixtures
        print(f"parity against text_preprocessing ({args.files} files):")
        started = time.perf_counter()
        result = subprocess.run(
            [sys.executable, os.path.join(HERE, "check_parity.py"), args.philo_db, str(args.files)],
            capture_output=True, text=True, env=environment,
        )
        elapsed = time.perf_counter() - started
        status = "ok  " if result.returncode == 0 else "FAIL"
        print(f"  {status} parity         {elapsed:5.1f}s")
        if result.returncode != 0:
            for line in (result.stdout + result.stderr).splitlines():
                print(f"        {line}")
        failures += bool(result.returncode)

    print("all checks pass" if not failures else f"{failures} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
