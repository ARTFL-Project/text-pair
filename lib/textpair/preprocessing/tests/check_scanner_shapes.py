#!/usr/bin/env python3
"""Checks the word-JSON scanner against a real parser, on every corpus it can find.

    check_scanner_shapes.py [--corpora N] [--lines N] [--philo-root DIR]...

philo_scan reads fields by name out of the raw buffer rather than parsing each
line. That is only safe if it agrees with a parser on every shape of word JSON
in the wild, and those differ: five to nine fields, and not in the same order.
This decodes the same lines both ways and compares them field by field.
"""
import argparse
import glob
import json
import os
import random
import sys

import lz4.frame
import numpy as np

from textpair.preprocessing import philo_scan

DEFAULT_ROOTS = ("/var/www/html/philologic5", "/disk2/webspace/philologic5")


def scan(blob, level=1):
    buffer = np.frombuffer(blob, dtype=np.uint8)
    breaks = np.flatnonzero(buffer == philo_scan.NEWLINE)
    starts = np.empty(breaks.size + 1, dtype=np.int64)
    starts[0] = 0
    starts[1:] = breaks + 1
    if starts[-1] >= buffer.size:
        starts = starts[:-1]
    ends = np.empty(starts.size + 1, dtype=np.int64)
    ends[:-1] = starts
    ends[-1] = buffer.size
    rows = starts.size
    columns = {name: np.empty(rows, dtype=np.int64)
               for name in ("token_lo", "token_hi", "start_byte", "end_byte",
                            "position_lo", "position_hi", "object_hi")}
    flags = {name: np.empty(rows, dtype=np.uint8)
             for name in ("token_escaped", "is_punct", "new_object")}
    philo_scan.scan_lines(
        buffer, ends, level, True, columns["token_lo"], columns["token_hi"],
        flags["token_escaped"], columns["start_byte"], columns["end_byte"],
        columns["position_lo"], columns["position_hi"], columns["object_hi"],
        flags["is_punct"], flags["new_object"])
    return buffer, columns, flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", type=int, default=0, help="0 means every corpus found")
    parser.add_argument("--lines", type=int, default=3000, help="lines compared per corpus")
    parser.add_argument("--philo-root", action="append", default=[])
    args = parser.parse_args()

    roots = args.philo_root or list(DEFAULT_ROOTS)
    directories = [d for root in roots
                   for d in sorted(glob.glob(os.path.join(root, "*/data/words_and_philo_ids")))]
    random.seed(11)
    if args.corpora:
        directories = random.sample(directories, min(args.corpora, len(directories)))

    failures = 0
    checked = 0
    shapes = {}
    for directory in directories:
        corpus = os.path.basename(os.path.dirname(os.path.dirname(directory)))
        files = sorted(glob.glob(os.path.join(directory, "*")))
        if not files:
            continue
        path = files[len(files) // 2]
        try:
            with open(path, "rb") as handle:
                raw = handle.read()
            blob = lz4.frame.decompress(raw) if path.endswith(".lz4") else raw
        except Exception as error:  # noqa: BLE001
            print(f"  SKIP {corpus}: {type(error).__name__}")
            continue
        lines = [line for line in blob.splitlines() if line.strip()][: args.lines]
        if not lines:
            continue
        blob = b"\n".join(lines) + b"\n"
        buffer, columns, flags = scan(blob)

        mismatches = []
        for row, line in enumerate(lines):
            want = json.loads(line)
            shapes.setdefault(tuple(want.keys()), set()).add(corpus)
            lo, hi = columns["token_lo"][row], columns["token_hi"][row]
            if lo < 0:
                mismatches.append((row, "token not found", line[:80]))
                continue
            raw_token = bytes(buffer[lo:hi])
            got_token = (json.loads(b'"' + raw_token + b'"')
                         if flags["token_escaped"][row] else raw_token.decode("utf8"))
            if got_token != want["token"]:
                mismatches.append((row, f"token {got_token!r} != {want['token']!r}", b""))
            for field in ("start_byte", "end_byte"):
                if int(columns[field][row]) != int(want[field]):
                    mismatches.append((row, f"{field} {columns[field][row]} != {want[field]}", b""))
            doc = bytes(buffer[columns["position_lo"][row]:columns["object_hi"][row]]).decode()
            if doc != want["position"].split(" ")[0]:
                mismatches.append((row, f"object id {doc!r} != {want['position'].split(' ')[0]!r}", b""))
            full = bytes(buffer[columns["position_lo"][row]:columns["position_hi"][row]]).decode()
            if full != want["position"]:
                mismatches.append((row, f"position {full!r} != {want['position']!r}", b""))
            if bool(flags["is_punct"][row]) != (want.get("philo_type") == "punct"):
                mismatches.append((row, "philo_type punct flag", b""))
        checked += len(lines)
        if mismatches:
            failures += 1
            print(f"  FAIL {corpus}: {len(mismatches)} mismatch(es) over {len(lines):,} lines")
            for row, why, sample in mismatches[:3]:
                print(f"        line {row}: {why} {sample.decode('utf8', 'replace')[:80]}")
        else:
            print(f"  ok   {corpus:28s} {len(lines):>6,} lines, {len(want)} fields")

    print()
    print(f"  {checked:,} lines compared across {len(directories)} corpora, "
          f"{len(shapes)} distinct key shapes")
    for shape in sorted(shapes, key=len):
        print(f"    {len(shape)} fields: {', '.join(shape)}")
    print("all shapes agree with the parser" if not failures else f"{failures} corpus/corpora differ")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
