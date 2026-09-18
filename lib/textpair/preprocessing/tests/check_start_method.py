#!/usr/bin/env python3
"""Checks which multiprocessing start methods work after an earlier pool.

    check_start_method.py [--workers N] [--timeout SECONDS] [--texts N]

Why this exists: generate_ngrams used to special-case macOS. multiprocess.Pool
defaults to fork() there, and forking again right after the preceding PhiloLogic
parse stage's own pool has torn down deadlocks (bpo-33725), so that stage kept
the preprocessor serial and fanned out with threads instead. The rewrite picks
its start method in preprocessing.worker_start_method -- spawn on Darwin, fork
elsewhere -- which should make the special case unnecessary.

This confirms that on a machine where it actually mattered. It builds its own
PhiloLogic database, tears down a pool to reproduce the preceding stage, then
runs the preprocessor under each start method in a subprocess with a timeout, so
a deadlock is reported rather than hanging the run.

Run it and paste the whole report.
"""
import argparse
import json
import os
import platform
import subprocess
import sqlite3
import sys
import tempfile
import time

import lz4.frame

TOMS_COLUMNS = ("philo_type", "philo_id", "philo_name", "filename", "author",
                "title", "year", "word_count", "next", "prev", "head")
WORDS_PER_TEXT = 400

# Alphabetic: the default configuration drops any token containing a digit, so
# generated names like "mot0" would leave nothing to build n-grams from.
VOCABULARY = (
    "maison", "cheval", "royaume", "assemblee", "peuple", "nation", "liberte",
    "vertu", "raison", "nature", "esprit", "societe", "justice", "pouvoir",
    "citoyen", "histoire", "lumiere", "verite", "estoit", "avoit", "sceptre",
    "volonte", "contrat", "origine", "inegalite", "discours", "confession",
)


def build_corpus(root, texts):
    """A PhiloLogic database of `texts` documents, one words file each."""
    data = os.path.join(root, "data")
    os.makedirs(os.path.join(data, "words_and_philo_ids"), exist_ok=True)
    os.makedirs(os.path.join(data, "TEXT"), exist_ok=True)
    connection = sqlite3.connect(os.path.join(data, "toms.db"))
    connection.execute(f"CREATE TABLE toms ({', '.join(TOMS_COLUMNS)})")
    paths = []
    for document in range(1, texts + 1):
        path = os.path.join(data, "words_and_philo_ids", f"{document}.lz4")
        with lz4.frame.open(path, mode="wb") as handle:
            for position in range(WORDS_PER_TEXT):
                word = {
                    "token": VOCABULARY[(position * 7 + document) % len(VOCABULARY)],
                    "position": f"{document} 1 0 0 {position // 40 + 1} {position // 8 + 1} {position + 1}",
                    "start_byte": position * 8,
                    "end_byte": position * 8 + 6,
                    "philo_type": "word",
                }
                handle.write((json.dumps(word) + "\n").encode("utf8"))
        connection.execute(
            f"INSERT INTO toms ({', '.join(TOMS_COLUMNS)}) VALUES ({','.join('?' * len(TOMS_COLUMNS))})",
            ("doc", f"{document} 0 0 0 0 0 0", f"doc{document}", f"{document}.xml",
             "Author", f"Title {document}", "1789", str(WORDS_PER_TEXT), "", "", ""),
        )
        paths.append(path)
    connection.commit()
    connection.close()
    return paths


def _child_noop(value):
    return value * 2


CHILD = r'''
import json, os, sys, time
os.environ["TEXTPAIR_START_METHOD"] = sys.argv[1]
paths = json.loads(sys.argv[2])
workers = int(sys.argv[3])

# Reproduce the stage that runs before preprocessing: a pool created and torn
# down. On macOS, forking again after this is what deadlocks.
import multiprocessing
with multiprocessing.Pool(min(workers, 4)) as pool:
    pool.map(abs, range(64))

from textpair.preprocessing import PreProcessor, worker_start_method
started = time.perf_counter()
preproc = PreProcessor(workers=workers, language="french", stemmer=True, modernize=True,
                       ngrams=3, text_object_type="doc")
total = 0
for text_object in preproc.process_texts(paths):
    total += len(text_object)
print(json.dumps({"method": worker_start_method(), "ngrams": total,
                  "seconds": round(time.perf_counter() - started, 2)}))
'''


def run_case(method, paths, workers, timeout):
    """Run the preprocessor under one start method, in a subprocess."""
    started = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, "-c", CHILD, method, json.dumps(paths), str(workers)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"status": "HUNG", "detail": f"no output within {timeout}s"}
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        return {"status": "ERROR", "detail": detail[-1] if detail else "no output"}
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"status": "ERROR", "detail": result.stdout.strip()[:200]}
    payload["status"] = "ok"
    payload["wall"] = round(elapsed, 2)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 2))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--texts", type=int, default=24)
    args = parser.parse_args()

    print("=" * 68)
    print("TextPAIR preprocessing: start-method check")
    print("=" * 68)
    print(f"platform         {platform.platform()}")
    print(f"machine          {platform.machine()}")
    print(f"python           {sys.version.split()[0]}")
    print(f"cpu_count        {os.cpu_count()}")
    print(f"workers          {args.workers}")

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
    from textpair.preprocessing import worker_start_method
    from textpair.sequence_alignment.ngram_index import sort_program, MAX_BATCH
    import multiprocessing

    print(f"default method   {worker_start_method()}")
    print(f"available        {', '.join(multiprocessing.get_all_start_methods())}")
    program, supports_listing = sort_program()
    route = ("--files0-from, one merge" if supports_listing
             else f"file arguments, batches of at most {MAX_BATCH}")
    print(f"sort for index   {program} ({route})")
    print()

    with tempfile.TemporaryDirectory() as root:
        paths = build_corpus(root, args.texts)
        print(f"corpus           {len(paths)} texts x {WORDS_PER_TEXT} words in {root}")
        print()
        print(f"{'method':12s} {'status':8s} {'ngrams':>10s} {'inner':>8s} {'wall':>8s}")
        print("-" * 52)
        results = {}
        for method in multiprocessing.get_all_start_methods():
            outcome = run_case(method, paths, args.workers, args.timeout)
            results[method] = outcome
            print(f"{method:12s} {outcome['status']:8s} "
                  f"{outcome.get('ngrams', ''):>10} "
                  f"{outcome.get('seconds', ''):>8} "
                  f"{outcome.get('wall', ''):>8}"
                  + (f"   {outcome['detail']}" if outcome.get("detail") else ""))

    print()
    counts = {method: outcome.get("ngrams") for method, outcome in results.items()
              if outcome["status"] == "ok"}
    if len(set(counts.values())) > 1:
        print("MISMATCH: start methods disagreed on the ngram count", counts)
        return 1
    working = sorted(counts)
    print(f"start methods that completed: {', '.join(working) or 'none'}")
    default = worker_start_method()
    if default in counts:
        print(f"the default for this platform ({default}) works")
        return 0
    print(f"the default for this platform ({default}) did NOT work -- report this")
    return 1


if __name__ == "__main__":
    sys.exit(main())
