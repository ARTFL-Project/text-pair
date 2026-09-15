#!/usr/bin/env python3
"""Run the Go and Python sequence aligners over a corpus and compare every artifact.

    compare_aligners.py --source-files DIR --source-metadata FILE [--target-files DIR
        --target-metadata FILE] [--threads N] [--workdir DIR] [--go-results DIR]
        [--run-cwd DIR] [--param name=value]...
    compare_aligners.py --fixtures [--threads N]          # the synthetic corpora here

Compared: chunk file names, every JSON record as a sorted multiset (and per chunk, byte
for byte), count.txt, duplicate_files.csv as a sorted multiset, and alignment_config.ini
apart from outputPath and the stale prebuilt binary's "<invalid reflect.Value>" lines.
Exits non-zero on any difference and prints the first record that differs. --go-results
reuses an existing Go run instead of running the binary again.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import lz4.frame
import orjson

FIXTURES = ("no_byte_range", "non_string_meta", "missing_text", "no_metadata")
GO_BINARY = os.environ.get("TEXTPAIR_GO_ALIGNER", "compareNgrams")
CANON = orjson.OPT_SORT_KEYS


def digest(line):
    return hashlib.sha1(orjson.dumps(orjson.loads(line), option=CANON)).digest()


def read_chunk(path):
    if not path:
        return b"", []
    with lz4.frame.open(path, "rb") as handle:
        raw = handle.read()
    return raw, [digest(line) for line in raw.splitlines() if line.strip()]


def compare_chunk(job):
    name, go_path, py_path = job
    go_raw, go_digests = read_chunk(go_path)
    py_raw, py_digests = read_chunk(py_path)
    return name, go_raw == py_raw, go_digests, py_digests


def chunks(results_dir):
    """Chunk files, wherever the aligner left them (a batched run keeps result_chunks)."""
    found = {}
    for path in glob.glob(os.path.join(results_dir, "result_batches", "**", "*.lz4"),
                          recursive=True):
        found[os.path.basename(path)] = path
    return found


def first_difference(go_chunks, py_chunks, only_go, only_py):
    """The first record, in chunk-name order, whose digest is unique to one side."""
    out = {}
    for label, source, wanted in (("go", go_chunks, only_go), ("python", py_chunks, only_py)):
        for name in sorted(source):
            with lz4.frame.open(source[name], "rb") as handle:
                for line in handle.read().splitlines():
                    if line.strip() and digest(line) in wanted:
                        out[label] = {"chunk": name, "record": orjson.loads(line)}
                        break
            if label in out:
                break
    if "go" in out and "python" in out:
        go_record, py_record = out["go"]["record"], out["python"]["record"]
        out["differing_fields"] = sorted(k for k in set(go_record) | set(py_record)
                                         if go_record.get(k) != py_record.get(k))
    return out


def read_text(path):
    with open(path, encoding="utf8") as handle:
        return handle.read()


def compare(go_dir, py_dir, workers):
    go_chunks, py_chunks = chunks(go_dir), chunks(py_dir)
    result = {
        "n_chunks_go": len(go_chunks),
        "n_chunks_python": len(py_chunks),
        "chunk_names_identical": sorted(go_chunks) == sorted(py_chunks),
    }
    if not result["chunk_names_identical"]:
        result["chunks_only_go"] = sorted(set(go_chunks) - set(py_chunks))[:20]
        result["chunks_only_python"] = sorted(set(py_chunks) - set(go_chunks))[:20]

    jobs = [(name, go_chunks.get(name), py_chunks.get(name))
            for name in sorted(set(go_chunks) | set(py_chunks))]
    go_counts, py_counts = Counter(), Counter()
    identical_bytes = 0
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = pool.map(compare_chunk, jobs, chunksize=16)
            for _name, same_bytes, go_digests, py_digests in results:
                identical_bytes += same_bytes
                go_counts.update(go_digests)
                py_counts.update(py_digests)
    else:
        for job in jobs:
            _name, same_bytes, go_digests, py_digests = compare_chunk(job)
            identical_bytes += same_bytes
            go_counts.update(go_digests)
            py_counts.update(py_digests)
    result["n_records_go"] = sum(go_counts.values())
    result["n_records_python"] = sum(py_counts.values())
    result["records_identical_multiset"] = go_counts == py_counts
    result["chunks_byte_identical"] = f"{identical_bytes}/{len(jobs)}"
    if not result["records_identical_multiset"]:
        only_go = set((go_counts - py_counts).elements())
        only_py = set((py_counts - go_counts).elements())
        result["records_only_go"] = len(only_go)
        result["records_only_python"] = len(only_py)
        result["first_difference"] = first_difference(go_chunks, py_chunks, only_go, only_py)

    go_count = read_text(os.path.join(go_dir, "count.txt"))
    py_count = read_text(os.path.join(py_dir, "count.txt"))
    result["count_txt"] = [go_count, py_count]
    result["count_txt_identical"] = go_count == py_count

    go_dups = read_text(os.path.join(go_dir, "duplicate_files.csv")).splitlines()
    py_dups = read_text(os.path.join(py_dir, "duplicate_files.csv")).splitlines()
    result["n_duplicates_go"] = max(0, len(go_dups) - 1)
    result["n_duplicates_python"] = max(0, len(py_dups) - 1)
    result["duplicates_identical_multiset"] = (go_dups[:1] == py_dups[:1]
                                               and sorted(go_dups[1:]) == sorted(py_dups[1:]))
    if not result["duplicates_identical_multiset"]:
        result["duplicates_only_go"] = sorted(set(go_dups) - set(py_dups))[:5]
        result["duplicates_only_python"] = sorted(set(py_dups) - set(go_dups))[:5]

    # The shipped prebuilt binary predates main.go:606's fix and still lists two removed
    # parameters as "<invalid reflect.Value>"; those lines are ignored here.
    def config(path):
        return [line for line in read_text(path).splitlines()
                if not line.startswith("outputPath:")
                and not line.endswith("<invalid reflect.Value>")]

    go_config = config(os.path.join(go_dir, "alignment_config.ini"))
    py_config = config(os.path.join(py_dir, "alignment_config.ini"))
    result["config_identical"] = go_config == py_config
    if not result["config_identical"]:
        result["config_diff"] = [[a, b] for a, b in zip(go_config, py_config) if a != b]
    result["PASS"] = all(result[key] for key in ("chunk_names_identical",
                                                 "records_identical_multiset",
                                                 "count_txt_identical",
                                                 "duplicates_identical_multiset",
                                                 "config_identical"))
    return result


def flags(args, params, output_path):
    out = [f"--output_path={output_path}", f"--threads={args.threads}",
           f"--source_files={args.source_files}", f"--source_metadata={args.source_metadata}"]
    if args.target_files:
        out.append(f"--target_files={args.target_files}")
    if args.target_metadata:
        out.append(f"--target_metadata={args.target_metadata}")
    return out + [f"--{key}={value}" for key, value in params.items()]


def run_pair(args, params):
    workdir = os.path.abspath(args.workdir)
    os.makedirs(workdir, exist_ok=True)
    py_dir = os.path.join(workdir, "python")
    if not args.python_results:
        shutil.rmtree(py_dir, ignore_errors=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    env.setdefault("NUMBA_CACHE_DIR", os.path.join(workdir, "numba_cache"))
    if args.python_results:
        py_dir = os.path.abspath(args.python_results)
    if args.go_results:
        go_dir = os.path.abspath(args.go_results)
    else:
        go_dir = os.path.join(workdir, "go")
        shutil.rmtree(go_dir, ignore_errors=True)
        print(f"--- {GO_BINARY} -> {go_dir}", flush=True)
        subprocess.run([GO_BINARY] + flags(args, params, go_dir), cwd=args.run_cwd,
                       check=True, stdout=subprocess.DEVNULL)
    if not args.python_results:
        print(f"--- python aligner -> {py_dir}", flush=True)
        subprocess.run([sys.executable, "-m", "textpair.sequence_alignment.aligner"]
                       + flags(args, params, py_dir), cwd=args.run_cwd, check=True, env=env,
                       stdout=subprocess.DEVNULL)
    return go_dir, py_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-files", default="")
    parser.add_argument("--source-metadata", default="")
    parser.add_argument("--target-files", default="")
    parser.add_argument("--target-metadata", default="")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workdir", default="./aligner_comparison")
    parser.add_argument("--go-results", default="", help="reuse an existing Go output tree")
    parser.add_argument("--python-results", default="",
                        help="reuse an existing Python output tree")
    parser.add_argument("--run-cwd", default=None, help="cwd for both aligners")
    parser.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                        help="matching parameter passed to both aligners")
    parser.add_argument("--compare-workers", type=int, default=8)
    parser.add_argument("--fixtures", action="store_true",
                        help="run every synthetic fixture corpus in fixtures/")
    args = parser.parse_args(argv)
    params = dict(item.split("=", 1) for item in args.param)

    if args.fixtures:
        here = os.path.dirname(os.path.abspath(__file__))
        failures = []
        for name in FIXTURES:
            print(f"\n=== fixture {name} ===", flush=True)
            fixture = argparse.Namespace(**vars(args))
            fixture.source_files = "ngrams"
            fixture.source_metadata = "metadata/metadata.json"
            fixture.target_files = fixture.target_metadata = ""
            fixture.run_cwd = os.path.join(here, "fixtures", name)
            fixture.workdir = os.path.join(os.path.abspath(args.workdir), name)
            fixture.go_results = fixture.python_results = ""
            result = compare(*run_pair(fixture, params), args.compare_workers)
            print(json.dumps(result, indent=1, ensure_ascii=False))
            if not result["PASS"]:
                failures.append(name)
        print("\nFAILED fixtures:" if failures else "\nAll fixtures PASS", *failures)
        return 1 if failures else 0

    if not args.source_files or not args.source_metadata:
        parser.error("--source-files and --source-metadata are required")
    result = compare(*run_pair(args, params), args.compare_workers)
    print(json.dumps(result, indent=1, ensure_ascii=False))
    return 0 if result["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
