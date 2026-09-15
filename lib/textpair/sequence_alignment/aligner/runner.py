"""Driver for the Python sequence aligner: the `align()` entry point.

`align()` takes the same parameters as the compareNgrams binary's command line flags and
writes the same artifacts to <output_path>: result_batches/ chunk files, count.txt,
duplicate_files.csv and alignment_config.ini.
"""
import gc
import math
import os
import shutil
import subprocess
import sys
from shlex import quote

import numpy as np

from . import loader, output, pipeline
from .gosort import get_files
from .gotext import load_metadata

# Go's flag defaults (main.go:132-154). sa_config.ini overrides some of them.
DEFAULTS = dict(
    threads=4,
    sort_by="year",
    source_batch=1,
    target_batch=1,
    matching_window_size=30,
    max_gap=15,
    flex_gap=False,
    minimum_matching_ngrams=4,
    minimum_matching_ngrams_in_window=4,
    minimum_matching_ngrams_in_docs=4,
    context_size=300,
    duplicate_threshold=80,
    merge_passages_on_byte_distance=True,
    merge_passages_on_ngram_distance=True,
    passage_distance_multiplier=0.5,
    debug=False,
    ngram_index="",
)

_INT_PARAMS = ("threads", "source_batch", "target_batch", "matching_window_size", "max_gap",
               "minimum_matching_ngrams", "minimum_matching_ngrams_in_window",
               "minimum_matching_ngrams_in_docs", "context_size")
_FLOAT_PARAMS = ("duplicate_threshold", "passage_distance_multiplier")
_BOOL_PARAMS = ("flex_gap", "merge_passages_on_byte_distance",
                "merge_passages_on_ngram_distance", "debug")
_TYPED_PARAMS = frozenset(_INT_PARAMS + _FLOAT_PARAMS + _BOOL_PARAMS)


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "t", "true", "yes", "y", "on")
    return bool(value)


def _normalize(params):
    """Coerce config-file strings to the types the kernels expect.

    An empty typed parameter means "use the default", but an empty sort_by is kept: it is
    how the caller asks for Go's unsorted, document-ID order (main.go:285).
    """
    clean = dict(DEFAULTS)
    clean.update({k: v for k, v in params.items()
                  if v is not None and not (v == "" and k in _TYPED_PARAMS)})
    for key in _INT_PARAMS:
        clean[key] = int(clean[key])
    for key in _FLOAT_PARAMS:
        clean[key] = float(clean[key])
    for key in _BOOL_PARAMS:
        clean[key] = _as_bool(clean[key])
    return clean


def _config_values(params, output_path):
    """The alignment_config.ini payload, with Go's key spellings (main.go:609-627)."""
    return {
        "matchingWindowSize": params["matching_window_size"],
        "maxGap": params["max_gap"],
        "flexGap": params["flex_gap"],
        "minimumMatchingNgrams": params["minimum_matching_ngrams"],
        "minimumMatchingNgramsInWindow": params["minimum_matching_ngrams_in_window"],
        "minimumMatchingNgramsInDocs": params["minimum_matching_ngrams_in_docs"],
        "contextSize": params["context_size"],
        "mergeOnByteDistance": params["merge_passages_on_byte_distance"],
        "mergeOnNgramDistance": params["merge_passages_on_ngram_distance"],
        "passageDistanceMultiplier": params["passage_distance_multiplier"],
        "duplicateThreshold": float(params["duplicate_threshold"]),
        "sourceBatch": params["source_batch"],
        "targetBatch": params["target_batch"],
        "outputPath": output_path,
        "numThreads": params["threads"],
        "sortingField": params["sort_by"],
        "debug": params["debug"],
    }


def _batches(files, count):
    """main.go:985-997. Consecutive slices of ceil(len/count) documents."""
    if not files:
        return []
    size = int(math.ceil(len(files) / count))
    return [files[i:i + size] for i in range(0, len(files), size)]


def chunk_ranges(position, n_targets, threads, same_array):
    """Target ranges of the chunk files Go writes for one source document.

    One file per goroutine, and the goroutine count itself depends on how many targets
    are left (main.go:437-481), so the file names only match Go's if this split does.
    """
    start = position + 1 if same_array else 0
    if start >= n_targets:
        return []
    local_length = n_targets - start
    needed = threads
    if threads > 1:
        per_thread = local_length // needed
        while per_thread < 10:
            needed //= 2
            if needed < 2:
                needed = 1
                break
            per_thread = local_length // needed
        increment = local_length // needed
    else:
        increment = local_length - start        # Go's expression; the clamp below hides it
    ranges = []
    end = start + increment
    for i in range(needed):
        end = min(end, n_targets)
        if i == needed - 1:
            end = n_targets
        ranges.append((start, end))
        start = end
        end += increment
    return ranges


def _jobs(rows, duplicates, docs, n_targets, target_base, threads, same_array):
    """Group one match task's rows into chunk files, as (slot, name, lo, hi).

    A range with no alignment but a duplicate still gets a file: Go appends an empty
    entry to localAlignments for a duplicate target (main.go:504).
    """
    row_range = {}
    start = 0
    n_rows = rows.shape[0]
    while start < n_rows:
        slot = int(rows[start, 0])
        end = start + 1
        while end < n_rows and rows[end, 0] == slot:
            end += 1
        row_range[slot] = (start, end)
        start = end
    dup_targets = {}
    for slot, target in duplicates:
        dup_targets.setdefault(int(slot), []).append(int(target))
    jobs = []
    for slot in sorted(set(row_range) | set(dup_targets)):
        lo, hi = row_range.get(slot, (0, 0))
        targets = rows[lo:hi, 1]
        dups = dup_targets.get(slot, ())
        for first, last in chunk_ranges(slot, n_targets, threads, same_array):
            low = target_base + first
            high = target_base + last
            begin = lo + int(np.searchsorted(targets, low, "left"))
            stop = lo + int(np.searchsorted(targets, high, "left"))
            if stop > begin or any(low <= t < high for t in dups):
                jobs.append((slot, f"{docs[slot]}-{docs[low]}-{docs[high - 1]}.lz4",
                             begin, stop))
    return jobs


def _run_combination(params, source_docs, target_docs, source_metadata, target_metadata,
                     same_array, chunk_dir, progress):
    """Compare one (source batch, target batch) pair. Returns (count, duplicate rows)."""
    threads = params["threads"]
    if same_array:
        docs = [doc for doc, _ in source_docs]
        paths = [path for _, path in source_docs]
        metas = [source_metadata.get(doc, {}) for doc in docs]
        same_doc = np.empty(0, np.int32)
        target_base = 0
        n_targets = n_sources = len(docs)
    else:
        docs = [doc for doc, _ in source_docs] + [doc for doc, _ in target_docs]
        paths = [path for _, path in source_docs] + [path for _, path in target_docs]
        metas = ([source_metadata.get(doc, {}) for doc, _ in source_docs]
                 + [target_metadata.get(doc, {}) for doc, _ in target_docs])
        target_base = len(source_docs)
        n_targets = len(target_docs)
        target_slot = {doc: target_base + i for i, (doc, _) in enumerate(target_docs)}
        same_doc = np.array([target_slot.get(doc, -1) for doc, _ in source_docs], np.int32)
    n_sources = len(source_docs)

    # Fork the writers before the ngram arrays exist: they inherit the cleaned metadata
    # and only a few tens of MB of page tables get copied.
    pool = output.OutputPool(params["output_workers"], docs, metas, chunk_dir,
                             params["context_size"], level=params["lz4_level"])
    count = 0
    duplicate_rows = []
    try:
        key_off, keys_all, off_all, idx_all, sb_all, eb_all = \
            loader.load_corpus(paths, threads)
        post_u, post_slot, bstart, max_bucket = pipeline.build_postings(keys_all, threads)
        em_tgt, em_sslot, em_tslot, src_off, per_source = pipeline.build_emissions(
            post_u, post_slot, bstart, max_bucket, key_off, n_sources, same_doc, threads)
        del post_u, post_slot, keys_all
        gc.collect()

        def on_result(rows, dups, percents, _stats):
            nonlocal count
            count += rows.shape[0]
            if rows.shape[0] or dups.shape[0]:
                pool.submit(np.ascontiguousarray(rows),
                            _jobs(rows, dups, docs, n_targets, target_base, threads,
                                  same_array))
            if dups.shape[0]:
                for i in np.lexsort((dups[:, 1], dups[:, 0])):
                    duplicate_rows.append(output.duplicate_row(metas[int(dups[i, 0])],
                                                               metas[int(dups[i, 1])],
                                                               float(percents[i])))

        pipeline.run_match(src_off, per_source, em_tgt, em_sslot, em_tslot, key_off, off_all,
                           idx_all, sb_all, eb_all, threads, params, on_result, progress)
    finally:
        pool.close()
    return count, duplicate_rows


def _merge_batch(chunk_dir, batch_file):
    """main.go:558. Concatenate a combination's chunks into one batch file."""
    command = (f"find {quote(chunk_dir)} -type f -print0 | sort -zV | "
               f"xargs -0 --no-run-if-empty lz4cat --rm | lz4 -q > {quote(batch_file)}")
    subprocess.run(["bash", "-c", command], check=False)


def align(source_files, source_metadata, output_path, target_files="", target_metadata="",
          output_workers=0, lz4_level=3, **params):
    """Run the sequence aligner. Mirrors the compareNgrams binary's flags.

    source_files / target_files   directories of per-document ngram JSON files
    source_metadata / target_metadata   paths to the corpora's metadata.json
    output_path                   directory for result_batches/, count.txt,
                                  duplicate_files.csv and alignment_config.ini
    output_workers                chunk writer processes, 0 means `threads`
    lz4_level                     chunk compression level, 3 as in main.go:895
    **params                      any of DEFAULTS: threads, sort_by, source_batch,
                                  target_batch, matching_window_size, max_gap, flex_gap,
                                  minimum_matching_ngrams,
                                  minimum_matching_ngrams_in_window,
                                  minimum_matching_ngrams_in_docs, context_size,
                                  duplicate_threshold, merge_passages_on_byte_distance,
                                  merge_passages_on_ngram_distance,
                                  passage_distance_multiplier, debug, ngram_index

    Returns the number of alignments found.
    """
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown alignment parameter(s): {', '.join(sorted(unknown))}")
    params = _normalize(params)
    params["output_workers"] = int(output_workers) or params["threads"]
    params["lz4_level"] = int(lz4_level)
    if params["debug"]:
        print("The Python aligner has no debug output; use aligner = go for that.",
              file=sys.stderr, flush=True)
    if not source_metadata:
        raise ValueError("no source metadata provided")
    if target_files == source_files:                                  # main.go:182
        target_files = ""
    print("Loading metadata...", end="", flush=True)
    source_meta = load_metadata(source_metadata)
    if not source_meta:
        raise ValueError(f"metadata file {source_metadata} is empty")
    target_meta = load_metadata(target_metadata)
    print("done.", flush=True)
    source_docs = get_files(source_files, source_meta, params["sort_by"])
    target_docs = get_files(target_files, target_meta, params["sort_by"])
    if target_docs and not target_metadata:
        raise ValueError("no target metadata provided")
    if not source_docs:
        raise ValueError(f"no ngram files in {source_files}")

    output.write_config(output_path, _config_values(params, output_path))
    source_batches = _batches(source_docs, min(params["source_batch"], len(source_docs)))
    if target_docs:
        target_batches = _batches(target_docs, min(params["target_batch"], len(target_docs)))
        same_corpus = False
    else:
        target_batches = source_batches                               # main.go:384
        same_corpus = True

    batch_path = os.path.join(output_path, "result_batches")
    batched = len(source_batches) > 1 or len(target_batches) > 1
    chunk_dir = os.path.join(batch_path, "result_chunks") if batched else batch_path
    if not batched:
        # main.go:564 replaces result_batches wholesale for a single combination.
        shutil.rmtree(batch_path, ignore_errors=True)
    os.makedirs(chunk_dir, exist_ok=True)
    output.write_duplicates(output_path, ())
    pipeline.warmup()
    loader.warmup()

    count = 0
    for source_number, source_batch in enumerate(source_batches):
        if len(source_batches) > 1:
            print(f"\n### Comparing source batch {source_number + 1} against all... ###",
                  flush=True)
        for target_number, target_batch in enumerate(target_batches):
            if same_corpus and source_number > target_number:
                continue                                              # main.go:406
            same_array = same_corpus and source_number == target_number

            def progress(done, total):
                print(f"\rComparing files... {100 * done // total}%", end="", flush=True)

            print("Comparing files... 0%", end="", flush=True)
            batch_count, duplicate_rows = _run_combination(
                params, source_batch, target_batch, source_meta, target_meta, same_array,
                chunk_dir, progress)
            print("\r\033[KComparing files... done.", flush=True)
            count += batch_count
            output.write_duplicates(output_path, duplicate_rows, append=True)
            if batched:
                print("Merging results... ", end="", flush=True)
                _merge_batch(chunk_dir, os.path.join(
                    batch_path, f"batch-{source_number + 1}-{target_number + 1}.lz4"))
                print("done.", flush=True)
            gc.collect()
    print(f"{count} pairwise alignments found...", flush=True)
    output.write_count(output_path, count)
    return count
