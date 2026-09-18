"""Driver for the Python sequence aligner: the `align()` entry point.

`align()` takes the alignment parameters as keyword arguments and
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

from concurrent.futures import ThreadPoolExecutor

from .. import ngram_binary
from . import tracing
from . import inverted_index, ngram_loader, output
from .documents import get_files, load_metadata

# sa_config.ini overrides some of these.
DEFAULTS = dict(
    threads=4,
    sort_by="year",
    source_batch=1,
    target_batch=1,
    # A combination may not load more n-gram positions than this: ngram_loader's CSR
    # offsets are int32, so 2^31 is the hard ceiling. A parameter rather than a constant
    # so a test can force batching on a corpus small enough to check the result.
    max_positions=2 ** 31,
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
    debug_minimum_ngrams=0,
    debug_pairs="",
)

_INT_PARAMS = ("threads", "source_batch", "target_batch", "max_positions",
               "matching_window_size", "max_gap",
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
    how the caller asks for unsorted, document-ID order.
    """
    clean = dict(DEFAULTS)
    clean.update({k: v for k, v in params.items()
                  if v is not None and not (v == "" and k in _TYPED_PARAMS)})
    for key in _INT_PARAMS:
        clean[key] = int(clean[key])
    for key in _FLOAT_PARAMS:
        clean[key] = float(clean[key])
    if clean["debug_minimum_ngrams"] <= 0:
        # Near misses only: a run one ngram short of the threshold. Shorter runs are
        # counted in the per-pair summary instead.
        clean["debug_minimum_ngrams"] = max(1, clean["minimum_matching_ngrams"] - 1)
    for key in _BOOL_PARAMS:
        clean[key] = _as_bool(clean[key])
    return clean


def _config_values(params, output_path):
    """The alignment_config.ini payload, with the published key spellings."""
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


def _positions(docs, threads):
    """N-gram positions per document, or None if any of them is JSON.

    A binary index carries the count in its 24-byte header, which is what
    `load_corpus`'s sizing pass reads anyway, so this costs one small read per
    document. Counting a JSON index means scanning it, and generation has not written
    JSON since the binary format landed, so that path keeps the loader's guard instead
    of being sized in advance.
    """
    if any(not ngram_loader.is_binary(path) for _, path in docs):
        return None
    counts = np.zeros(len(docs), np.int64)

    def count(i):
        counts[i] = ngram_binary.read_header(docs[i][1])[1]

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(count, range(len(docs))))
    return counts


def _slice_totals(counts, batch_count):
    """Positions per batch, for the slices `_batches` would cut at this count."""
    size = int(math.ceil(counts.shape[0] / batch_count))
    return [int(counts[i:i + size].sum()) for i in range(0, counts.shape[0], size)]


def _worst_combination(counts, batch_count):
    """The most positions one combination would load, comparing a corpus with itself.

    One batch means one combination holding everything. Two or more means the heaviest
    combination is the off-diagonal one that loads the two largest batches -- which is
    why `source_batch = 2` buys nothing on a self-comparison: the two batches together
    are still the whole corpus.
    """
    totals = _slice_totals(counts, batch_count)
    if len(totals) < 2:
        return totals[0] if totals else 0
    return sum(sorted(totals)[-2:])


def _raise_batches(counts, requested, budget, worst):
    """The smallest batch count at or above `requested` that keeps `worst` within budget.

    `_batches` slices by document count, so raising the count shrinks every slice. It
    starts from the count the total positions alone require, which is a lower bound, so
    this walks up over a couple of values rather than from one.
    """
    total = int(counts.sum())
    start = max(requested, 1, int(math.ceil(total / budget)) if budget > 0 else 1)
    for batch_count in range(start, counts.shape[0] + 1):
        if worst(counts, batch_count) <= budget:
            return batch_count
    return counts.shape[0]


def _size_batches(params, source_docs, target_docs, budget, threads):
    """Raise source_batch / target_batch until no combination exceeds `budget` positions.

    Only ever raises. An explicit batch count is a memory choice the caller made, and
    this is a correctness floor under it, not a second opinion. It prints whenever it
    moves: a run that silently sliced itself differently than asked would be worse than
    either outcome.
    """
    source_counts = _positions(source_docs, threads)
    if source_counts is None or source_counts.size == 0:
        return
    if target_docs:
        target_counts = _positions(target_docs, threads)
        if target_counts is None:
            return
        # The sides are sliced independently and one of each is loaded together, so each
        # gets half the budget.
        def worst(counts, count):
            return max(_slice_totals(counts, count))
        share = budget // 2
        plan = (("source_batch", source_docs, source_counts, share),
                ("target_batch", target_docs, target_counts, share))
    else:
        worst = _worst_combination
        plan = (("source_batch", source_docs, source_counts, budget),)
    for name, docs, counts, allowance in plan:
        biggest = int(counts.max())
        if biggest > allowance:
            doc = docs[int(counts.argmax())][0]
            raise ValueError(
                f"document {doc} alone holds {biggest:,} ngram positions, more than the "
                f"{allowance:,} one combination may load. No batch count can help; the "
                f"index would have to be split or max_positions raised, and the CSR "
                f"offsets are int32.")
        asked = int(params[name])
        needed = _raise_batches(counts, asked, allowance, worst)
        if needed > asked:
            print(f"{name} raised {asked} -> {needed}: at {asked} a combination would "
                  f"load {worst(counts, asked):,} ngram positions, over the "
                  f"{allowance:,} allowed by max_positions={budget:,}.", flush=True)
            params[name] = needed


def _batches(files, count):
    """Consecutive slices of ceil(len/count) documents.

    The callers clamp `count` to the document count, so a batch count larger than the
    corpus gives one document per batch rather than empty slices.
    """
    if not files:
        return []
    size = int(math.ceil(len(files) / count))
    return [files[i:i + size] for i in range(0, len(files), size)]


def chunk_ranges(position, n_targets, threads, same_array):
    """Target ranges of the chunk files written for one source document.

    One file per worker, and the worker count itself depends on how many targets are
    left, so the chunk file names depend on this split. The names carry the source
    document first, which is what keeps a source's records together once the chunks
    are concatenated in `sort -V` order: alignment_merger.first_step_merge reads that
    stream once and flushes whenever source_doc_id changes.
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
        increment = local_length - start        # the clamp below hides this
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

    A range with no alignment but a duplicate still gets a file, matching the empty
    entry a duplicate target produces.
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
                     same_array, chunk_dir, progress, trace=None):
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
        key_offsets, ngram_keys, position_offsets, ngram_indices, start_bytes, end_bytes = \
            ngram_loader.load_corpus(paths, threads)
        posting_keys, posting_slots, bucket_starts = inverted_index.build_postings(ngram_keys, threads)
        posting_docs, sweep_starts, per_source = inverted_index.index_postings(
            posting_keys, posting_slots, bucket_starts, key_offsets, n_sources, same_doc, threads)
        del bucket_starts
        gc.collect()

        def on_result(rows, dups, percents, _stats):
            nonlocal count
            count += rows.shape[0]
            if rows.shape[0] or dups.shape[0]:
                pool.submit(np.ascontiguousarray(rows),
                            _jobs(rows, dups, docs, n_targets, target_base, threads,
                                  same_array))
            for i in range(dups.shape[0]):
                source_slot, target_slot = int(dups[i, 0]), int(dups[i, 1])
                duplicate_rows.append((source_slot, target_slot,
                                       output.duplicate_row(metas[source_slot],
                                                            metas[target_slot],
                                                            float(percents[i]))))

        inverted_index.run_match(ngram_keys, key_offsets, sweep_starts, posting_keys,
                           posting_slots, posting_docs,
                           per_source, same_doc, position_offsets, ngram_indices,
                           start_bytes, end_bytes,
                           threads, params, on_result, progress)
        # The caller's progress line is finished here so the trace's own message is
        # not overwritten by it.
        print("\r\033[KComparing files... done.", flush=True)
        if trace is not None:
            print("Tracing compared pairs... ", end="", flush=True)
            written = tracing.write_traces(
                trace["output_path"], docs,
                tracing.Corpus(key_offsets, ngram_keys, position_offsets,
                                    ngram_indices, start_bytes, end_bytes),
                params, same_doc, n_sources, trace["ngram_index"], trace["pairs"])
            print(f"{written} pair(s) written.", flush=True)
    finally:
        pool.close()
    # Workers complete in a non-reproducible order;
    # sort by document slot so the file is.
    duplicate_rows.sort()
    return count, [row for _, _, row in duplicate_rows]


def _merge_batch(chunk_dir, batch_file):
    """Concatenate a combination's chunks into one batch file."""
    command = (f"find {quote(chunk_dir)} -type f -print0 | sort -zV | "
               f"xargs -0 --no-run-if-empty lz4cat --rm | lz4 -q > {quote(batch_file)}")
    subprocess.run(["bash", "-c", command], check=False)


def align(source_files, source_metadata, output_path, target_files="", target_metadata="",
          output_workers=0, lz4_level=3, **params):
    """Run the sequence aligner.

    source_files / target_files   directories of per-document ngram files, JSON or binary
    source_metadata / target_metadata   paths to the corpora's metadata.json
    output_path                   directory for result_batches/, count.txt,
                                  duplicate_files.csv and alignment_config.ini
    output_workers                chunk writer processes, 0 means `threads`
    lz4_level                     chunk compression level, 3 by default
    **params                      any of DEFAULTS: threads, sort_by, source_batch,
                                  target_batch, matching_window_size, max_gap, flex_gap,
                                  minimum_matching_ngrams,
                                  minimum_matching_ngrams_in_window,
                                  minimum_matching_ngrams_in_docs, context_size,
                                  duplicate_threshold, merge_passages_on_byte_distance,
                                  merge_passages_on_ngram_distance,
                                  passage_distance_multiplier, debug, ngram_index,
                                  debug_minimum_ngrams, debug_pairs

    Returns the number of alignments found.
    """
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown alignment parameter(s): {', '.join(sorted(unknown))}")
    params = _normalize(params)
    params["output_workers"] = int(output_workers) or params["threads"]
    params["lz4_level"] = int(lz4_level)
    ngram_index = {}
    if params["debug"]:
        if params["ngram_index"]:
            ngram_index = tracing.load_ngram_index(params["ngram_index"])
        else:
            print("--debug without --ngram_index: traces cannot name their ngrams.",
                  file=sys.stderr, flush=True)
    pair_filter = tracing.parse_pairs(params["debug_pairs"])
    if not source_metadata:
        raise ValueError("no source metadata provided")
    if target_files == source_files:
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

    # Before write_config, so the file records the batch counts the run will use.
    if int(params["max_positions"]) > 0:
        _size_batches(params, source_docs, target_docs, int(params["max_positions"]),
                      params["threads"])
    output.write_config(output_path, _config_values(params, output_path))
    source_batches = _batches(source_docs, min(params["source_batch"], len(source_docs)))
    if target_docs:
        target_batches = _batches(target_docs, min(params["target_batch"], len(target_docs)))
        same_corpus = False
    else:
        target_batches = source_batches
        target_meta = source_meta
        same_corpus = True

    batch_path = os.path.join(output_path, "result_batches")
    batched = len(source_batches) > 1 or len(target_batches) > 1
    chunk_dir = os.path.join(batch_path, "result_chunks") if batched else batch_path
    if not batched:
        # result_batches is replaced wholesale for a single combination.
        shutil.rmtree(batch_path, ignore_errors=True)
    os.makedirs(chunk_dir, exist_ok=True)
    output.write_duplicates(output_path, ())
    trace = ({"output_path": output_path, "ngram_index": ngram_index,
              "pairs": pair_filter} if params["debug"] else None)
    inverted_index.warmup()
    ngram_loader.warmup()

    count = 0
    for source_number, source_batch in enumerate(source_batches):
        if len(source_batches) > 1:
            print(f"\n### Comparing source batch {source_number + 1} against all... ###",
                  flush=True)
        for target_number, target_batch in enumerate(target_batches):
            if same_corpus and source_number > target_number:
                continue
            same_array = same_corpus and source_number == target_number

            def progress(done, total):
                print(f"\rComparing files... {100 * done // total}%", end="", flush=True)

            print("Comparing files... 0%", end="", flush=True)
            batch_count, duplicate_rows = _run_combination(
                params, source_batch, target_batch, source_meta, target_meta, same_array,
                chunk_dir, progress, trace)
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
