"""Threaded phases of the inverted-index aligner: postings, emissions, matching.

Each phase is a numpy/threads driver over the nogil kernels in `invidx`. The
document arrays come from `loader.load_corpus`; see `invidx` for the layout.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from . import invidx


def _split_ranges(total, count):
    edges = np.linspace(0, total, count + 1).astype(np.int64)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(count) if edges[i + 1] > edges[i]]


def _bucket_ranges(bucket_count_cum, n_workers):
    """Contiguous MSD-bucket ranges holding roughly equal numbers of postings entries."""
    total = int(bucket_count_cum[-1])
    n_buckets = bucket_count_cum.shape[0] - 1
    edges = [0]
    for worker in range(1, n_workers):
        target = total * worker // n_workers
        edge = int(np.searchsorted(bucket_count_cum, target, side="left"))
        edges.append(min(max(edge, edges[-1]), n_buckets))
    edges.append(n_buckets)
    return [(edges[i], edges[i + 1]) for i in range(n_workers) if edges[i + 1] > edges[i]]


def build_postings(keys_all, threads):
    """Sort every (document, key) slot by biased key: MSD bucketing, then per-bucket LSD."""
    n_slots = keys_all.shape[0]
    ranges = _split_ranges(n_slots, max(1, min(threads * 2, n_slots)))
    hist = np.zeros((len(ranges), invidx.MSD_BUCKETS), np.int64)
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(lambda a: invidx.hist_msd(keys_all, a[1][0], a[1][1], hist[a[0]]),
                      list(enumerate(ranges))))
    totals = hist.sum(axis=0)
    bstart = np.zeros(invidx.MSD_BUCKETS + 1, np.int64)
    np.cumsum(totals, out=bstart[1:])
    write_off = bstart[:-1][None, :] + np.cumsum(hist, axis=0) - hist    # per-chunk cursors
    post_u = np.empty(n_slots, np.uint32)
    post_slot = np.empty(n_slots, np.int32)
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(lambda a: invidx.scatter_msd(keys_all, a[1][0], a[1][1], write_off[a[0]],
                                                   post_u, post_slot),
                      list(enumerate(ranges))))
    del write_off, hist
    max_bucket = int(totals.max()) if totals.size else 0

    def sort_task(bucket_range):
        invidx.sort_buckets(bstart, bucket_range[0], bucket_range[1], post_u, post_slot,
                            np.empty(max_bucket, np.uint32), np.empty(max_bucket, np.int32),
                            np.empty(invidx.LSD_RADIX, np.int64))

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(sort_task,
                      _bucket_ranges(bstart, max(1, min(threads * 4, invidx.MSD_BUCKETS)))))
    return post_u, post_slot, bstart, max_bucket


def build_emissions(post_u, post_slot, bstart, max_bucket, key_off, n_src, same_doc, threads):
    """One emission per (shared key, comparable document pair), source-major."""
    ranges = _bucket_ranges(bstart, max(1, min(threads, invidx.MSD_BUCKETS)))
    n_workers = len(ranges)
    counts = np.zeros((n_workers, n_src), np.int64)
    results = [None] * n_workers

    def count_task(arg):
        worker, bucket_range = arg
        results[worker] = invidx.emit_count(bstart, bucket_range[0], bucket_range[1], post_u,
                                            post_slot, key_off, n_src, same_doc, counts[worker],
                                            np.empty(max_bucket, np.int32))

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(count_task, list(enumerate(ranges))))
    repeats = sum(r[1] for r in results)
    if repeats:
        raise RuntimeError(f"{repeats} ngram key(s) repeat within a document")
    per_source = counts.sum(axis=0)
    total = int(per_source.sum())
    src_off = np.zeros(n_src + 1, np.int64)
    np.cumsum(per_source, out=src_off[1:])
    write_off = src_off[:-1][None, :] + np.cumsum(counts, axis=0) - counts
    del counts
    em_tgt = np.empty(total, np.int32)
    em_sslot = np.empty(total, np.int32)
    em_tslot = np.empty(total, np.int32)

    def scatter_task(arg):
        worker, bucket_range = arg
        invidx.emit_scatter(bstart, bucket_range[0], bucket_range[1], post_u, post_slot,
                            key_off, n_src, same_doc, write_off[worker],
                            em_tgt, em_sslot, em_tslot,
                            np.empty(max_bucket, np.int32), np.empty(max_bucket, np.int32))

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(scatter_task, list(enumerate(ranges))))
    return em_tgt, em_sslot, em_tslot, src_off, per_source


def run_match(src_off, per_source, em_tgt, em_sslot, em_tslot, key_off, off_all, idx_all,
              sb_all, eb_all, threads, params, on_result, progress=None, task_mult=4):
    """Compare every document pair with emissions, longest source first.

    `on_result(rows, duplicates, percents, stats)` is called once per task, from the
    calling thread, so results can be written out while later tasks are still running.
    `progress(done, total)` is called with the task counts after each result.
    """
    n_docs = key_off.shape[0] - 1
    npass = max(1, (max(1, int(n_docs - 1)).bit_length() + 7) // 8)
    live = np.nonzero(per_source > 0)[0]
    order = live[np.argsort(-per_source[live], kind="stable")]
    n_tasks = max(1, min(threads * task_mult, order.shape[0]))
    tasks = [np.ascontiguousarray(order[i::n_tasks].astype(np.int32)) for i in range(n_tasks)]
    args = (params["minimum_matching_ngrams_in_docs"], params["duplicate_threshold"],
            params["matching_window_size"], params["max_gap"], params["flex_gap"],
            params["minimum_matching_ngrams"], params["minimum_matching_ngrams_in_window"],
            params["merge_passages_on_byte_distance"],
            params["merge_passages_on_ngram_distance"],
            params["passage_distance_multiplier"])

    def one(task):
        return invidx.match_sources(task, src_off, em_tgt, em_sslot, em_tslot, key_off,
                                    off_all, idx_all, sb_all, eb_all, npass, *args)

    done = 0
    if threads == 1:
        for task in tasks:
            rows, _, stats, dups, percents = one(task)
            on_result(rows, dups, percents, stats)
            done += 1
            if progress:
                progress(done, n_tasks)
    else:
        with ThreadPoolExecutor(threads) as pool:
            futures = [pool.submit(one, task) for task in tasks]
            for future in as_completed(futures):
                rows, _, stats, dups, percents = future.result()
                on_result(rows, dups, percents, stats)
                done += 1
                if progress:
                    progress(done, n_tasks)
    return n_tasks


def warmup():
    """Compile every kernel on tiny inputs, off the critical path."""
    keys = np.array([1, 2], np.int32)
    hist = np.zeros(invidx.MSD_BUCKETS, np.int64)
    invidx.hist_msd(keys, 0, 2, hist)
    bstart = np.zeros(invidx.MSD_BUCKETS + 1, np.int64)
    np.cumsum(hist, out=bstart[1:])
    post_u = np.empty(2, np.uint32)
    post_slot = np.empty(2, np.int32)
    invidx.scatter_msd(keys, 0, 2, bstart[:-1].copy(), post_u, post_slot)
    invidx.sort_buckets(bstart, 0, invidx.MSD_BUCKETS, post_u, post_slot,
                        np.empty(4, np.uint32), np.empty(4, np.int32),
                        np.empty(invidx.LSD_RADIX, np.int64))
    key_off = np.array([0, 1, 2], np.int64)
    for same_doc in (np.empty(0, np.int32), np.array([-1], np.int32)):
        n_src = key_off.shape[0] - 1 if same_doc.shape[0] == 0 else 1
        invidx.emit_count(bstart, 0, invidx.MSD_BUCKETS, post_u, post_slot, key_off, n_src,
                          same_doc, np.zeros(n_src, np.int64), np.empty(4, np.int32))
        invidx.emit_scatter(bstart, 0, invidx.MSD_BUCKETS, post_u, post_slot, key_off, n_src,
                            same_doc, np.zeros(n_src, np.int64), np.empty(1, np.int32),
                            np.empty(1, np.int32), np.empty(1, np.int32),
                            np.empty(4, np.int32), np.empty(4, np.int32))
    ones = np.array([0, 0], np.int32)
    invidx.match_sources(np.array([0], np.int32), np.array([0, 1, 1], np.int64),
                         np.array([1], np.int32), np.array([0], np.int32),
                         np.array([1], np.int32), key_off,
                         np.array([0, 1, 1, 2], np.int32), ones, ones, ones, 1,
                         1, 200.0, 30, 15, False, 1, 1, True, True, 0.5)
