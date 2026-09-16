"""Threaded phases of the inverted-index aligner: postings, key-group index, matching.

Each phase is a numpy/threads driver over the nogil kernels in `invidx`. The
document arrays come from `loader.load_corpus`; see `invidx` for the layout.
"""
import threading
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
    return post_u, post_slot, bstart


def index_postings(post_u, post_slot, bstart, key_off, n_src, same_doc, threads):
    """Index the key groups for the per-source sweeps: each posting's document, where a
    source's sweep of its group starts, and how many emissions each source will yield."""
    ranges = _bucket_ranges(bstart, max(1, min(threads, invidx.MSD_BUCKETS)))
    n_workers = len(ranges)
    separate = same_doc.shape[0] != 0
    post_doc = np.empty(post_u.shape[0], np.int32)
    sweep_at = np.empty(int(key_off[n_src]), np.int32)   # source slots only
    counts = np.zeros((n_workers, n_src), np.int64)
    results = [None] * n_workers

    def task(arg):
        worker, bucket_range = arg
        results[worker] = invidx.index_postings(bstart, bucket_range[0], bucket_range[1],
                                                post_u, post_slot, key_off, n_src, separate,
                                                post_doc, sweep_at, counts[worker])

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(task, list(enumerate(ranges))))
    repeats = sum(results)
    if repeats:
        raise RuntimeError(f"{repeats} ngram key(s) repeat within a document")
    return post_doc, sweep_at, counts.sum(axis=0)


class _Scratch:
    """One thread's emission buffers, grown to the largest source the thread has owned."""

    __slots__ = ("tcnt", "toff", "tcur", "size", "arrays")

    def __init__(self, n_docs):
        self.tcnt = np.zeros(n_docs, np.int32)
        self.toff = np.empty(n_docs + 1, np.int64)
        self.tcur = np.empty(n_docs, np.int64)
        self.size = 0
        self.arrays = ()

    def sized(self, need):
        if self.size < need:
            self.arrays = (np.empty(need, np.int32), np.empty(need, np.int32))
            self.size = need
        return self.arrays


def run_match(keys_all, key_off, sweep_at, post_u, post_slot, post_doc, per_source,
              same_doc, off_all, idx_all, sb_all, eb_all, threads, params, on_result,
              progress=None):
    """Compare every source document with the documents it is paired with, longest first.

    One source per task: its emissions are built, grouped and matched inside the task, so
    only the sources in flight hold emissions.

    `on_result(rows, duplicates, percents, stats)` is called once per source, from the
    calling thread, so results can be written out while later sources are still running.
    `progress(done, total)` is called with the source counts after each result.
    """
    n_docs = key_off.shape[0] - 1
    live = np.nonzero(per_source > 0)[0]
    order = live[np.argsort(-per_source[live], kind="stable")]
    args = (params["minimum_matching_ngrams_in_docs"], params["duplicate_threshold"],
            params["matching_window_size"], params["max_gap"], params["flex_gap"],
            params["minimum_matching_ngrams"], params["minimum_matching_ngrams_in_window"],
            params["merge_passages_on_byte_distance"],
            params["merge_passages_on_ngram_distance"],
            params["passage_distance_multiplier"])
    local = threading.local()

    def one(source):
        scratch = getattr(local, "scratch", None)
        if scratch is None:
            scratch = local.scratch = _Scratch(n_docs)
        ssl, tsl = scratch.sized(int(per_source[source]))
        excl = int(same_doc[source]) if same_doc.shape[0] else -1
        return invidx.align_source(int(source), excl, keys_all, key_off, sweep_at, post_u,
                                   post_slot, post_doc, scratch.tcnt, scratch.toff,
                                   scratch.tcur, ssl, tsl, off_all, idx_all, sb_all, eb_all,
                                   *args)

    n_tasks = order.shape[0]
    done = 0
    if threads == 1:
        for source in order:
            rows, _, stats, dups, percents = one(source)
            on_result(rows, dups, percents, stats)
            done += 1
            if progress:
                progress(done, n_tasks)
    else:
        with ThreadPoolExecutor(threads) as pool:
            futures = [pool.submit(one, source) for source in order]
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
    post_doc = np.empty(2, np.int32)
    sweep_at = np.empty(2, np.int32)
    for separate in (False, True):
        n_src = 1 if separate else 2
        invidx.index_postings(bstart, 0, invidx.MSD_BUCKETS, post_u, post_slot, key_off,
                              n_src, separate, post_doc, sweep_at,
                              np.zeros(n_src, np.int64))
    ones = np.array([0, 0], np.int32)
    one = np.ones(1, np.int32)
    invidx.align_source(0, -1, keys, key_off, sweep_at, post_u, post_slot, post_doc,
                        np.zeros(2, np.int32), np.zeros(3, np.int64), np.zeros(2, np.int64),
                        one, one, np.array([0, 1, 1, 2], np.int32), ones, ones, ones,
                        1, 200.0, 30, 15, False, 1, 1, True, True, 0.5)
