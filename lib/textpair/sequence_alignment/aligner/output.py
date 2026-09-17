"""Output phase of the sequence aligner.

Writes the alignment chunks, count.txt, duplicate_files.csv
(650-658) and alignment_config.ini (602-637). Chunk files are named
{sourceDocID}-{firstTargetDocID}-{lastTargetDocID}.lz4 under <output_path>/result_batches,
one per source document and per target range.

Writers are processes, not threads: regex, unescape and dumps all hold the GIL.
"""
import mmap
import os
import time
import unicodedata

import lz4.frame
import orjson

from .gotext import alignment_to_text, rel_pos

DUP_HEADER = ("source_title", "source_author", "source_filename", "source_philo_id",
              "source_byte_offsets", "target_title", "target_author", "target_filename",
              "target_philo_id", "target_byte_offsets", "overlap")

CONFIG_KEYS = ("matchingWindowSize", "maxGap", "flexGap", "minimumMatchingNgrams",
               "minimumMatchingNgramsInWindow", "minimumMatchingNgramsInDocs", "contextSize",
               "mergeOnByteDistance", "mergeOnNgramDistance", "passageDistanceMultiplier",
               "duplicateThreshold", "sourceBatch", "targetBatch", "outputPath", "numThreads",
               "sortingField", "debug")

_MMAP_CACHE_LIMIT = 4096
_DUMP_OPT = orjson.OPT_SORT_KEYS          # the format has object keys sorted


# --------------------------------------------------------------- side artifacts

def _go_float(value):
    """fmt %v of a float64 == strconv.FormatFloat(value, 'g', -1, 64)."""
    if value != value:
        return "NaN"
    if value in (float("inf"), float("-inf")):
        return "+Inf" if value > 0 else "-Inf"
    if value == int(value) and abs(value) < 1e21:
        return str(int(value))
    return repr(value)


def _go_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return _go_float(value)
    return str(value)


def write_config(output_path, config):
    """CONFIG_KEYS in order, with the published value spelling."""
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "alignment_config.ini"), "w", encoding="utf8") as output:
        output.write("## Alignment Parameters ##\n\n")
        for key in CONFIG_KEYS:
            output.write(f"{key}: {_go_value(config[key])}\n")


def write_count(output_path, count):
    """No trailing newline."""
    with open(os.path.join(output_path, "count.txt"), "w", encoding="utf8") as output:
        output.write(str(count))


def _needs_quotes(field):
    """encoding/csv fieldNeedsQuotes with Comma=',' -- note the leading-space rule,
    which the csv module's QUOTE_MINIMAL does not have."""
    if field == "":
        return False
    if field == "\\.":
        return True
    if "," in field or '"' in field or "\r" in field or "\n" in field:
        return True
    return unicodedata.category(field[0]) == "Zs" or field[0] in "\t\n\v\f\r \x85\xa0"


def _csv_field(field):
    if not _needs_quotes(field):
        return field
    return '"' + field.replace('"', '""') + '"'


def write_duplicates(output_path, rows, append=False):
    """DUP_HEADER then one row per duplicate pair, LF line endings."""
    path = os.path.join(output_path, "duplicate_files.csv")
    with open(path, "a" if append else "w", encoding="utf8") as output:
        if not append:
            output.write(",".join(_csv_field(c) for c in DUP_HEADER) + "\n")
        for row in rows:
            output.write(",".join(_csv_field(c) for c in row) + "\n")


def duplicate_row(source_meta, target_meta, percent):
    """The row duplicate_files.csv expects: source fields, target fields, overlap."""
    def field(meta, key):
        return meta.get(key, "")
    return (field(source_meta, "title"), field(source_meta, "author"),
            field(source_meta, "filename"), field(source_meta, "philo_id"),
            f'{field(source_meta, "start_byte")}-{field(source_meta, "end_byte")}',
            field(target_meta, "title"), field(target_meta, "author"),
            field(target_meta, "filename"), field(target_meta, "philo_id"),
            f'{field(target_meta, "start_byte")}-{field(target_meta, "end_byte")}',
            "%.2f" % percent)


# --------------------------------------------------------------- chunk writing

class ChunkWriter:
    """Per-worker state: metadata, prefixed-key dicts and mmap caches.

    `docs` and `metas` are indexed by document slot: sources first, then targets when
    source and target corpora differ.
    """

    def __init__(self, docs, metas, out_dir, context_size, go_escape=True, level=3):
        self.docs = docs
        self.metas = metas
        self.dir = out_dir
        self.ctx = context_size
        self.level = level
        self.go_escape = go_escape
        self._src = {}
        self._tgt = {}
        self._buf = {}
        self.n_rec = 0
        self.n_chunk = 0

    def _prefixed(self, cache, slot, prefix):
        fields = cache.get(slot)
        if fields is None:
            fields = {prefix + k: v for k, v in self.metas[slot].items()}
            cache[slot] = fields
        return fields

    def _text(self, slot):
        buf = self._buf.get(slot)
        if buf is None:
            if len(self._buf) >= _MMAP_CACHE_LIMIT:
                for mapped in self._buf.values():
                    if mapped:
                        mapped.close()
                self._buf.clear()
            try:
                handle = open(self.metas[slot]["filename"], "rb")
                buf = mmap.mmap(handle.fileno(), 0, prot=mmap.PROT_READ)
                handle.close()
            except (OSError, ValueError, KeyError):
                # An unreadable document is not an error: every passage of it comes
                # out as "", which is what the published output contains.
                buf = b""
            self._buf[slot] = buf
        return buf

    def write(self, rows, jobs):
        """rows: int32[K, 11] of (source slot, target slot, source start byte,
        source end byte, source first index, source last index, target start byte,
        target end byte, target first index, target last index, matching ngrams),
        grouped by source slot and ascending target slot within a source.
        jobs: [(source_slot, chunk_name, lo, hi)] -- hi == lo for a range holding only
        duplicates, which still gets a chunk file because the duplicate list is
        non-empty for it too.
        """
        ctx = self.ctx
        docs = self.docs
        to_text = alignment_to_text
        dumps = orjson.dumps
        for slot, name, lo, hi in jobs:
            source_doc = docs[slot]
            lines = []
            source_meta = self.metas[slot]
            source_fields = self._prefixed(self._src, slot, "source_")
            source_buf = self._text(slot) if hi > lo else None
            i = lo
            while i < hi:
                target_slot = int(rows[i, 1])
                j = i + 1
                while j < hi and rows[j, 1] == target_slot:
                    j += 1
                target_meta = self.metas[target_slot]
                record = dict(source_fields)
                record.update(self._prefixed(self._tgt, target_slot, "target_"))
                record["source_doc_id"] = source_doc
                record["target_doc_id"] = docs[target_slot]
                target_buf = self._text(target_slot)
                for k in range(i, j):
                    source_start_byte = int(rows[k, 2])
                    source_end_byte = int(rows[k, 3])
                    target_start_byte = int(rows[k, 6])
                    target_end_byte = int(rows[k, 7])
                    source_text = to_text(source_buf, source_start_byte, source_end_byte, ctx)
                    target_text = to_text(target_buf, target_start_byte, target_end_byte, ctx)
                    record["source_start_byte"] = source_start_byte
                    record["source_end_byte"] = source_end_byte
                    record["source_context_before"] = source_text[0]
                    record["source_passage"] = source_text[1]
                    record["source_context_after"] = source_text[2]
                    record["source_start_position"], record["source_end_position"] = \
                        rel_pos(source_start_byte, source_end_byte, source_meta)
                    record["target_start_byte"] = target_start_byte
                    record["target_end_byte"] = target_end_byte
                    record["target_context_before"] = target_text[0]
                    record["target_passage"] = target_text[1]
                    record["target_context_after"] = target_text[2]
                    record["target_start_position"], record["target_end_position"] = \
                        rel_pos(target_start_byte, target_end_byte, target_meta)
                    line = dumps(record, option=_DUMP_OPT)
                    if self.go_escape:
                        # Each replace copies the whole line, so test first: the
                        # membership check is a memchr and most lines contain none of
                        # these. U+2028 and U+2029 share their first two bytes.
                        if b"&" in line:
                            line = line.replace(b"&", b"\\u0026")
                        if b"<" in line:
                            line = line.replace(b"<", b"\\u003c")
                        if b">" in line:
                            line = line.replace(b">", b"\\u003e")
                        if b"\xe2\x80" in line:
                            line = (line.replace(b"\xe2\x80\xa8", b"\\u2028")
                                    .replace(b"\xe2\x80\xa9", b"\\u2029"))
                    lines.append(line)
                    lines.append(b"\n")
                i = j
            blob = lz4.frame.compress(b"".join(lines), compression_level=self.level)
            with open(os.path.join(self.dir, name), "wb") as output:
                output.write(blob)
            self.n_rec += len(lines) >> 1
            self.n_chunk += 1


# --------------------------------------------------------------- worker process

def _worker(jobq, resq, docs, metas, out_dir, context_size, go_escape, level):
    writer = ChunkWriter(docs, metas, out_dir, context_size, go_escape, level)
    busy = 0.0
    while True:
        job = jobq.get()
        if job is None:
            break
        start = time.perf_counter()
        writer.write(*job)
        busy += time.perf_counter() - start
    resq.put(dict(pid=os.getpid(), n_rec=writer.n_rec, n_chunk=writer.n_chunk, busy=busy))


class OutputPool:
    """Forked writer pool. Fork before the ngram arrays are allocated so the workers
    inherit the cleaned metadata and only tens of MB of page tables get copied."""

    def __init__(self, n_workers, docs, metas, out_dir, context_size,
                 go_escape=True, level=3):
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        os.makedirs(out_dir, exist_ok=True)
        self.jobq = ctx.Queue()
        self.resq = ctx.Queue()
        self.procs = [ctx.Process(target=_worker, daemon=True,
                                  args=(self.jobq, self.resq, docs, metas, out_dir,
                                        context_size, go_escape, level))
                      for _ in range(n_workers)]
        for proc in self.procs:
            proc.start()

    def submit(self, rows, jobs):
        self.jobq.put((rows, jobs))

    def close(self, timeout=600):
        for _ in self.procs:
            self.jobq.put(None)
        stats = []
        while len(stats) < len(self.procs):
            try:
                stats.append(self.resq.get(timeout=1))
            except Exception:
                dead = [p for p in self.procs if p.exitcode not in (None, 0)]
                if dead:
                    raise RuntimeError(
                        f"output worker(s) died: {[(p.pid, p.exitcode) for p in dead]}")
                timeout -= 1
                if timeout <= 0:
                    raise RuntimeError("output workers timed out")
        for proc in self.procs:
            proc.join()
        return stats
