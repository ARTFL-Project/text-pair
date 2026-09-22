"""Output phase of the sequence aligner.

Two halves. `passage text` turns a byte range of a document into the passage and its
surrounding context, stripping markup and resolving entities. The rest writes the
artifacts: the alignment chunks, count.txt, duplicate_files.csv and
alignment_config.ini. Chunk files are named
{sourceDocID}-{firstTargetDocID}-{lastTargetDocID}.lz4 under <output_path>/result_batches,
one per source document and per target range.

Writers are processes, not threads: regex, unescape and dumps all hold the GIL.
"""
import codecs
import mmap
import os
import re
import time
import unicodedata
from html.entities import html5 as _HTML5

import lz4.frame
import orjson
from tqdm import tqdm

DUP_HEADER = ("source_title", "source_author", "source_filename", "source_philo_id",
              "source_byte_offsets", "target_title", "target_author", "target_filename",
              "target_philo_id", "target_byte_offsets", "overlap")

CONFIG_KEYS = ("matchingWindowSize", "maxGap", "flexGap", "minimumMatchingNgrams",
               "minimumMatchingNgramsInWindow", "minimumMatchingNgramsInDocs", "contextSize",
               "mergeOnByteDistance", "mergeOnNgramDistance", "passageDistanceMultiplier",
               "duplicateThreshold", "sourceBatch", "targetBatch", "outputPath", "numThreads",
               "sortingField", "debug")

_MMAP_CACHE_LIMIT = 4096
# Extracted passages held per write() call. A chunk's records reach at most twice this
# many distinct passages, so the working set fits and the cap only bounds carry-over.
_TEXT_CACHE_LIMIT = 8192
_DUMP_OPT = orjson.OPT_SORT_KEYS          # the format has object keys sorted


# --------------------------------------------------------------- passage text

# The output format is fixed by the corpora already published with it, so three details
# here are load-bearing and easy to get wrong: the `\Z` anchor rather than `$`, the
# character classes written out rather than `\s`, and one U+FFFD per invalid byte. The
# `escape.go` line numbers cite the Go implementation this reproduces, kept at
# /disk1/shared/text-pair-validation/go_variants_source.

tags = re.compile(r"<[^>]*?>")
brokenBeginTags = re.compile(r"^[^<]*?>")
brokenEndTags = re.compile(r"<[^>]*?\Z")     # \Z, not $: a trailing \n must not end it
spaces = re.compile(r" +")
tabEntities = re.compile(r"(&#9;)+")
cleanStart = re.compile(r"^[^\t\n\f\r ]+ ")
cleanEnd = re.compile(r" [^\t\n\f\r ]+\Z")

# The format wants one U+FFFD per invalid byte; Python's default replacement
# emits one per maximal invalid subpart instead.
codecs.register_error("replace_each_byte", lambda e: ("�" * (e.end - e.start), e.end))

# Not Python's html.unescape: it resolves a truncated "&#8" and blanks the C0
# code points, where this format leaves the first alone and emits the second.
# The table is html.entities.html5 minus nGt;/nLt;, which the format never resolved;
# its without-semicolon fallback only consults the single-rune map.
_ENT = {k: v for k, v in _HTML5.items() if k not in ("nGt;", "nLt;")}
_ENT1 = {k: v for k, v in _ENT.items() if len(v) == 1}
_WIN1252 = ("\u20ac\u0081\u201a\u0192\u201e\u2026\u2020\u2021\u02c6\u2030\u0160"
            "\u2039\u0152\u008d\u017d\u008f\u0090\u2018\u2019\u201c\u201d\u2022"
            "\u2013\u2014\u02dc\u2122\u0161\u203a\u0153\u009d\u017e\u0178")   # escape.go:16
_REF = re.compile(r"&(?:#[xX][0-9a-fA-F]*;?|#[0-9]*;?|[0-9A-Za-z]*;?)")


def _ref(m):
    s = m.group(0)
    if len(s) <= 1:
        return s
    if s[1] == "#":
        body = s[2:]
        hexd = body[:1] in ("x", "X")
        if hexd:
            body = body[1:]
        semi = body.endswith(";")
        if semi:
            body = body[:-1]
        if 2 + hexd + len(body) + semi <= 3:       # escape.go:106, no characters matched
            return s
        x = int(body, 16 if hexd else 10) if body else 0
        x = (x & 0xFFFFFFFF) - (0x100000000 if x & 0x80000000 else 0)   # code points wrap as int32
        if 0x80 <= x <= 0x9F:
            return _WIN1252[x - 0x80]
        if x <= 0 or 0xD800 <= x <= 0xDFFF or x > 0x10FFFF:
            return "\ufffd"                      # x < 0 reaches utf8.EncodeRune as RuneError
        return chr(x)
    name = s[1:]
    if not name:
        return s
    v = _ENT.get(name)
    if v is not None:
        return v
    for j in range(min(len(name) - 1, 6), 1, -1):  # longestEntityWithoutSemicolon = 6
        v = _ENT1.get(name[:j])
        if v is not None:
            return v + name[j:]
    return s


def unescape(text):
    """html.UnescapeString (escape.go:38-162)."""
    return _REF.sub(_ref, text)


_unescape = unescape
_tags_sub = tags.sub
_bb_sub = brokenBeginTags.sub
_be_sub = brokenEndTags.sub
_sp_sub = spaces.sub
_tab_sub = tabEntities.sub


def get_text(buf, start, end, is_match):
    """`buf` is the whole document, as an mmap or bytes.

    A passage running past EOF is simply short: the equivalent read would pad with
    NULs and then trim them, which yields the same text as a slice.
    """
    if start < 0:
        start = 0
    passage = buf[start:end]
    if b"\x00" in passage:
        passage = passage.strip(b"\x00")
    if b"\xc2\xa0" in passage:
        passage = passage.replace(b"\xc2\xa0", b" ")
    text = passage.decode("utf-8", "replace_each_byte")
    # Each substitution below is skipped when its pattern provably cannot match, which
    # is the common case for prose: `tags` and `brokenEndTags` both need a '<',
    # `brokenBeginTags` needs a '>', and `spaces` can only change the text where two
    # spaces meet, since replacing one space with one space is a no-op. Between them
    # these are most of the aligner's regex calls.
    if "<" in text:
        text = _tags_sub("", text)
    if not is_match:
        if ">" in text:
            text = _bb_sub("", text)
        if "<" in text:
            text = _be_sub("", text)
    if "&" in text:
        text = _unescape(text)
    if "\\" in text:
        text = text.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
    if "\t" in text:
        text = text.replace("\t", " ")
    if "&#9;" in text:
        text = _tab_sub(" ", text)
    if "\n" in text:
        text = text.replace("\n", " ")
    if "  " in text:
        text = _sp_sub(" ", text)
    return text


def alignment_to_text(buf, start_byte, end_byte, context_size):
    """cleanStart/cleanEnd apply to the contexts only, never the passage."""
    return (cleanStart.sub("", get_text(buf, start_byte - context_size, start_byte, False)),
            get_text(buf, start_byte, end_byte, True),
            cleanEnd.sub("", get_text(buf, end_byte, end_byte + context_size, False)))


def _atoi(s):
    """0 on any parse error, which is what the metadata's consumers expect."""
    try:
        return int(s)
    except (TypeError, ValueError):
        return 0


def rel_pos(start_byte, end_byte, doc_meta):
    """%.2f of byte/((end-start)/100), keeping the published spelling of the
    non-finite results a zero-length document produces."""
    coef = (_atoi(doc_meta.get("end_byte")) - _atoi(doc_meta.get("start_byte"))) / 100.0
    if coef == 0.0:
        return (_nonfinite(start_byte), _nonfinite(end_byte))
    return ("%.2f" % (start_byte / coef), "%.2f" % (end_byte / coef))


def _nonfinite(v):
    if v == 0:
        return "NaN"
    return "+Inf" if v > 0 else "-Inf"


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

    def _text(self, slot, keep=None):
        buf = self._buf.get(slot)
        if buf is None:
            if len(self._buf) >= _MMAP_CACHE_LIMIT:
                # `keep` is the source document of the job in progress: the caller holds
                # its buffer across every target, so closing it here would fail the rest
                # of the job with "mmap closed or invalid".
                kept = self._buf.pop(keep, None)
                for mapped in self._buf.values():
                    if mapped:
                        mapped.close()
                self._buf.clear()
                if kept is not None:
                    self._buf[keep] = kept
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
        # One source passage aligns to many targets, and each of those is a separate
        # record, so without this every one of them re-decodes and re-scrubs the same
        # bytes: two fifths of the extraction calls in a run are repeats.
        cache = {}
        cached = cache.get
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
                target_buf = self._text(target_slot, slot)
                for k in range(i, j):
                    source_start_byte = int(rows[k, 2])
                    source_end_byte = int(rows[k, 3])
                    target_start_byte = int(rows[k, 6])
                    target_end_byte = int(rows[k, 7])
                    key = (slot, source_start_byte, source_end_byte)
                    source_text = cached(key)
                    if source_text is None:
                        source_text = to_text(source_buf, source_start_byte,
                                              source_end_byte, ctx)
                        if len(cache) >= _TEXT_CACHE_LIMIT:
                            cache.clear()
                        cache[key] = source_text
                    key = (target_slot, target_start_byte, target_end_byte)
                    target_text = cached(key)
                    if target_text is None:
                        target_text = to_text(target_buf, target_start_byte,
                                              target_end_byte, ctx)
                        if len(cache) >= _TEXT_CACHE_LIMIT:
                            cache.clear()
                        cache[key] = target_text
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

def _worker(jobq, resq, written, tokens, docs, metas, out_dir, context_size, go_escape,
            level):
    writer = ChunkWriter(docs, metas, out_dir, context_size, go_escape, level)
    busy = 0.0
    while True:
        job = jobq.get()
        if job is None:
            break
        # A token per chunk, not per record: it costs microseconds against a write of
        # milliseconds, and `with` returns it even if the write raises.
        with tokens:
            start = time.perf_counter()
            writer.write(*job)
            busy += time.perf_counter() - start
        # One lock per submitted source, not per chunk: enough to pace a bar, and far
        # too rare to show up against the write itself.
        with written.get_lock():
            written.value += len(job[1])
    resq.put(dict(pid=os.getpid(), n_rec=writer.n_rec, n_chunk=writer.n_chunk, busy=busy))


def ensure_stream(path):
    """Leave a readable lz4 stream at `path` even if nothing was written to it.

    Concatenating no files leaves an empty file, and `lz4.frame.open` raises EOFError on
    one. A combination that found no alignments has to be readable like any other.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "wb") as handle:
            handle.write(lz4.frame.compress(b""))


class OutputPool:
    """Forked writer pool. Fork before the ngram arrays are allocated so the workers
    inherit the cleaned metadata and only tens of MB of page tables get copied.

    `n_workers` processes are forked but only `active` of them write at a time, so the
    run holds to one worker budget while matching is using the rest. `expand()` hands
    out the remaining tokens once matching is done and its threads have gone idle.
    """

    def __init__(self, n_workers, docs, metas, out_dir, context_size,
                 go_escape=True, level=3, active=0):
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        os.makedirs(out_dir, exist_ok=True)
        active = min(n_workers, active or n_workers)
        self.jobq = ctx.Queue()
        self.resq = ctx.Queue()
        self.written = ctx.Value("q", 0)     # chunks the writers have finished
        self.queued = 0                      # chunks handed to them
        self.tokens = ctx.Semaphore(active)
        self.held = n_workers - active       # tokens expand() has yet to hand out
        self.procs = [ctx.Process(target=_worker, daemon=True,
                                  args=(self.jobq, self.resq, self.written, self.tokens,
                                        docs, metas, out_dir, context_size, go_escape,
                                        level))
                      for _ in range(n_workers)]
        for proc in self.procs:
            proc.start()

    def expand(self):
        """Let every writer run. Matching has finished, so its share of the budget is
        idle and the writers are the only thing left between here and the end."""
        for _ in range(self.held):
            self.tokens.release()
        self.held = 0

    def submit(self, rows, jobs):
        self.queued += len(jobs)
        self.jobq.put((rows, jobs))

    def close(self, timeout=600):
        for _ in self.procs:
            self.jobq.put(None)
        # Matching finishes well before the writers do, so without this the run sits on
        # a full "Comparing files" bar with no sign of what it is waiting for. Writers
        # are the larger consumer of CPU on a big corpus, so the wait is not short.
        bar = tqdm(total=self.queued, initial=min(self.written.value, self.queued),
                   desc="Writing results", unit=" chunk", unit_scale=True,
                   mininterval=1.0, leave=False, disable=self.queued == 0)
        stats = []
        while len(stats) < len(self.procs):
            bar.update(min(self.written.value, self.queued) - bar.n)
            try:
                stats.append(self.resq.get(timeout=1))
            except Exception:
                dead = [p for p in self.procs if p.exitcode not in (None, 0)]
                if dead:
                    bar.close()
                    raise RuntimeError(
                        f"output worker(s) died: {[(p.pid, p.exitcode) for p in dead]}")
                timeout -= 1
                if timeout <= 0:
                    bar.close()
                    raise RuntimeError("output workers timed out")
        bar.update(min(self.written.value, self.queued) - bar.n)
        bar.close()
        for proc in self.procs:
            proc.join()
        return stats
