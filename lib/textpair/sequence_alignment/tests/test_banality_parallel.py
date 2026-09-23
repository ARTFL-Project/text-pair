#!/usr/bin/env python3
"""Parallel banality detection must produce exactly what the serial pass produces.

    test_banality_parallel.py

The pass splits the results file on lz4 frame boundaries and rewrites each range in its
own process, so what is checked here is that the split, the windowed reads, the
re-framing and the concatenation put the records back byte for byte and in order.

Everything is synthesised: a frame layout to split, an n-gram index to look up and a
document frequencies file to score against. No corpus needed. The verdict itself is
checked too, on passages built to sit either side of it.
"""
import os
import shutil
import struct
import sys
import tempfile

import lz4.frame
import numpy as np
import orjson

from textpair.sequence_alignment import banality_finder as bf
from textpair.sequence_alignment import ngram_binary, ngram_index

FAILURES = []


def check(label, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        FAILURES.append(label)


def write_ngram_doc(path, n_ngrams, keys):
    """An ngrams_in_order file, as ngram_binary documents it: the magic, the count, the
    int64 keys in n-gram order, then their ascending int32 start bytes."""
    starts = np.arange(n_ngrams, dtype=np.int32) * 10
    with open(path, "wb") as handle:
        handle.write(ngram_binary.ORDER_MAGIC)
        handle.write(struct.pack("<q", n_ngrams))
        handle.write(keys.astype("<i8").tobytes())
        handle.write(starts.astype("<i4").tobytes())


def write_frequencies(path, documents, frequencies):
    """A document_frequencies file: the magic, the corpus's document count and the
    number of keys, then the keys ascending and how many documents hold each."""
    keys = np.array(sorted(frequencies), dtype="<i8")
    counts = np.array([frequencies[key] for key in keys.tolist()], dtype="<i4")
    with open(path, "wb") as handle:
        handle.write(ngram_index.FREQUENCY_HEADER.pack(
            ngram_index.FREQUENCY_MAGIC, documents, keys.size))
        handle.write(keys.tobytes())
        handle.write(counts.tobytes())


def build_corpus(root, documents=6, ngrams=400):
    """An n-gram index and its document frequencies: keys 1..50 are in 50 of the
    corpus's 100 documents, every other key in one."""
    order = os.path.join(root, "ngrams_in_order")
    os.makedirs(order, exist_ok=True)
    for doc in range(documents):
        # Even documents have every other n-gram common, odd ones every fourth, so a
        # passage between two even documents scores twice one between two odd ones.
        step = 2 if doc % 2 == 0 else 4
        keys = np.where(np.arange(ngrams) % step == 0,
                        np.arange(ngrams) % 50 + 1,
                        np.arange(ngrams) + 1000 + doc * 10000)
        write_ngram_doc(os.path.join(order, f"{doc}.bin"), ngrams, keys)
    frequencies = os.path.join(root, "document_frequencies.bin")
    write_frequencies(frequencies, 100, {key: 50 for key in range(1, 51)})
    return order, frequencies


# 0.82 for keys 1..50, so even pairs score about 0.41, odd ones about 0.20.
THRESHOLD = 0.3


def sides(record, documents, start):
    """Source and target fields for a record, both sides of a same-parity pair."""
    return {
        "source_doc_id": str(record % documents),
        "source_ngrams": f"{record % documents}.bin",
        "source_start_byte": start,
        "source_end_byte": start + 300,
        "target_doc_id": str((record + 2) % documents),
        "target_ngrams": f"{(record + 2) % documents}.bin",
        "target_start_byte": start,
        "target_end_byte": start + 300,
    }


def write_alignments(path, frames, per_frame, documents=6, with_field=False):
    """A results file of `frames` lz4 frames, the shape the chunk writers leave behind."""
    record = 0
    with open(path, "wb") as raw:
        for frame in range(frames):
            lines = []
            for _ in range(per_frame):
                start = (record % 30) * 10
                body = {**sides(record, documents, start),
                        "source_passage": f"passage {record}"}
                if with_field:
                    body["banality"] = False
                lines.append(orjson.dumps(body) + b"\n")
                record += 1
            raw.write(lz4.frame.compress(b"".join(lines), compression_level=1))
    return record


def records_of(path):
    with lz4.frame.open(path, "rb") as handle:
        return handle.read()


def run(work_dir, frames, per_frame, workers, with_field=False):
    """Returns (verdict count, decompressed bytes) for one configuration."""
    order, common = build_corpus(work_dir)
    path = os.path.join(work_dir, f"alignments_{frames}_{workers}.lz4")
    total = write_alignments(path, frames, per_frame, with_field=with_field)
    found = bf.banality_auto_detect(path, common, order, total, THRESHOLD, workers)
    return found, records_of(path), total


def write_phrases(path):
    """A phrase list: one of these appears in every third synthetic passage."""
    with open(path, "w", encoding="utf8") as handle:
        handle.write("banal phrase here\n")
        handle.write("another stock phrase\n")
    return path


def write_flagged(path, frames, per_frame, documents=6):
    """Like write_alignments but with a banality verdict already decided, which is the
    state separate_banalities runs against."""
    record = 0
    with open(path, "wb") as raw:
        for _ in range(frames):
            lines = []
            for _ in range(per_frame):
                start = (record % 30) * 10
                lines.append(orjson.dumps({
                    **sides(record, documents, start),
                    "source_passage": f"passage {record}",
                    "banality": record % 3 == 0,
                }) + b"\n")
                record += 1
            raw.write(lz4.frame.compress(b"".join(lines), compression_level=1))
    return record


def write_phrasey(path, frames, per_frame, documents=6):
    """Passages where every third one carries a listed phrase."""
    record = 0
    with open(path, "wb") as raw:
        for _ in range(frames):
            lines = []
            for _ in range(per_frame):
                start = (record % 30) * 10
                text = ("a banal phrase here and more" if record % 3 == 0
                        else f"ordinary passage {record}")
                lines.append(orjson.dumps({
                    **sides(record, documents, start),
                    "source_passage": text,
                }) + b"\n")
                record += 1
            raw.write(lz4.frame.compress(b"".join(lines), compression_level=1))
    return record


def second_file(path, name):
    """The other file a two-output pass writes, beside the alignments."""
    return path.replace("alignments.jsonl", name)


def run_filter(work_dir, frames, per_frame, workers):
    order, common = build_corpus(work_dir)
    path = os.path.join(work_dir, "alignments.jsonl.lz4")
    total = write_phrasey(path, frames, per_frame)
    phrases = write_phrases(os.path.join(work_dir, "phrases.txt"))
    counts = bf.filter_and_flag(path, phrases, common, order, total, THRESHOLD, workers)
    return counts, records_of(path), records_of(second_file(path, "filtered_passages.jsonl"))


def run_phrase(work_dir, frames, per_frame, workers):
    build_corpus(work_dir)
    path = os.path.join(work_dir, "alignments.jsonl.lz4")
    total = write_phrasey(path, frames, per_frame)
    phrases = write_phrases(os.path.join(work_dir, "phrases.txt"))
    filtered = bf.phrase_matcher(path, phrases, total, workers)
    return filtered, records_of(path), records_of(second_file(path, "filtered_passages.jsonl"))


def run_separate(work_dir, frames, per_frame, workers):
    path = os.path.join(work_dir, "alignments.jsonl.lz4")
    total = write_flagged(path, frames, per_frame)
    separated = bf.separate_banalities(path, total, workers)
    return separated, records_of(path), records_of(second_file(path, "banal_alignments.jsonl"))


def has_field(blob, field=b'"banality"'):
    """True if every record carries the field, False if none does."""
    lines = [line for line in blob.split(b"\n") if line.strip()]
    hits = sum(field in line for line in lines)
    return hits == len(lines) if hits else False


def check_combinations(root):
    """Each detector on its own, both together, and banalities kept or separated."""
    workers = 8

    # Phrase filter alone: hits go to the filtered file, the rest are kept untouched.
    work = os.path.join(root, "combo_phrase")
    os.makedirs(work, exist_ok=True)
    build_corpus(work)
    path = os.path.join(work, "alignments.jsonl.lz4")
    total = write_phrasey(path, 6, 120)
    phrases = write_phrases(os.path.join(work, "phrases.txt"))
    filtered = bf.phrase_matcher(path, phrases, total, workers)
    kept = records_of(path)
    check("phrase filter alone: hits are filtered out", filtered == total // 3)
    check("phrase filter alone: the rest are kept",
          kept.count(b"\n") == total - filtered)
    check("phrase filter alone: no banality verdict is added", not has_field(kept))

    # Auto-detection alone: every record gets a verdict, nothing is filtered.
    work = os.path.join(root, "combo_auto")
    os.makedirs(work, exist_ok=True)
    order, common = build_corpus(work)
    path = os.path.join(work, "alignments.jsonl.lz4")
    total = write_phrasey(path, 6, 120)
    found = bf.banality_auto_detect(path, common, order, total, THRESHOLD, workers)
    kept = records_of(path)
    check("auto-detection alone: nothing is removed", kept.count(b"\n") == total)
    check("auto-detection alone: every record carries a verdict", has_field(kept))
    check("auto-detection alone: some records are banal, not all", 0 < found < total)
    check("auto-detection alone: no filtered file is written",
          not os.path.exists(second_file(path, "filtered_passages.jsonl")))

    # Both: phrase hits leave, and everything kept carries a verdict.
    work = os.path.join(root, "combo_both")
    os.makedirs(work, exist_ok=True)
    order, common = build_corpus(work)
    path = os.path.join(work, "alignments.jsonl.lz4")
    total = write_phrasey(path, 6, 120)
    phrases = write_phrases(os.path.join(work, "phrases.txt"))
    filtered, found = bf.filter_and_flag(path, phrases, common, order, total,
                                         THRESHOLD, workers)
    kept = records_of(path)
    check("both: phrase hits are filtered out", filtered == total // 3)
    check("both: what is kept carries a verdict", has_field(kept))
    check("both: the filtered file holds the hits",
          records_of(second_file(path, "filtered_passages.jsonl")).count(b"\n") == filtered)

    # store_banalities off: the flagged records move to their own file.
    work = os.path.join(root, "combo_separate")
    os.makedirs(work, exist_ok=True)
    path = os.path.join(work, "alignments.jsonl.lz4")
    total = write_flagged(path, 6, 120)
    separated = bf.separate_banalities(path, total, workers)
    kept = records_of(path)
    banal = records_of(second_file(path, "banal_alignments.jsonl"))
    check("store_banalities off: the banal records are separated",
          separated == banal.count(b"\n") > 0)
    check("store_banalities off: they are gone from the alignments",
          kept.count(b"\n") == total - separated)
    check("store_banalities off: nothing is lost",
          kept.count(b"\n") + banal.count(b"\n") == total)

    # store_banalities on: separate_banalities is simply not called, so the verdicts
    # stay in the alignments for the database to flag.
    work = os.path.join(root, "combo_keep")
    os.makedirs(work, exist_ok=True)
    path = os.path.join(work, "alignments.jsonl.lz4")
    total = write_flagged(path, 6, 120)
    before = records_of(path)
    check("store_banalities on: the flagged records stay put",
          before.count(b"\n") == total and has_field(before))
    check("store_banalities on: no separate file is written",
          not os.path.exists(second_file(path, "banal_alignments.jsonl")))


def check_scoring(root):
    """The verdict: mean commonness over both sides, 0 below three documents."""
    work = os.path.join(root, "scoring")
    order = os.path.join(work, "ngrams_in_order")
    os.makedirs(order, exist_ok=True)
    # keys 1..4 in 2 documents, 11..14 in 3, 21..24 in 1,000 -- of 1,000
    write_frequencies(os.path.join(work, "document_frequencies.bin"), 1000,
                      {**{k: 2 for k in range(1, 5)}, **{k: 3 for k in range(11, 15)},
                       **{k: 1000 for k in range(21, 25)}})
    for name, keys in (("rare", [1, 2, 3, 4]), ("three", [11, 12, 13, 14]),
                       ("everywhere", [21, 22, 23, 24]), ("half", [1, 2, 21, 22])):
        write_ngram_doc(os.path.join(order, f"{name}.bin"), 4, np.array(keys))
    common = bf.load_commonness(os.path.join(work, "document_frequencies.bin"))
    check("scoring: a key in two documents counts for nothing",
          common.total(np.array([1, 2], dtype=np.int64)) == 0.0)
    check("scoring: a key in three documents barely counts",
          0.0 < common.total(np.array([11], dtype=np.int64)) < 0.1)
    check("scoring: a key in every document counts fully",
          abs(common.total(np.array([21], dtype=np.int64)) - 1.0) < 1e-6)
    check("scoring: an unknown key counts for nothing",
          common.total(np.array([999], dtype=np.int64)) == 0.0)

    class Passage:
        def __init__(self, source, target):
            self.source_ngrams, self.source_start_byte, self.source_end_byte = source, 0, 40
            self.target_ngrams, self.target_start_byte, self.target_end_byte = target, 0, 40

    score = bf.banality_score(os.path.join(work, "document_frequencies.bin"), order, 0.4)
    check("scoring: common on both sides is banal",
          score.is_banal(Passage("everywhere.bin", "everywhere.bin")))
    check("scoring: rare on both sides is not",
          not score.is_banal(Passage("rare.bin", "three.bin")))
    # 1.0 on the source and 0 on the target: banal from the source alone, not from both.
    stricter = bf.banality_score(os.path.join(work, "document_frequencies.bin"), order, 0.6)
    check("scoring: both sides count, not only the source",
          not stricter.is_banal(Passage("everywhere.bin", "rare.bin")))
    check("scoring: the verdict does not depend on which side is the source",
          score.is_banal(Passage("half.bin", "everywhere.bin"))
          == score.is_banal(Passage("everywhere.bin", "half.bin")))


def main():
    root = tempfile.mkdtemp(prefix="textpair_banality_")
    try:
        check_scoring(root)
        # The frame layouts that matter: one frame cannot be split, many frames split
        # unevenly, and more workers than frames must not lose or duplicate a range.
        for frames, per_frame in ((1, 500), (7, 300), (64, 50), (5, 1)):
            serial = run(root, frames, per_frame, 1)
            for workers in (2, 8, 96):
                parallel = run(root, frames, per_frame, workers)
                check(f"{frames} frame(s) x {per_frame}, {workers} workers: "
                      f"same records as serial", serial[1] == parallel[1])
                check(f"{frames} frame(s) x {per_frame}, {workers} workers: "
                      f"same verdict count ({serial[0]})", serial[0] == parallel[0])
                check(f"{frames} frame(s) x {per_frame}, {workers} workers: "
                      f"every record kept ({serial[2]})",
                      parallel[1].count(b"\n") == serial[2])

        # A record that already carries the field takes the decode-and-re-encode path
        # rather than the splice, in both passes.
        serial = run(root, 9, 100, 1, with_field=True)
        parallel = run(root, 9, 100, 16, with_field=True)
        check("existing banality field: same records as serial", serial[1] == parallel[1])
        check("existing banality field: same verdict count", serial[0] == parallel[0])

        # The other three passes over the file, each writing two outputs. A real config
        # reaches filter_and_flag rather than banality_auto_detect, and separate_banalities
        # runs after it whenever store_banalities is off, so both have to hold too.
        for label, runner in (("filter_and_flag", run_filter),
                              ("phrase_matcher", run_phrase),
                              ("separate_banalities", run_separate)):
            serial = runner(root, 9, 200, 1)
            for workers in (3, 12, 96):
                parallel = runner(root, 9, 200, workers)
                check(f"{label}, {workers} workers: same kept records",
                      serial[1] == parallel[1])
                check(f"{label}, {workers} workers: same second file",
                      serial[2] == parallel[2])
                check(f"{label}, {workers} workers: same counts {serial[0]}",
                      serial[0] == parallel[0])

        # A pass that matches nothing must still leave readable files: an empty file is
        # not an lz4 stream, and a phrase list that hits nothing is entirely normal.
        for workers in (1, 8):
            empty = os.path.join(root, "nomatch")
            os.makedirs(empty, exist_ok=True)
            build_corpus(empty)
            path = os.path.join(empty, "alignments.jsonl.lz4")
            total = write_alignments(path, 4, 50)
            phrases = os.path.join(empty, "phrases.txt")
            with open(phrases, "w", encoding="utf8") as handle:
                handle.write("nothing here matches this phrase at all\n")
            filtered = bf.phrase_matcher(path, phrases, total, workers)
            second = second_file(path, "filtered_passages.jsonl")
            check(f"no phrase matches, {workers} worker(s): nothing filtered",
                  filtered == 0)
            check(f"no phrase matches, {workers} worker(s): every record kept",
                  records_of(path).count(b"\n") == total)
            check(f"no phrase matches, {workers} worker(s): empty file still readable",
                  records_of(second) == b"")
            shutil.rmtree(empty, ignore_errors=True)

        # The four ways the two detectors can be configured, and what each is supposed to
        # leave behind. textpair/__main__ picks between these on `phrase_filter` and
        # `banality_auto_detection`, and either has to work without the other.
        check_combinations(root)

        # Frame boundaries have to be found without decompressing anything.
        path = os.path.join(root, "bounds.lz4")
        written = write_alignments(path, 11, 40)
        bounds = bf.frame_bounds(path)
        check("frame_bounds finds every frame", len(bounds) == 11)
        check("frame_bounds accounts for every byte",
              sum(length for _, length in bounds) == os.path.getsize(path))
        check("frame_bounds ranges are contiguous",
              all(bounds[i][0] + bounds[i][1] == bounds[i + 1][0]
                  for i in range(len(bounds) - 1)))
        # Each range has to decode on its own, which is what lets a worker take one.
        middle = bounds[5]
        with bf._lz4_window(path, middle[0], middle[1]) as handle:
            check("a single frame range decodes alone",
                  handle.read().count(b"\n") == 40)
        check("the file is still readable as one stream",
              records_of(path).count(b"\n") == written)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("all checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
