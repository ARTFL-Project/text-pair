#!/usr/bin/env python3
"""Checks the PhiloLogic reader: text-object grouping, metadata, byte spans.

    test_reader.py

Builds a small PhiloLogic database in a temporary directory rather than carrying
a fixture, so the JSON and toms.db shapes it asserts against are visible here.
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile

import lz4.frame

from textpair.preprocessing import PreProcessor

FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


TOMS_COLUMNS = ("philo_type", "philo_id", "philo_name", "filename", "author", "title",
                "year", "word_count", "next", "prev", "head")


def build_db(root, words, rows):
    """Write data/words_and_philo_ids/1.lz4 and data/toms.db."""
    data = os.path.join(root, "data")
    os.makedirs(os.path.join(data, "words_and_philo_ids"), exist_ok=True)
    os.makedirs(os.path.join(data, "TEXT"), exist_ok=True)
    path = os.path.join(data, "words_and_philo_ids", "1.lz4")
    with lz4.frame.open(path, mode="wb") as handle:
        for word in words:
            handle.write((json.dumps(word) + "\n").encode("utf8"))
    connection = sqlite3.connect(os.path.join(data, "toms.db"))
    connection.execute(f"CREATE TABLE toms ({', '.join(TOMS_COLUMNS)})")
    for row in rows:
        connection.execute(
            f"INSERT INTO toms ({', '.join(TOMS_COLUMNS)}) VALUES ({','.join('?' * len(TOMS_COLUMNS))})",
            tuple(row.get(column, "") for column in TOMS_COLUMNS),
        )
    connection.commit()
    connection.close()
    return path


def word(token, position, start, philo_type="word"):
    return {"token": token, "position": position, "start_byte": start,
            "end_byte": start + len(token), "philo_type": philo_type}


# Two paragraphs, two sentences each, inside one div1 of one document.
WORDS = [
    word("alpha",   "1 1 0 0 1 1 1", 0),
    word("bravo",   "1 1 0 0 1 1 2", 10),
    word(".",       "1 1 0 0 1 1 3", 20, philo_type="punct"),
    word("charlie", "1 1 0 0 1 2 1", 30),
    word("delta",   "1 1 0 0 1 2 2", 40),
    word("echo",    "1 1 0 0 2 1 1", 50),
    word("foxtrot", "1 1 0 0 2 1 2", 60),
]
ROWS = [
    {"philo_type": "doc", "philo_id": "1 0 0 0 0 0 0", "filename": "one.xml",
     "author": "Author", "title": "Title", "year": "1789", "next": ""},
    {"philo_type": "div1", "philo_id": "1 1 0 0 0 0 0", "head": "Chapter", "next": ""},
    {"philo_type": "para", "philo_id": "1 1 0 0 1 0 0", "word_count": "5"},
    {"philo_type": "para", "philo_id": "1 1 0 0 2 0 0", "word_count": "2"},
]


def run(root, **kwargs):
    path = build_db(root, WORDS, ROWS)
    preproc = PreProcessor(workers=1, language="french", stemmer=False, modernize=False,
                           min_word_length=2, **kwargs)
    return list(preproc.process_texts([path], keep_all=kwargs.pop("_keep_all", False),
                                      keep_surface=True))


def test_doc_level_is_one_object():
    with tempfile.TemporaryDirectory() as root:
        objects = run(root, text_object_type="doc")
        check("one object", len(objects), 1)
        check("all surviving tokens", objects[0].forms,
              ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"])
        check("first start byte", objects[0].start_bytes[0], 0)
        check("last end byte", objects[0].end_bytes[-1], 67)
        check("doc metadata author", objects[0].metadata["author"], "Author")
        check("doc metadata title", objects[0].metadata["title"], "Title")
        check("philo_doc_id", objects[0].metadata["philo_doc_id"], "1")


def test_para_level_splits_on_the_paragraph():
    with tempfile.TemporaryDirectory() as root:
        objects = run(root, text_object_type="para")
        check("two paragraphs", len(objects), 2)
        check("first paragraph", objects[0].forms, ["alpha", "bravo", "charlie", "delta"])
        check("second paragraph", objects[1].forms, ["echo", "foxtrot"])
        check("paragraph word_count from toms", objects[0].metadata["word_count"], "5")
        check("inherits the document author", objects[1].metadata["author"], "Author")


def test_sent_level_and_punctuation_reattachment():
    """A punct token belongs to the sentence it closes, not the next one."""
    with tempfile.TemporaryDirectory() as root:
        objects = run(root, text_object_type="sent")
        check("three sentences", len(objects), 3)
        check("first sentence", objects[0].forms, ["alpha", "bravo"])
        check("second sentence", objects[1].forms, ["charlie", "delta"])
        check("third sentence", objects[2].forms, ["echo", "foxtrot"])
        # The '.' carries position "1 1 0 0 1 1 3" but philo_type punct, so it is
        # rewritten onto the sentence just closed -- raw_length counts it there.
        check("punct counted in the sentence it closes", objects[0].metadata["word_count"], 3)
        check("sentence philo_type", objects[0].metadata["philo_type"], "sent")
        check("sentence philo_id", objects[0].metadata["philo_id"], "1 1 0 0 1 1 0")
        # Sentences have no toms row, so no philo_sent_id is invented.
        if "philo_sent_id" in objects[0].metadata:
            FAILURES.append("philo_sent_id was invented for a level with no toms row")


def test_sentence_span_covers_filtered_tokens():
    """The sentence's extent is its raw extent, not its surviving tokens' extent."""
    with tempfile.TemporaryDirectory() as root:
        objects = run(root, text_object_type="sent")
        # '.' is dropped by normalization but still ends the first sentence at 21.
        check("raw start", objects[0].metadata["start_byte"], 0)
        check("raw end includes the punct", objects[0].metadata["end_byte"], 21)


def test_keep_all_leaves_placeholders():
    with tempfile.TemporaryDirectory() as root:
        path = build_db(root, WORDS, ROWS)
        preproc = PreProcessor(workers=1, language="french", stemmer=False,
                               modernize=False, text_object_type="doc")
        kept = list(preproc.process_texts([path], keep_all=True, keep_surface=True))[0]
        check("every token has a slot", len(kept.forms), len(WORDS))
        check("the punct slot is empty", kept.forms[2], "")
        check("surface form retained", kept.surface_forms[2], ".")
        check("byte offsets still aligned", kept.start_bytes[2], 20)
        dropped = list(preproc.process_texts([path], keep_all=False, keep_surface=True))[0]
        check("without keep_all the slot is gone", len(dropped.forms), len(WORDS) - 1)


def test_object_with_no_surviving_tokens_is_still_reported():
    """It existed, so it keeps its metadata entry -- as it did before."""
    words = [word("a", "1 1 0 0 1 1 1", 0), word("b", "1 1 0 0 1 1 2", 10),
             word("alpha", "1 1 0 0 1 2 1", 20)]
    with tempfile.TemporaryDirectory() as root:
        path = build_db(root, words, ROWS)
        preproc = PreProcessor(workers=1, language="french", stemmer=False, modernize=False,
                               min_word_length=2, text_object_type="sent")
        objects = list(preproc.process_texts([path], keep_surface=True))
        check("both sentences reported", len(objects), 2)
        check("the all-filtered sentence is empty", objects[0].forms, [])
        check("but counted its words", objects[0].metadata["word_count"], 2)


def test_metadata_does_not_depend_on_sibling_order():
    """The old per-level cache stored only fields assigned when it was built."""
    rows = list(ROWS) + [
        {"philo_type": "div1", "philo_id": "1 2 0 0 0 0 0", "head": "Two", "next": "1 3"},
    ]
    words = WORDS + [word("golf", "1 2 0 0 1 1 1", 70)]
    with tempfile.TemporaryDirectory() as root:
        path = build_db(root, words, rows)
        preproc = PreProcessor(workers=1, language="french", stemmer=False,
                               modernize=False, text_object_type="div1")
        objects = list(preproc.process_texts([path], keep_surface=True))
        check("two div1 objects", len(objects), 2)
        # The first div1's own `next` is empty; it must still be present, taking
        # the document's value, whichever sibling was processed first.
        for index, obj in enumerate(objects):
            if "next" not in obj.metadata:
                FAILURES.append(f"div1 {index} lost the `next` field")


def main():
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    if FAILURES:
        print(f"test_reader: {len(FAILURES)} failure(s)")
        for failure in FAILURES:
            print(f"  {failure}")
        return 1
    print("test_reader: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
