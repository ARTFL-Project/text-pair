#!/usr/bin/env python3
"""Regenerate the synthetic aligner fixtures in this directory.

Each fixture is a miniature corpus (ngrams/, metadata/metadata.json, text/) covering a
metadata or text-file variant that the real corpora do not, and that once produced a
wrong record: metadata without start_byte/end_byte, non-string metadata values, a
missing text file, and an ngram file with no metadata entry at all. Metadata filenames
are relative, so both aligners must run with the fixture directory as the cwd.
"""
import os

import orjson

# Tokens 0..9 are the words an ngram can cover; markup and entities sit between them so
# the text also exercises tag stripping and unescaping.
TOKENS = ["alpha", "beta", "<i>gamma</i>", "delta", "epsilon&amp;", "zeta", "eta",
          "theta", "iota ", "kappa"]
SHARED = [100001, 100002, 100003, 100004, 100005, 100006, 100007, 100008]


def build_text(doc_id):
    """Document text plus the byte offset and length of every token."""
    head = f"<div id='{doc_id}'>Fixture document {doc_id}. "
    parts = [head]
    offsets = []
    for token in TOKENS:
        offsets.append((sum(len(p.encode()) for p in parts), len(token.encode())))
        parts.append(token + " ")
    parts.append("</div>\n")
    return "".join(parts), offsets


def ngrams(offsets, hashes, unique_base, n_unique=12):
    """Trigram-shaped positions: ngram i spans tokens i..i+2. The unique keys keep the
    shared fraction under duplicate_threshold, so the pair is compared and not dismissed."""
    doc = {}
    for index, key in enumerate(hashes):
        start = offsets[index][0]
        end = offsets[index + 2][0] + offsets[index + 2][1]
        doc.setdefault(str(key), []).append([index, start, end])
    for i in range(n_unique):
        index = len(hashes) + i
        token = offsets[i % (len(offsets) - 2)]
        doc[str(unique_base + i)] = [[index, token[0], token[0] + token[1]]]
    return doc


def write_corpus(name, docs, metadata, missing_text=()):
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    for sub in ("ngrams", "metadata", "text"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    for doc_id, (text, index) in docs.items():
        with open(os.path.join(root, "ngrams", f"{doc_id}.json"), "wb") as output:
            output.write(orjson.dumps(index))
        if doc_id not in missing_text:
            with open(os.path.join(root, "text", f"{doc_id}.txt"), "w", encoding="utf8") as out:
                out.write(text)
    with open(os.path.join(root, "metadata", "metadata.json"), "wb") as output:
        output.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))


def corpus(doc_ids):
    """Documents 1 and 2 share SHARED, every other document shares nothing."""
    docs = {}
    for position, doc_id in enumerate(doc_ids):
        text, offsets = build_text(doc_id)
        base = 1000 + position * 100
        keys = SHARED if position < 2 else [base + i for i in range(len(SHARED))]
        docs[doc_id] = (text, ngrams(offsets, keys, base + 50))
    return docs


def base_meta(doc_id, year):
    return {"title": f"Fixture {doc_id}", "author": f"Author {doc_id}", "year": year,
            "philo_id": f"{doc_id} 0 0 0 0 0 0", "filename": f"text/{doc_id}.txt",
            "start_byte": "0", "end_byte": "260", "word_count": "10"}


def main():
    # 1. No start_byte/end_byte: the relative positions become Go's "+Inf".
    docs = corpus(["1", "2", "3"])
    meta = {}
    for doc_id, year in (("1", "1750"), ("2", "1751"), ("3", "1752")):
        fields = base_meta(doc_id, year)
        del fields["start_byte"], fields["end_byte"]
        meta[doc_id] = fields
    write_corpus("no_byte_range", docs, meta)

    # 2. Non-string metadata values: Go unmarshals into map[string]string and keeps the
    # zero value, so every one of these must come out as "".
    docs = corpus(["1", "2", "3"])
    meta = {doc_id: base_meta(doc_id, year)
            for doc_id, year in (("1", "1750"), ("2", "1751"), ("3", "1752"))}
    meta["1"]["word_count"] = 1234
    meta["1"]["author"] = None
    meta["1"]["keywords"] = ["a", "b"]
    meta["2"]["extent"] = {"pages": 12}
    meta["2"]["title"] = 42
    meta["3"]["author"] = True
    write_corpus("non_string_meta", docs, meta)

    # 3. Missing text files: every passage and context is "".
    docs = corpus(["1", "2", "3"])
    meta = {doc_id: base_meta(doc_id, year)
            for doc_id, year in (("1", "1750"), ("2", "1751"), ("3", "1752"))}
    meta["1"]["filename"] = "text/gone.txt"
    meta["2"]["filename"] = "text/also_gone.txt"
    write_corpus("missing_text", docs, meta, missing_text=("1", "2"))

    # 4. An ngram file with no metadata entry: no source_* fields at all.
    docs = corpus(["1", "2", "3"])
    meta = {doc_id: base_meta(doc_id, year) for doc_id, year in (("2", "1751"), ("3", "1752"))}
    write_corpus("no_metadata", docs, meta)


if __name__ == "__main__":
    main()
