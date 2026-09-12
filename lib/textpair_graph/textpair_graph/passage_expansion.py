"""Expand short alignment passages with their neighbouring sentences.

Bare short passages are tags, incipits and word lists an embedder cannot place.

Sentence boundaries come from PhiloLogic's words_and_philo_ids dumps, whose
`position` field carries `doc div1 div2 div3 para sent word ...`; text is then
sliced from the original TEI by byte offset, since those tokens are lowercased
with punctuation split off.

Modelled on textpair.vector_space_alignment.expansion rather than importing it:
that module reads a cache written only during a VSA parse.
"""

import os
from collections import defaultdict
from html import unescape as unescape_html
from xml.sax.saxutils import unescape as unescape_xml

import lz4.frame
import orjson
import regex as re
from tqdm import tqdm

# Word floor for embedding a passage as-is.
MIN_PASSAGE_WORDS = 25

# Character floor, for scripts that do not delimit words with spaces.
MIN_PASSAGE_CHARS = 120


def _long_enough(text: str, min_words: int) -> bool:
    return len(text.split()) >= min_words or len(text.strip()) >= MIN_PASSAGE_CHARS

# Leading `position` fields a neighbouring sentence must share: doc div1 div2
# div3. Not the paragraph -- in verse every line is its own.
BOUNDARY_FIELDS = 4

TAGS = re.compile(r"<[^>]+>")


def _clean_text(text: str) -> str:
    """Vendored from textpair.utils.clean_text."""
    text = TAGS.sub("", text)
    text = unescape_xml(text)
    text = unescape_html(text)
    return text.replace("\n", " ").strip()


def _get_text(start_byte: int, end_byte: int, filename: str) -> str:
    """Vendored from textpair.utils.get_text.

    The leading/trailing rules matter because an arbitrary byte range can begin
    or end in the middle of a tag, which TAGS alone would not remove.
    """
    if start_byte < 0:
        start_byte = 0
    with open(filename, "rb") as text_file:
        text_file.seek(start_byte)
        text = text_file.read(end_byte - start_byte).decode("utf8", "ignore")
    if text.startswith("<"):
        text = re.sub(r"^<[^>]+>", "", text, count=1).strip()
    if text.endswith(">"):
        text = re.sub(r"<[^>]+>$", "", text, count=1).strip()
    text = re.sub(r"<[^>]+$", "", text).strip()
    return _clean_text(text)


def _sentence_spans(words_path: str) -> tuple[list[list[str]], list[int], list[int]]:
    """Sentences in one document as (id fields, start byte, end byte) columns."""
    ids: list[list[str]] = []
    starts: list[int] = []
    ends: list[int] = []
    with lz4.frame.open(words_path, "rb") as handle:
        for line in handle:
            token = orjson.loads(line)
            fields = token["position"].split()[:6]
            start = token["start_byte"]
            end = token["end_byte"]
            if ids and ids[-1] == fields:
                if end > ends[-1]:
                    ends[-1] = end
                if start < starts[-1]:
                    starts[-1] = start
            else:
                ids.append(fields)
                starts.append(start)
                ends.append(end)
    return ids, starts, ends


def _expand_one(ids, starts, ends, start_byte, end_byte, filename, min_words, original_words, original_chars):
    """Widen a byte range outward by whole sentences until it reads long enough.

    Judged against the original passage length, not against whether widening
    happened. A fragment often sits inside a sentence that is already long
    enough on its own -- measured here, 1,839 of 2,665 short passages -- and
    that enclosing sentence is the context, so returning it is the whole point.
    None means only that nothing better than the passage was found.
    """
    covering = [i for i in range(len(ids)) if starts[i] <= end_byte and ends[i] >= start_byte]
    if not covering:
        return None
    low, high = covering[0], covering[-1]
    boundary = ids[low][:BOUNDARY_FIELDS]

    def usable(i):
        return 0 <= i < len(ids) and ids[i][:BOUNDARY_FIELDS] == boundary

    def result(text):
        return text if len(text.split()) > original_words or len(text) > original_chars else None

    # After first: the sentence following a fragment is usually the author's own
    # framing of it, so it disambiguates more per word added than the one before.
    while True:
        text = _get_text(starts[low], ends[high], filename)
        if _long_enough(text, min_words):
            return result(text)
        if usable(high + 1):
            high += 1
        elif usable(low - 1):
            low -= 1
        else:
            return result(_get_text(starts[low], ends[high], filename))


def build_expansion_map(alignments_file: str, alignment_counts: int, min_words: int = MIN_PASSAGE_WORDS) -> dict:
    """Map alignment index -> expanded source text, for the short ones only.

    Grouped by source document so each words_and_philo_ids dump is read once:
    on this corpus the short passages touch 628 documents and about 2.2 GB.
    """
    pending: defaultdict[str, list] = defaultdict(list)
    missing_words_file = 0
    with lz4.frame.open(alignments_file, "rb") as handle:
        for idx, line in enumerate(handle):
            if idx >= alignment_counts:
                break
            alignment = orjson.loads(line)
            passage = alignment.get("source_passage") or ""
            if _long_enough(passage, min_words):
                continue
            filename = alignment.get("source_filename") or ""
            doc_id = alignment.get("source_doc_id") or ""
            if not filename or not doc_id:
                continue
            # .../data/TEXT/FILE.tei -> .../data/words_and_philo_ids/<doc>.lz4
            words_path = os.path.join(
                os.path.dirname(os.path.dirname(filename)), "words_and_philo_ids", f"{doc_id}.lz4"
            )
            if not os.path.exists(words_path):
                missing_words_file += 1
                continue
            pending[words_path].append(
                (
                    idx,
                    int(alignment.get("source_start_byte") or 0),
                    int(alignment.get("source_end_byte") or 0),
                    filename,
                    len(passage.split()),
                    len(passage.strip()),
                )
            )

    expanded: dict[int, str] = {}
    if not pending:
        if missing_words_file:
            print(f"  no words_and_philo_ids dumps found ({missing_words_file} passages); skipping expansion")
        return expanded

    for words_path, items in tqdm(pending.items(), desc="Expanding short passages", leave=False):
        try:
            ids, starts, ends = _sentence_spans(words_path)
        except (OSError, ValueError, KeyError):
            continue
        if not ids:
            continue
        for idx, start_byte, end_byte, filename, original_words, original_chars in items:
            try:
                text = _expand_one(
                    ids, starts, ends, start_byte, end_byte, filename, min_words,
                    original_words, original_chars,
                )
            except OSError:
                continue
            if text:
                expanded[idx] = text
    print(
        f"  expanded {len(expanded):,} of {sum(len(v) for v in pending.values()):,} short passages "
        f"across {len(pending):,} documents"
        + (f"; {missing_words_file:,} had no token dump" if missing_words_file else "")
    )
    return expanded
