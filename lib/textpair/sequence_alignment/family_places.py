"""Each passage family's passage projected onto the places that reuse it.

A place is where a document other than the anchor's has the family's passage: the ends of
the family's alignments in one file, overlapping ones taken together, as the group page
counts them. Those ends come from alignments with the passage itself or with another reuse,
so they may hold more than the passage, or less. Each place's text, with text around it, is
compared with the family's passage letter by letter (textpair.shared_text, the group page's
comparison), and the place's passage becomes what it shares with the family's in the same
order, in whole words, with its context as the alignments have it.
"""
import mmap
import multiprocessing as mp
import os
from array import array
from typing import Union

import lz4.frame
import msgspec
import numpy as np
import orjson
from numba import njit
from tqdm import tqdm

from textpair.shared_text import Passage

from .aligner import extension
from .aligner.output import alignment_to_text

MARGIN = 600  # bytes read on each side of a place, at least: as far as its passage may grow
CHUNK = 500  # families a worker projects at a time
_MAPS = 256  # documents a worker keeps mapped
_READ_SIZE = 1 << 24


class _Record(msgspec.Struct):
    group_id: list[int] = []
    source_filename: str = ""
    source_doc_id: Union[str, int, None] = None
    source_start_byte: int = 0
    source_end_byte: int = 0
    target_filename: str = ""
    target_doc_id: Union[str, int, None] = None
    target_start_byte: int = 0
    target_end_byte: int = 0


class _Group(msgspec.Struct):
    group_id: int
    source_filename: str
    source_doc_id: Union[str, int, None] = None
    source_start_byte: int = 0
    source_end_byte: int = 0


@njit(cache=True)
def _places(groups, files, starts, ends):
    """Ends sorted by family, file and start, overlapping ones of a family in a file taken together."""
    n = groups.shape[0]
    pg, pf = np.empty(n, np.int64), np.empty(n, np.int32)
    ps, pe = np.empty(n, np.int64), np.empty(n, np.int64)
    k = -1
    for i in range(n):
        if k >= 0 and groups[i] == pg[k] and files[i] == pf[k] and starts[i] < pe[k]:
            pe[k] = max(pe[k], ends[i])
        else:
            k += 1
            pg[k], pf[k], ps[k], pe[k] = groups[i], files[i], starts[i], ends[i]
    return pg[:k + 1], pf[:k + 1], ps[:k + 1], pe[:k + 1]


def _lines(path):
    with lz4.frame.open(path) as handle:
        remainder = b""
        while True:
            chunk = handle.read(_READ_SIZE)
            if not chunk:
                break
            lines = (remainder + chunk).split(b"\n")
            remainder = lines.pop()
            yield from lines
        if remainder:
            yield remainder


def _read_places(results_file, families):
    """Every family's places: (family, file, start, end) arrays, and the file names."""
    names, ids = [], {}
    groups, files, starts, ends = array("q"), array("i"), array("q"), array("q")
    decode = msgspec.json.Decoder(_Record).decode
    for line in _lines(results_file):
        if not line:
            continue
        r = decode(line)
        for side in ((r.source_filename, r.source_doc_id, r.source_start_byte, r.source_end_byte),
                     (r.target_filename, r.target_doc_id, r.target_start_byte, r.target_end_byte)):
            name, doc, start, end = side
            for family in r.group_id:
                anchor = families.get(family)
                if anchor is None or str(doc) == anchor:
                    continue  # the anchor's own document holds the passage, not a reuse of it
                f = ids.get(name)
                if f is None:
                    f = ids[name] = len(names)
                    names.append(name)
                groups.append(family)
                files.append(f)
                starts.append(start)
                ends.append(end)
    groups, files = np.frombuffer(groups, np.int64), np.frombuffer(files, np.int32)
    starts, ends = np.frombuffer(starts, np.int64), np.frombuffer(ends, np.int64)
    order = np.lexsort((starts, files, groups))
    return _places(groups[order], files[order], starts[order], ends[order]), names


def _whole_words(kinds, shared, first, last):
    """first and last moved to word boundaries: out over a word at an edge when half its
    letters are shared or more, in past it otherwise, and in past what is no word."""
    def word(at):
        lo = hi = at
        while lo > 0 and kinds[lo - 1] == 1:
            lo -= 1
        while hi < kinds.shape[0] and kinds[hi] == 1:
            hi += 1
        return lo, hi
    if kinds[first] == 1:
        lo, hi = word(first)
        first = lo if 2 * shared[lo:hi].sum() >= hi - lo else hi
    if kinds[last - 1] == 1:
        lo, hi = word(last - 1)
        last = hi if 2 * shared[lo:hi].sum() >= hi - lo else lo
    while first < last and kinds[first] == 0:
        first += 1
    while last > first and kinds[last - 1] == 0:
        last -= 1
    return first, last


class _Reader:
    """Documents, mapped and kept: a worker reads each one many times."""

    def __init__(self):
        self.maps = {}

    def __call__(self, name):
        buf = self.maps.get(name)
        if buf is None:
            if len(self.maps) >= _MAPS:
                self.maps.clear()  # not closed: a family being projected may still hold them
            try:
                with open(name, "rb") as handle:
                    buf = mmap.mmap(handle.fileno(), 0, prot=mmap.PROT_READ)
            except (OSError, ValueError):
                buf = b""
            self.maps[name] = buf
        return buf


def _text(buf, start, end, cut=True):
    """Folded text of buf[start:end], with each character's byte offsets and kind; with `cut`,
    start may fall inside a character or a tag."""
    codes, kinds, starts, ends = extension._text(np.frombuffer(buf[start:end], np.uint8), extension.FOLD,
                                                 extension.KIND, cut and start > 0)
    folded = codes.astype(np.uint32).tobytes().decode("utf-32-le", "surrogatepass")
    return folded, kinds, starts + start, ends + start


_STATE = {}


def _project(task):
    """One chunk of families' places, projected: JSON lines."""
    names, context = _STATE["names"], _STATE["context"]
    read = _STATE.setdefault("reader", _Reader())
    out = []
    for family, anchor_name, anchor_start, anchor_end, places in task:
        anchor = read(anchor_name)
        text, _, _, _ = _text(anchor, anchor_start, anchor_end, cut=False)
        passage = Passage(text)
        letters = np.array([c.isalnum() for c in text], np.bool_)
        margin = max(MARGIN, anchor_end - anchor_start)
        found_here = []
        for f, start, end in places:
            name = names[f]
            buf = read(name)
            shown, shared = (start, end), 0.0
            if buf:
                lo, hi = max(0, start - margin), min(len(buf), end + margin)
                other, kinds, starts, ends = _text(buf, lo, hi)
                if other:
                    # what the place shares with the passage in the passage's own order
                    found = passage.compare(other, detail=True)
                    ours = found.shared[0] & ~found.moved[0]
                    theirs = found.shared[1] & ~found.moved[1]
                    shared = float(ours[letters].mean()) if letters.any() else 0.0
                    on = np.flatnonzero(theirs & (kinds > 0))
                    if on.size:
                        first, last = _whole_words(kinds, theirs, int(on[0]), int(on[-1]) + 1)
                        if first < last:
                            shown = (int(starts[first]), int(ends[last - 1]))
            found_here.append([f, start, end, shown, shared, buf])
        # a projection off its own place may not land on another of the family's places in that
        # document, nor on what another projection took: that place keeps its span
        for k, (f, start, end, shown, shared, buf) in enumerate(found_here):
            if shown[1] <= start or shown[0] >= end:
                taken = any(other[0] == f and m != k and (other[1] < shown[1] and other[2] > shown[0]
                                                          or other[3][0] < shown[1] and other[3][1] > shown[0] and m < k)
                            for m, other in enumerate(found_here))
                if taken:
                    found_here[k][3], found_here[k][4] = (start, end), 0.0
        for f, start, end, shown, shared, buf in found_here:
            before, passage_text, after = alignment_to_text(buf, shown[0], shown[1], context) if buf else ("", "", "")
            out.append(orjson.dumps({
                "group_id": family, "filename": names[f], "start_byte": start, "end_byte": end,
                "passage_start_byte": shown[0], "passage_end_byte": shown[1], "shared": round(shared, 3),
                "context_before": before, "passage": passage_text, "context_after": after,
            }))
    return out


def project_places(results_file: str, groups_file: str, workers: int = 1, context_size: int = 300) -> str:
    """Write passage_group_places.jsonl beside the group file: one row per family place."""
    families, spans = {}, {}
    decode = msgspec.json.Decoder(_Group).decode
    with open(groups_file, "rb") as handle:
        for line in handle:
            if line.strip():
                g = decode(line)
                families[g.group_id] = str(g.source_doc_id)
                spans[g.group_id] = (g.source_filename, g.source_start_byte, g.source_end_byte)
    (pg, pf, ps, pe), names = _read_places(results_file, families)
    bounds = np.flatnonzero(np.diff(pg)) + 1 if pg.size else np.empty(0, np.int64)
    by_family = []
    for lo, hi in zip(np.concatenate(([0], bounds)), np.concatenate((bounds, [pg.size]))):
        if hi > lo:
            family = int(pg[lo])
            name, start, end = spans[family]
            by_family.append((family, name, start, end, list(zip(pf[lo:hi].tolist(), ps[lo:hi].tolist(), pe[lo:hi].tolist()))))
    by_family.sort(key=lambda item: (item[1], item[2]))  # anchors file by file, so workers keep their maps
    tasks = [by_family[k:k + CHUNK] for k in range(0, len(by_family), CHUNK)]
    places_file = os.path.join(os.path.dirname(groups_file), "passage_group_places.jsonl")
    _STATE.update(names=names, context=int(context_size))
    bar = tqdm(total=int(pg.size), desc="Projecting family passages onto their reuses", unit=" place", leave=False)
    with open(places_file, "wb") as output:
        if workers > 1 and len(tasks) > 1:
            with mp.get_context("fork").Pool(min(workers, len(tasks))) as pool:
                for lines in pool.imap(_project, tasks):
                    output.writelines(line + b"\n" for line in lines)
                    bar.update(len(lines))
        else:
            for task in tasks:
                lines = _project(task)
                output.writelines(line + b"\n" for line in lines)
                bar.update(len(lines))
    bar.close()
    _STATE.clear()
    print(f"  {int(pg.size):,} places reusing them, written to {os.path.basename(places_file)}", flush=True)
    return places_file
