"""Each passage family's passage projected onto the places that reuse it.

A place is where a document other than the anchor's has the family's passage: the ends of
the family's alignments in one file, overlapping ones taken together, as the group page
counts them. Those ends come from alignments with the passage itself or with another reuse,
so they may hold more than the passage, or less. Each place's text, with text around it, is
compared with the family's passage letter by letter (textpair.shared_text, the group page's
comparison), and the place's passage becomes what it shares with the family's in the same
order, in whole words, with its context as the alignments have it. Only the stretches of
shared text the place reaches count, those overlapping it or as near it, or each other, as
shared_text joins stretches, or as the passage has words, if more: what it shares further
off, as when a phrase of a short passage recurs a paragraph away, is no evidence of it, and
is compared no more, so that it takes none of the passage's words from the place. A place
sharing nothing else shares nothing.

The grouper (alignment_merger) builds the places, settles families on what they hold, and
has the rows written.
"""
import mmap
import multiprocessing as mp

import numpy as np
import orjson
from numba import njit
from tqdm import tqdm

from textpair.shared_text import BRIDGE, Passage

from .aligner import extension
from .aligner.output import alignment_to_text

MARGIN = 600  # bytes read on each side of a place, at least: as far as its passage may grow
CHUNK = 500  # families a worker projects at a time
ROWS = 5000  # places a worker writes at a time
_MAPS = 256  # documents a worker keeps mapped


@njit(cache=True)
def places(groups, files, starts, ends):
    """Ends sorted by family, file and start, overlapping ones of a family in a file taken
    together: the places' families, files, starts and ends, and each end's place."""
    n = groups.shape[0]
    pg, pf = np.empty(n, np.int64), np.empty(n, np.int64)
    ps, pe = np.empty(n, np.int64), np.empty(n, np.int64)
    of = np.empty(n, np.int64)
    k = -1
    for i in range(n):
        if k >= 0 and groups[i] == pg[k] and files[i] == pf[k] and starts[i] < pe[k]:
            pe[k] = max(pe[k], ends[i])
        else:
            k += 1
            pg[k], pf[k], ps[k], pe[k] = groups[i], files[i], starts[i], ends[i]
        of[i] = k
    return pg[:k + 1], pf[:k + 1], ps[:k + 1], pe[:k + 1], of


@njit(cache=True)
def same_places(pg, pf, ps, pe, qg, qf, qs, qe):
    """For each place q, the place p with the very same span, or -1; both sorted as `places` leaves them."""
    out = np.full(qg.shape[0], -1, np.int64)
    i = 0
    for j in range(qg.shape[0]):
        while i < pg.shape[0] and (pg[i] < qg[j] or pg[i] == qg[j] and (
                pf[i] < qf[j] or pf[i] == qf[j] and ps[i] < qs[j])):
            i += 1
        if i < pg.shape[0] and pg[i] == qg[j] and pf[i] == qf[j] and ps[i] == qs[j] and pe[i] == qe[j]:
            out[j] = i
    return out


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


def _reached(found, word_at, p0, p1, reach):
    """Of the stretches `found` has, those the characters p0 to p1 reach through gaps of
    `reach` words of the other text or fewer, each one reached reaching on: their (start, end)
    characters in the passage and the other text, the extent of the other's they and the place
    cover, and whether any was left out."""
    r0, r1 = p0, p1
    kept, left = [], list(found.stretches)
    while True:
        reached = []
        for stretch in left:
            b0, b1 = stretch[2], stretch[3]
            if b1 <= r0:
                gap = word_at[r0 - 1] - word_at[b1 - 1]
            elif b0 >= r1:
                gap = word_at[b0 - 1] - word_at[r1 - 1]
            else:
                gap = 0
            if gap <= reach:
                reached.append(stretch)
                r0, r1 = min(r0, b0), max(r1, b1)
        if not reached:
            return kept, (r0, r1), bool(left)
        kept += reached
        left = [stretch for stretch in left if stretch not in reached]


def _shared(passage, length, other, kinds, starts, ends, start, end):
    """What the place (bytes start to end of `other`) shares with the passage in its order, in
    the stretches it reaches: (the passage's characters, the other's)."""
    words = (kinds > 0) & np.concatenate(([True], kinds[:-1] == 0)) | (kinds == 2)
    word_at = np.cumsum(words)
    p0 = int(np.searchsorted(ends, start, side="right"))
    p1 = max(int(np.searchsorted(starts, end)), p0 + 1)
    reach = max(BRIDGE, len(passage.words))
    found = passage.compare(other, detail=True)
    kept, (r0, r1), dropped = _reached(found, word_at, p0, p1, reach)
    c0 = 0
    if dropped and kept:
        # compared again on what the place reaches alone, give or take as many words
        c0 = int(np.searchsorted(word_at, max(word_at[r0] - reach, 1)))
        c1 = int(np.searchsorted(word_at, word_at[r1 - 1] + reach + 1))
        found = passage.compare(other[c0:c1], detail=True)
        kept, _, _ = _reached(found, word_at[c0:c1] - (word_at[c0 - 1] if c0 else 0), p0 - c0, p1 - c0, reach)
    ours, theirs = np.zeros(length, np.bool_), np.zeros(len(other), np.bool_)
    for a0, a1, b0, b1 in kept:
        ours[a0:a1] = found.shared[0][a0:a1] & ~found.moved[0][a0:a1]
        theirs[c0 + b0:c0 + b1] = found.shared[1][b0:b1] & ~found.moved[1][b0:b1]
    return ours, theirs


_STATE = {}


def _project(task):
    """One chunk of families' places, projected: (place, start, end, shared) each."""
    read = _STATE.setdefault("reader", _Reader())
    out = []
    for anchor_name, anchor_start, anchor_end, family_places in task:
        text, _, _, _ = _text(read(anchor_name), anchor_start, anchor_end, cut=False)
        passage = Passage(text)
        letters = np.array([c.isalnum() for c in text], np.bool_)
        margin = max(MARGIN, anchor_end - anchor_start)
        for i, name, start, end in family_places:
            shown, shared = (start, end), 0.0
            buf = read(name)
            if buf:
                lo, hi = max(0, start - margin), min(len(buf), end + margin)
                other, kinds, starts, ends = _text(buf, lo, hi)
                if other:
                    # what the place shares with the passage in the passage's own order
                    ours, theirs = _shared(passage, len(text), other, kinds, starts, ends, start, end)
                    on = np.flatnonzero(theirs & (kinds > 0))
                    if on.size:
                        first, last = _whole_words(kinds, theirs, int(on[0]), int(on[-1]) + 1)
                        if first < last and starts[first] < end and ends[last - 1] > start:
                            shown = (int(starts[first]), int(ends[last - 1]))
                            shared = float(ours[letters].mean()) if letters.any() else 0.0
            out.append((i, shown[0], shown[1], shared))
    return out


def _rows(task):
    """One chunk of places as JSON lines, each with its passage and the text around it."""
    read, context = _STATE.setdefault("reader", _Reader()), _STATE["context"]
    out = []
    for family, name, start, end, shown_s, shown_e, shared in task:
        buf = read(name)
        before, passage, after = alignment_to_text(buf, shown_s, shown_e, context) if buf else ("", "", "")
        out.append(orjson.dumps({
            "group_id": family, "filename": name, "start_byte": start, "end_byte": end,
            "passage_start_byte": shown_s, "passage_end_byte": shown_e, "shared": round(shared, 3),
            "context_before": before, "passage": passage, "context_after": after,
        }))
    return out


def _run(work, tasks, workers, total, desc):
    """Each task's results, in order, from a pool when there are workers and tasks to share."""
    bar = tqdm(total=total, desc=desc, unit=" place", leave=False)
    try:
        if workers > 1 and len(tasks) > 1:
            with mp.get_context("fork").Pool(min(workers, len(tasks))) as pool:
                for done in pool.imap(work, tasks):
                    bar.update(len(done))
                    yield done
        else:
            for task in tasks:
                done = work(task)
                bar.update(len(done))
                yield done
    finally:
        bar.close()
        _STATE.clear()


def project(names, passages, pg, pf, ps, pe, workers=1, desc="Projecting family passages onto their reuses"):
    """Each place's passage, its family's projected onto it: start, end, and the share of the
    family's passage it holds. `passages` gives each family's (file, start, end); files are
    indices into `names`."""
    shown_s, shown_e, shared = ps.copy(), pe.copy(), np.zeros(pg.shape[0])
    p_file, p_start, p_end = passages
    bounds = np.flatnonzero(np.diff(pg)) + 1
    by_family = []
    for lo, hi in zip(np.concatenate(([0], bounds)), np.concatenate((bounds, [pg.size]))):
        if hi > lo:
            g = int(pg[lo])
            by_family.append((names[p_file[g]], int(p_start[g]), int(p_end[g]),
                              [(i, names[pf[i]], int(ps[i]), int(pe[i])) for i in range(lo, hi)]))
    by_family.sort(key=lambda item: (item[0], item[1]))  # anchors file by file, so workers keep their maps
    tasks = [by_family[k:k + CHUNK] for k in range(0, len(by_family), CHUNK)]
    for done in _run(_project, tasks, workers, int(pg.size), desc):
        for i, start, end, share in done:
            shown_s[i], shown_e[i], shared[i] = start, end, share
    return shown_s, shown_e, shared


def write_places(places_file, names, pg, pf, ps, pe, shown, workers=1, context_size=300,
                 desc="Writing family places"):
    """One JSON row per place: its span, its passage (`shown`, from `project`) and the text around it."""
    shown_s, shown_e, shared = shown
    rows = [(int(pg[i]), names[pf[i]], int(ps[i]), int(pe[i]), int(shown_s[i]), int(shown_e[i]), float(shared[i]))
            for i in range(pg.shape[0])]
    tasks = [rows[k:k + ROWS] for k in range(0, len(rows), ROWS)]
    _STATE["context"] = int(context_size)
    with open(places_file, "wb") as output:
        for lines in _run(_rows, tasks, workers, len(rows), desc):
            output.writelines(line + b"\n" for line in lines)
    return len(rows)
