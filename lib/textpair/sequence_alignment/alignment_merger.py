"""Group alignments into passage families: one per reused passage, anchored on its earliest occurrence.

Every alignment has two ends, spans of text in two documents. At each place in a
document, the ends that overlap are carved into sites, each built around the passage
most of them cover. A family is then the earliest site of a passage together with the
sites aligned directly with it.
"""

import contextlib
import multiprocessing as mp
import os
import sys
from array import array
from bisect import bisect_left
from typing import Union

import lz4.frame
import msgspec
import numpy as np
import orjson
import regex
from numba import njit
from tqdm import tqdm

from textpair.utils import clean_passage

from .banality_finder import (
    _bump,
    _concatenate,
    _Framer,
    _lz4_window,
    _run_segments,
    _SHARED,
    _work_split,
)


_STEPS = ("reading alignments", "finding passages", "forming families",
          "writing group sources", "rewriting results")


def _label(step: int, doing: str | None = None) -> str:
    return f"  {step}/{len(_STEPS)} {doing or _STEPS[step - 1]}"


def _status(text: str = "") -> None:
    """Put `text` on the progress line, for the stretches no bar covers."""
    if sys.stderr.isatty():  # in a log, each would be a stray line
        sys.stderr.write(f"\r\x1b[K{text}")
        sys.stderr.flush()


_INSIDE = 0.5    # a window joins a site when at least this much of it lies in the site's core
_FLOOR = 0.1     # ... and it is at least this long next to the core
_CONTAIN = 0.8   # a window holding this much of a core, but too long to join, is listed under it
_ADOPT = 0.5     # a site joins the family holding most of its partners once this share are placed
_SENTENCE_WORDS = 5  # words a group's passage may move back to start its sentence: never a passage
# punctuation a group's passage never moves back past: what Unicode says ends a clause or
# sentence, in any script -- not apostrophes, hyphens, quotes or brackets
_STOPS = regex.compile(r"\p{Terminal_Punctuation}")
_UNDATED = 9999
_NONE = -(1 << 62)


class _Ends(msgspec.Struct):
    """What grouping reads from a record: its two spans, and the years of their documents."""

    source_filename: str
    source_start_byte: int
    source_end_byte: int
    target_filename: str
    target_start_byte: int
    target_end_byte: int
    source_year: Union[msgspec.Raw, msgspec.UnsetType] = msgspec.UNSET
    target_year: Union[msgspec.Raw, msgspec.UnsetType] = msgspec.UNSET
    # where each document's words_and_philo_ids dump is; read once per document
    source_parsed_filename: Union[msgspec.Raw, msgspec.UnsetType] = msgspec.UNSET
    target_parsed_filename: Union[msgspec.Raw, msgspec.UnsetType] = msgspec.UNSET
    source_doc_id: Union[msgspec.Raw, msgspec.UnsetType] = msgspec.UNSET
    target_doc_id: Union[msgspec.Raw, msgspec.UnsetType] = msgspec.UNSET


_DECODE_ENDS = msgspec.json.Decoder(_Ends).decode


def _value(raw):
    return None if raw is msgspec.UNSET else orjson.loads(bytes(raw))


def _year(raw) -> int:
    """A document's year, for ordering; undated ones come last."""
    if raw is msgspec.UNSET:
        return _UNDATED
    try:
        return int(str(orjson.loads(bytes(raw)))[:4])
    except ValueError:
        return _UNDATED


def read_ends(results_file: str, count: int):
    """Every alignment's two spans: end 2k is record k's source, end 2k + 1 its target.

    Returns the file names, each file's year and (words dump, doc id), and each end's
    file, start and end.
    """
    names: list[str] = []
    ids: dict[str, int] = {}
    years: list[int] = []
    dumps: list[tuple] = []
    files, starts, ends = array("i"), array("q"), array("q")
    with lz4.frame.open(results_file) as input_file:
        progress = tqdm(total=count, desc=_label(1), leave=False)
        for batch in read_line_batches(input_file):
            for line in batch:
                r = _DECODE_ENDS(line)
                source = ids.get(r.source_filename)
                if source is None:
                    source = ids[r.source_filename] = len(names)
                    names.append(r.source_filename)
                    years.append(_year(r.source_year))
                    dumps.append((_value(r.source_parsed_filename), _value(r.source_doc_id)))
                target = ids.get(r.target_filename)
                if target is None:
                    target = ids[r.target_filename] = len(names)
                    names.append(r.target_filename)
                    years.append(_year(r.target_year))
                    dumps.append((_value(r.target_parsed_filename), _value(r.target_doc_id)))
                files.extend((source, target))
                starts.extend((r.source_start_byte, r.target_start_byte))
                ends.extend((r.source_end_byte, r.target_end_byte))
            progress.update(len(batch))
        progress.close()
    if not files:
        return [names, np.empty(0, np.int64), dumps, np.empty(0, np.int32), np.empty(0, np.int64), np.empty(0, np.int64)]
    # a list, so the grouping can take the spans over and free them once it has the windows
    return [names, np.array(years, np.int64), dumps, np.frombuffer(files, np.int32),
            np.frombuffer(starts, np.int64), np.frombuffer(ends, np.int64)]


@njit(nogil=True, cache=True)
def _windows(files, starts, ends, order):
    """Distinct spans in (file, start, end) order, each end's window, and each window's first end."""
    n_windows = 0
    for j in range(order.shape[0]):
        k, p = order[j], order[j - 1]
        if j == 0 or files[k] != files[p] or starts[k] != starts[p] or ends[k] != ends[p]:
            n_windows += 1
    window_of = np.empty(order.shape[0], np.int32)
    wf = np.empty(n_windows, np.int32)
    ws = np.empty(n_windows, np.int64)
    we = np.empty(n_windows, np.int64)
    first_end = np.empty(n_windows, np.int64)
    w = -1
    for j in range(order.shape[0]):
        k = order[j]
        if w < 0 or files[k] != wf[w] or starts[k] != ws[w] or ends[k] != we[w]:
            w += 1
            wf[w], ws[w], we[w], first_end[w] = files[k], starts[k], ends[k], k
        elif k < first_end[w]:
            first_end[w] = k
        window_of[k] = w
    return window_of, wf, ws, we, first_end


@njit(nogil=True, cache=True)
def _clusters(wf, ws, we):
    """Where each run of windows joined by overlap begins, plus the end of the last."""
    bounds = np.empty(wf.shape[0] + 1, np.int64)
    n, reach = 0, -1
    for w in range(wf.shape[0]):
        if w == 0 or wf[w] != wf[w - 1] or ws[w] >= reach:
            bounds[n] = w
            n += 1
            reach = we[w]
        elif we[w] > reach:
            reach = we[w]
    bounds[n] = wf.shape[0]
    return bounds[: n + 1].copy()


@njit(nogil=True, cache=True)
def _depth_add(mx, add, size, lo, hi, v):
    """Add v to the depth of segments [lo, hi); each node keeps its own add and its subtree's max."""
    lo += size
    hi += size
    l0, r0 = lo, hi - 1
    while lo < hi:
        if lo & 1:
            add[lo] += v
            mx[lo] += v
            lo += 1
        if hi & 1:
            hi -= 1
            add[hi] += v
            mx[hi] += v
        lo >>= 1
        hi >>= 1
    for node in (l0 >> 1, r0 >> 1):
        while node >= 1:
            mx[node] = max(mx[2 * node], mx[2 * node + 1]) + add[node]
            node >>= 1


@njit(nogil=True, cache=True)
def _deepest(mx, add, size):
    """The leftmost segment of greatest depth."""
    node = 1
    while node < size:
        node = 2 * node if mx[2 * node] == mx[node] - add[node] else 2 * node + 1
    return node - size


@njit(nogil=True, cache=True)
def _end_set(tree, size, i, value):
    i += size
    tree[i] = value
    i >>= 1
    while i >= 1:
        tree[i] = max(tree[2 * i], tree[2 * i + 1])
        i >>= 1


@njit(nogil=True, cache=True)
def _ending_past(tree, size, limit, beyond, out, stack):
    """Live windows among the first `limit` that end past `beyond`, in index order."""
    found, top = 0, 1
    stack[0] = 1
    while top:
        top -= 1
        node = stack[top]
        if tree[node] <= beyond:
            continue
        leftmost = node
        while leftmost < size:
            leftmost <<= 1
        if leftmost - size >= limit:
            continue
        if node >= size:
            out[found] = node - size
            found += 1
            continue
        stack[top] = 2 * node + 1
        stack[top + 1] = 2 * node
        top += 2
    return found


@njit(nogil=True, cache=True)
def _carve(starts, ends, bounds, inside, floor, contain):
    """Sites for every cluster of windows. Returns each window's site, each site's core and
    first member, and the (site, window) pairs of longer windows listed under a site.

    A site is built at the point most live windows cover: its core is where at least
    half of those overlap, and it takes the windows lying mostly inside the core.
    """
    site_of = np.full(starts.shape[0], -1, np.int64)
    core_s = np.empty(starts.shape[0], np.int64)
    core_e = np.empty(starts.shape[0], np.int64)
    first = np.empty(starts.shape[0], np.int64)
    held_site = np.empty(16, np.int64)
    held_window = np.empty(16, np.int64)
    n_sites = n_held = 0
    for c in range(bounds.shape[0] - 1):
        c0, c1 = bounds[c], bounds[c + 1]
        k = c1 - c0
        s, e = starts[c0:c1], ends[c0:c1]
        xs = np.unique(np.concatenate((s, e)))
        n_seg = max(xs.shape[0] - 1, 1)
        size = 1
        while size < n_seg:
            size <<= 1
        mx = np.zeros(2 * size, np.int64)
        add = np.zeros(2 * size, np.int64)
        for j in range(n_seg, size):
            mx[size + j] = _NONE
            add[size + j] = _NONE
        for node in range(size - 1, 0, -1):
            mx[node] = max(mx[2 * node], mx[2 * node + 1])
        lo = np.searchsorted(xs, s)
        hi = np.searchsorted(xs, e)
        esize = 1
        while esize < k:
            esize <<= 1
        tree = np.full(2 * esize, _NONE, np.int64)
        alive = 0
        for i in range(k):
            if e[i] > s[i]:
                _depth_add(mx, add, size, lo[i], hi[i], 1)
                tree[esize + i] = e[i]
                alive += 1
            else:  # an empty span covers nothing: a site of its own
                site_of[c0 + i] = n_sites
                core_s[n_sites], core_e[n_sites], first[n_sites] = s[i], e[i], c0 + i
                n_sites += 1
        for node in range(esize - 1, 0, -1):
            tree[node] = max(tree[2 * node], tree[2 * node + 1])
        out = np.empty(k, np.int64)
        stack = np.empty(2 * esize + 2, np.int64)
        while alive:
            point = xs[_deepest(mx, add, size)]
            g = _ending_past(tree, esize, np.searchsorted(s, point, side="right"), point, out, stack)
            need = (g + 1) // 2
            cs = np.sort(s[out[:g]])[need - 1]
            ce = np.sort(e[out[:g]])[g - need]
            core = ce - cs
            m = _ending_past(tree, esize, np.searchsorted(s, ce, side="left"), cs, out, stack)
            n_members = 0
            best, best_overlap = -1, _NONE
            for j in range(m):
                i = out[j]
                overlap = min(e[i], ce) - max(s[i], cs)
                if overlap > best_overlap:
                    best, best_overlap = i, overlap
                if overlap >= inside * (e[i] - s[i]) and (e[i] - s[i]) >= floor * core:
                    out[n_members] = i  # members fill the front of `out`, behind j
                    n_members += 1
                elif overlap >= contain * core:
                    if n_held == held_site.shape[0]:
                        held_site = np.concatenate((held_site, np.empty_like(held_site)))
                        held_window = np.concatenate((held_window, np.empty_like(held_window)))
                    held_site[n_held], held_window[n_held] = n_sites, c0 + i
                    n_held += 1
            if n_members == 0:  # nothing sits inside: the window overlapping it most stands alone
                out[0] = best
                n_members = 1
            for j in range(n_members):
                i = out[j]
                site_of[c0 + i] = n_sites
                _depth_add(mx, add, size, lo[i], hi[i], -1)
                _end_set(tree, esize, i, _NONE)
                alive -= 1
            core_s[n_sites], core_e[n_sites], first[n_sites] = cs, ce, c0 + out[0]
            n_sites += 1
    return (site_of, core_s[:n_sites].copy(), core_e[:n_sites].copy(), first[:n_sites].copy(),
            held_site[:n_held].copy(), held_window[:n_held].copy())


@njit(nogil=True, cache=True)
def _anchor(order, offsets, partners, adopt):
    """Families over sites taken in `order`: a site joins the family most of its placed
    partners are in, or else anchors a new one and takes its unplaced partners."""
    family = np.full(order.shape[0], -1, np.int64)
    anchors = np.empty(order.shape[0], np.int64)
    placed = np.empty(max(partners.shape[0], 1), np.int64)
    n_families = 0
    for sid in order:
        if family[sid] >= 0:
            continue
        p0, p1 = offsets[sid], offsets[sid + 1]
        n_placed = 0
        for j in range(p0, p1):
            if family[partners[j]] >= 0:
                placed[n_placed] = family[partners[j]]
                n_placed += 1
        if n_placed and n_placed >= adopt * (p1 - p0):
            ranked = np.sort(placed[:n_placed])
            best, best_count, run = ranked[0], 0, 0
            for j in range(n_placed):
                run = run + 1 if j and ranked[j] == ranked[j - 1] else 1
                if run > best_count:
                    best, best_count = ranked[j], run
            family[sid] = best
            continue
        family[sid] = n_families
        anchors[n_families] = sid
        for j in range(p0, p1):
            if family[partners[j]] < 0:
                family[partners[j]] = n_families
        n_families += 1
    return family, anchors[:n_families].copy()


@njit(nogil=True, cache=True)
def _families_of(k, window_of, site_of, family, listed_offsets, listed, out):
    """Alignment k's families, sorted and distinct, into `out`; returns how many."""
    a, b = window_of[2 * k], window_of[2 * k + 1]
    fa, fb = family[site_of[a]], family[site_of[b]]
    extra = listed_offsets[a + 1] - listed_offsets[a] + listed_offsets[b + 1] - listed_offsets[b]
    if not extra:  # the usual case: just the two ends
        if fa == fb:
            out[0] = fa
            return 1
        out[0], out[1] = min(fa, fb), max(fa, fb)
        return 2
    found = np.empty(2 + extra, np.int64)
    found[0], found[1] = fa, fb
    j = 2
    for w in (a, b):
        for x in range(listed_offsets[w], listed_offsets[w + 1]):
            found[j] = listed[x]
            j += 1
    found = np.unique(found)
    out[:found.shape[0]] = found
    return found.shape[0]


@njit(nogil=True, cache=True)
def _group_lists(window_of, site_of, family, listed_offsets, listed):
    """Each alignment's families: its two ends', and any listing either end as a longer passage."""
    n = window_of.shape[0] // 2
    widest = 2
    for w in range(listed_offsets.shape[0] - 1):
        widest = max(widest, 2 + 2 * (listed_offsets[w + 1] - listed_offsets[w]))
    out = np.empty(widest, np.int64)
    offsets = np.zeros(n + 1, np.int64)
    for k in range(n):
        offsets[k + 1] = offsets[k] + _families_of(k, window_of, site_of, family, listed_offsets, listed, out)
    members = np.empty(offsets[n], np.int64)
    for k in range(n):
        found = _families_of(k, window_of, site_of, family, listed_offsets, listed, out)
        members[offsets[k]:offsets[k] + found] = out[:found]
    return offsets, members


class _Families:
    """What the two output steps need from the grouping."""

    __slots__ = ("names", "site_file", "core_s", "core_e", "anchors", "shown_s", "documents",
                 "rep_passage", "rep_family", "rep_side", "list_offsets", "list_members")

    def group_ids(self, passage_id: int) -> bytes:
        o = self.list_offsets
        if passage_id + 1 >= o.shape[0]:
            return b"[]"
        return orjson.dumps(self.list_members[o[passage_id]:o[passage_id + 1]], option=orjson.OPT_SERIALIZE_NUMPY)

    __getitem__ = group_ids


def group_passages(corpus: list, run_dir: str = "", workers: int = 1) -> _Families:
    """Sites, then families, then what each alignment and each family row gets.

    Takes `corpus` from `read_ends` and empties it: the per-end spans are the largest
    thing held, and nothing needs them once the windows are built.
    """
    names, years, dumps, files, starts, ends = corpus
    corpus.clear()
    _status(_label(2))
    order = np.lexsort((ends, starts, files))
    window_of, wf, ws, we, first_end = _windows(files, starts, ends, order)
    del order, files, starts, ends
    site_of, core_s, core_e, first, held_site, held_window = _carve(
        ws, we, _clusters(wf, ws, we), _INSIDE, _FLOOR, _CONTAIN)
    site_file = wf[first]
    n_sites, n_files = core_s.shape[0], len(names)

    _status(_label(3))
    a = site_of[window_of[0::2]]
    b = site_of[window_of[1::2]]
    keep = a != b
    keys = np.unique(np.concatenate((a[keep] * n_sites + b[keep], b[keep] * n_sites + a[keep])))
    del a, b, keep
    offsets = np.concatenate(([0], np.cumsum(np.bincount(keys // n_sites, minlength=n_sites)))).astype(np.int64)
    partners = (keys % n_sites).astype(np.int64)
    del keys
    rank = np.empty(n_files, np.int64)
    rank[sorted(range(n_files), key=names.__getitem__)] = np.arange(n_files)
    visit = np.lexsort((np.arange(n_sites), core_s, rank[site_file], years[site_file]))
    family, anchors = _anchor(visit, offsets, partners, _ADOPT)
    # two families anchored on the very same passage are one family
    keys = np.stack((site_file[anchors], core_s[anchors], core_e[anchors]), axis=1)
    _, first_of, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    if first_of.shape[0] < anchors.shape[0]:
        kept = np.sort(first_of)
        family = np.searchsorted(kept, first_of[inverse.ravel()])[family]
        anchors = anchors[kept]
    n_families = anchors.shape[0]
    # a family's passage is shown from the first word of its sentence, or the first after the
    # punctuation nearest before it, when that is only a few words back and no two families
    # then show the same passage
    head_file, head_s, head_e = site_file[anchors], core_s[anchors], core_e[anchors]
    shown_s = _sentence_starts_for(names, dumps, run_dir, head_file, head_s, workers)
    shown_s = _kept_apart(head_file, head_s, head_e, shown_s)

    listed_family = family[held_site]
    pairs = np.unique(held_window * n_families + listed_family) if held_site.size else np.empty(0, np.int64)
    listed_offsets = np.concatenate(([0], np.cumsum(np.bincount(pairs // n_families, minlength=wf.shape[0])))).astype(np.int64)
    list_offsets, list_members = _group_lists(window_of, site_of, family, listed_offsets, pairs % n_families)

    # a family's documents are those of the alignments carrying it: what its page shows
    per_alignment = np.repeat(np.arange(list_offsets.shape[0] - 1), np.diff(list_offsets))
    doc_keys = np.unique(np.concatenate((list_members * n_files + wf[window_of[2 * per_alignment]],
                                         list_members * n_files + wf[window_of[2 * per_alignment + 1]])))
    del per_alignment
    documents = np.bincount(doc_keys // n_files, minlength=n_families)
    del doc_keys

    result = _Families()
    result.names = names
    result.site_file, result.core_s, result.core_e = site_file, core_s, core_e
    result.anchors, result.shown_s, result.documents = anchors, shown_s, documents
    rep_end = first_end[first[anchors]]  # a family's row comes from its anchor's first window
    by_passage = np.argsort(rep_end // 2, kind="stable")
    result.rep_passage = (rep_end // 2)[by_passage]
    result.rep_family = by_passage
    result.rep_side = (rep_end % 2)[by_passage]
    result.list_offsets, result.list_members = list_offsets, list_members
    return result


class _SourceText:
    """Passage text, holding open the file it last read from.

    Groups are written a file at a time -- four in five consecutive ones come
    from the same file -- and opening it again was over a third of what
    fetching a passage cost.
    """

    __slots__ = ("name", "handle")

    def __init__(self):
        self.name: str | None = None
        self.handle = None

    def read(self, start_byte: int, end_byte: int, filename: str) -> str:
        if filename != self.name:
            self.close()
            self.handle = open(filename, "rb")
            self.name = filename
        if start_byte < 0:
            start_byte = 0
        self.handle.seek(start_byte)
        return clean_passage(self.handle.read(end_byte - start_byte))

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
        self.handle = None
        self.name = None


class _Token(msgspec.Struct):
    """A word or punctuation mark in a words_and_philo_ids dump."""

    position: str  # doc div1 div2 div3 para sent word ...
    start_byte: int
    token: str = ""
    philo_type: str = "word"


_DECODE_TOKEN = msgspec.json.Decoder(_Token).decode


def _sentence_starts(path: str, upto: int):
    """Each token's start; the first of the words leading up to it, back to its sentence's start
    or the last mark that separates clauses; and how many words that is, as far as `upto`."""
    starts, firsts, before = array("q"), array("q"), array("q")
    current, first, words = None, None, 0
    try:
        with lz4.frame.open(path) as handle:
            for batch in read_line_batches(handle):
                for line in batch:
                    token = _DECODE_TOKEN(line)
                    sentence = token.position.split(" ", 6)[:6]
                    if sentence != current:
                        current, first, words = sentence, None, 0
                    starts.append(token.start_byte)
                    if token.philo_type == "punct" and _STOPS.search(token.token):
                        firsts.append(token.start_byte)  # a passage starting here stays put
                        before.append(0)
                        first, words = None, 0
                        continue
                    if first is None and token.philo_type == "word":
                        first = token.start_byte
                    firsts.append(token.start_byte if first is None else first)
                    before.append(words)
                    words += token.philo_type == "word"
                if starts and starts[-1] > upto:
                    break
    except (OSError, ValueError, msgspec.DecodeError):
        return array("q"), array("q"), array("q")
    return starts, firsts, before


def _words_dump(filename: str, parsed, doc_id, run_dir: str) -> str | None:
    """A document's words_and_philo_ids dump: the path its records name, or the one beside its TEI."""
    candidates = [parsed if os.path.isabs(parsed) else os.path.join(run_dir, parsed)] if parsed else []
    candidates.append(os.path.join(os.path.dirname(os.path.dirname(filename)), "words_and_philo_ids", f"{doc_id}.lz4"))
    return next((path for path in candidates if os.path.exists(path)), None)


def _sentence_starts_in(task: tuple) -> list[int]:
    """The first of the words leading up to the first token at or after each byte -- the first
    word of its sentence, or the first after a mark that separates clauses -- if only a few
    words back; otherwise the byte itself."""
    path, wanted = task
    starts, firsts, before = _sentence_starts(path, wanted[-1])
    found = []
    for byte in wanted:
        j = bisect_left(starts, byte)
        near = j < len(starts) and before[j] <= _SENTENCE_WORDS and firsts[j] <= byte
        found.append(firsts[j] if near else byte)
    return found


def _kept_apart(files, starts, ends, shown) -> np.ndarray:
    """`shown`, except that no two families show the very same passage.

    Where moves would make them, the family whose own passage starts earliest keeps its
    move and the others keep their own start -- again until none do, since a start kept
    can meet another family's move.
    """
    shown = shown.copy()
    while True:
        _, inverse, counts = np.unique(np.stack((files, shown, ends), axis=1), axis=0,
                                       return_inverse=True, return_counts=True)
        inverse = inverse.ravel()
        crowded = np.flatnonzero(counts[inverse] > 1)
        if not crowded.size:
            return shown
        order = crowded[np.lexsort((starts[crowded], inverse[crowded]))]
        later = np.ones(order.shape[0], bool)
        later[0] = False
        later[1:] = inverse[order][1:] == inverse[order][:-1]
        shown[order[later]] = starts[order[later]]


def _sentence_starts_for(names, dumps, run_dir, site_file, byte, workers: int) -> np.ndarray:
    """`byte` moved back to the start of its sentence or clause, where the document's dump says so."""
    moved = byte.copy()
    order = np.lexsort((byte, site_file))
    tasks, positions = [], []
    lo = 0
    while lo < order.shape[0]:
        f = site_file[order[lo]]
        hi = lo
        while hi < order.shape[0] and site_file[order[hi]] == f:
            hi += 1
        path = _words_dump(names[f], *dumps[f], run_dir)
        if path is not None:
            tasks.append((path, byte[order[lo:hi]].tolist()))
            positions.append(order[lo:hi])
        lo = hi
    if not tasks:
        return moved
    bar = tqdm(total=len(tasks), desc=_label(3, "finding sentence starts"), leave=False)
    if workers > 1 and len(tasks) > 1:
        with mp.get_context("fork").Pool(min(workers, len(tasks))) as pool:
            for where, found in zip(positions, pool.imap(_sentence_starts_in, tasks)):
                moved[where] = found
                bar.update()
    else:
        for where, task in zip(positions, tasks):
            moved[where] = _sentence_starts_in(task)
            bar.update()
    bar.close()
    return moved


class _GroupFields(msgspec.Struct):
    """Whether a record already carries the fields the rewrite sets.

    Raw so the values are never parsed -- only their presence matters, and a
    record that has them has to be rewritten rather than appended to.
    """

    group_id: Union[msgspec.Raw, msgspec.UnsetType] = msgspec.UNSET
    count: Union[msgspec.Raw, msgspec.UnsetType] = msgspec.UNSET


_DECODE_GROUP_FIELDS = msgspec.json.Decoder(_GroupFields).decode


def with_group_ids(line: bytes, groups: bytes) -> bytes:
    """`line` with its group list set, and any stale count dropped.

    `groups` comes already encoded as JSON. The rewrite adds one field to a
    record of about a hundred, so the list is spliced into the bytes rather
    than the record decoded and encoded back. orjson wrote these files and
    reproduces them byte for byte, so the result is what the round trip
    produced. A record that already carries either field goes through the
    round trip, since one has to keep its place and the other has to go.
    """
    present = _DECODE_GROUP_FIELDS(line)
    if present.group_id is msgspec.UNSET and present.count is msgspec.UNSET:
        body = line[:-1] if line.endswith(b"\n") else line
        if body.endswith(b"}") and not body.endswith(b"{}"):
            return body[:-1] + b',"group_id":' + groups + b"}\n"
    fields = orjson.loads(line)
    fields["group_id"] = orjson.loads(groups)
    # Remove old 'count' field if it exists, use final_group_counts later
    if "count" in fields:
        del fields["count"]
    return orjson.dumps(fields) + b"\n"


_READ_SIZE = 1 << 22  # 4 MiB


def read_line_batches(input_file, read_size: int = _READ_SIZE):
    """The file's lines, a block at a time: line-by-line reads through lz4 are slow."""
    remainder = b""
    while True:
        chunk = input_file.read(read_size)
        if not chunk:
            break
        lines = (remainder + chunk).split(b"\n")
        remainder = lines.pop()
        if lines:
            yield lines
    if remainder:
        yield [remainder]


def _join_parts(parts: list[str], target: str) -> None:
    """Concatenate the parts, raising if the join came out short.

    `_concatenate` ignores `cat`'s exit status, and the results file is replaced
    straight after.
    """
    expected = sum(os.path.getsize(part) for part in parts)
    _concatenate(parts, target)
    joined = os.path.getsize(target)
    if joined != expected:
        raise OSError(f"{target}: joined {joined:,} bytes where the parts held {expected:,}")


def _flush(progress, since: int) -> None:
    """Add what `_bump` left behind, which is under the stride by definition."""
    if progress is not None and since:
        with progress.get_lock():
            progress.value += since


def _count_segment(job: tuple[str, int, int, str]) -> int:
    """Records in one frame range, counted the way `read_line_batches` splits them."""
    path, offset, size, _part = job
    progress = _SHARED.get("progress")
    total = 0
    pending = 0
    last = b"\n"
    with _lz4_window(path, offset, size) as input_file:
        while True:
            chunk = input_file.read(_READ_SIZE)
            if not chunk:
                break
            found = chunk.count(b"\n")
            total += found
            pending = _bump(progress, pending + found)
            last = chunk[-1:]
    if last != b"\n":
        total += 1  # a last line without a newline is still a line
    _flush(progress, pending)
    return total


def _segment_bases(path: str, segments: list[tuple[int, int]], count: int) -> dict[int, int]:
    """The passage id each segment starts at, keyed by offset: frames don't record it."""
    counts = _run_segments(_count_segment, path, segments, count, _label(4, "locating records"))
    bases = {}
    running = 0
    for (offset, _size), found in zip(segments, counts):
        bases[offset] = running
        running += found
    return bases


def _segment_plan(results_file: str, count: int, workers: int):
    """Frame ranges for the two output passes, and where each starts; one range means run here."""
    segments = _work_split(results_file, workers)
    if len(segments) < 2:
        return segments, {}
    return segments, _segment_bases(results_file, segments, count)


def merge_alignments(results_file: str, count: int, workers: int = 1):
    """Group alignments into passage families and write the group file"""
    corpus = read_ends(results_file, count)
    if not corpus[3].shape[0]:
        _status()
        print("  No passage groups found.")
        return None
    # a relative words_and_philo_ids path in the records is rooted in the run's directory
    run_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(results_file))))
    families = group_passages(corpus, run_dir, workers)

    # Both output steps read the records back, so they share one split of the file
    _status(_label(4))
    segments, bases = _segment_plan(results_file, count, workers)

    groups_file = os.path.join(os.path.dirname(results_file), "passage_group_source.jsonl")
    groups = write_group_sources(results_file, groups_file, families, segments, bases)

    temp_results_file = f"{results_file}.temp_final.lz4"
    rewrite_results(results_file, temp_results_file, families, count, segments, bases)
    os.remove(results_file)
    os.rename(temp_results_file, results_file)

    _status()
    print(f"  {groups:,} passage groups, written to {os.path.basename(groups_file)}")
    return groups_file


def _rewrite_records(input_file, output_file, group_lists: _Families, passage_id: int,
                     progress=None, bar=None) -> int:
    """Put each record's group list into it. Returns the id after the last record."""
    pending = 0
    for batch in read_line_batches(input_file):
        rewritten = []
        for line in batch:
            # Get list, default empty
            rewritten.append(with_group_ids(line, group_lists[passage_id]))
            passage_id += 1
        output_file.write(b"".join(rewritten))
        if bar is not None:
            bar.update(len(batch))
        else:
            pending = _bump(progress, pending + len(batch))
    _flush(progress, pending)
    return passage_id


def _rewrite_segment(job: tuple[str, int, int, str]) -> str:
    """One frame range rewritten into its own part file.

    The group lists come through the fork; being flat arrays, reading them copies no pages.
    """
    path, offset, size, part = job
    group_lists = _SHARED["group_lists"]
    with contextlib.ExitStack() as stack:
        input_file = stack.enter_context(_lz4_window(path, offset, size))
        output_file = stack.enter_context(_Framer(stack.enter_context(open(part, "wb"))))
        _rewrite_records(input_file, output_file, group_lists,
                         _SHARED["bases"][offset], _SHARED["progress"])
    return part


def rewrite_results(results_file: str, target: str, group_lists: _Families,
                    count: int, segments: list[tuple[int, int]], bases: dict[int, int]) -> None:
    """Write the results file back with a group list on every record.

    Split over frames and joined back in order. Written as many frames, so the
    next pass can split it too.
    """
    if len(segments) < 2:
        with contextlib.ExitStack() as stack:
            input_file = stack.enter_context(lz4.frame.open(results_file))
            output_file = stack.enter_context(_Framer(stack.enter_context(open(target, "wb"))))
            bar = stack.enter_context(tqdm(total=count, desc=_label(5), leave=False))
            _rewrite_records(input_file, output_file, group_lists, 0, bar=bar)
        return

    _SHARED.update(bases=bases, group_lists=group_lists)
    try:
        parts = _run_segments(_rewrite_segment, results_file, segments, count, _label(5))
    finally:
        # A module global, so a failed pass would otherwise leave the lists
        # pinned for the rest of the run.
        for key in ("bases", "group_lists"):
            _SHARED.pop(key, None)
    _status(_label(5))
    _join_parts(parts, target)


def write_group_sources(results_file: str, groups_file: str, families: _Families,
                        segments: list[tuple[int, int]], bases: dict[int, int]) -> int:
    """Write one row per family: its anchor's passage, from its sentence's start when that is
    a few words back, with the metadata of the record that first has it, and `count`, the
    documents its alignments span. Returns the rows written.

    Rows come out in the order their records appear; the database load keys on group_id.
    """
    total = families.anchors.shape[0]
    if len(segments) < 2:
        with contextlib.ExitStack() as stack:
            input_file = stack.enter_context(lz4.frame.open(results_file))
            output_file = stack.enter_context(open(groups_file, "wb"))
            bar = stack.enter_context(tqdm(total=total, desc=_label(4), leave=False))
            _group_source_records(input_file, output_file, families, 0, bar=bar)
        return total

    _SHARED.update(bases=bases, families=families)
    try:
        parts = _run_segments(_group_source_segment, results_file, segments, total, _label(4))
    finally:
        for key in ("bases", "families"):
            _SHARED.pop(key, None)
    # Plain jsonl, not lz4, so the parts join as they are.
    _status(_label(4))
    _join_parts(parts, groups_file)
    return total


def _as_source(fields: dict, side: int) -> dict:
    """A record's fields for one of its ends, named as source fields and in source order."""
    if side == 0:
        return {k: v for k, v in fields.items() if not k.startswith("target_")}
    return {k: fields.get("target_" + k[7:]) if k.startswith("source_") else v
            for k, v in fields.items() if not k.startswith("target_")}


def _group_source_records(input_file, output_file, families: _Families, passage_id: int,
                          progress=None, bar=None) -> None:
    """Write the row of every family whose record is in this range."""
    reps = families.rep_passage
    j = int(np.searchsorted(reps, passage_id))
    source_text = _SourceText()
    pending = 0
    for batch in read_line_batches(input_file):
        records = []
        for line in batch:
            if j < reps.shape[0] and reps[j] == passage_id:
                fields = orjson.loads(line)
                fields["passage_id"] = passage_id
                while j < reps.shape[0] and reps[j] == passage_id:
                    family = int(families.rep_family[j])
                    side = int(families.rep_side[j])
                    site = families.anchors[family]
                    filename = families.names[families.site_file[site]]
                    start_byte, end_byte = int(families.shown_s[family]), int(families.core_e[site])
                    records.append(
                        orjson.dumps(
                            {
                                **_as_source(fields, side),
                                "source_filename": filename,
                                "source_passage": source_text.read(start_byte, end_byte, filename),
                                "group_id": family,
                                "source_start_byte": start_byte,
                                "source_end_byte": end_byte,
                                "count": int(families.documents[family]),
                            }
                        )
                        + b"\n"
                    )
                    j += 1
            passage_id += 1
        if records:
            output_file.write(b"".join(records))
            if bar is not None:
                bar.update(len(records))
            else:
                pending = _bump(progress, pending + len(records))
    _flush(progress, pending)
    source_text.close()


def _group_source_segment(job: tuple[str, int, int, str]) -> str:
    """One frame range's group rows, into its own part file."""
    path, offset, size, part = job
    with contextlib.ExitStack() as stack:
        input_file = stack.enter_context(_lz4_window(path, offset, size))
        output_file = stack.enter_context(open(part, "wb"))
        _group_source_records(input_file, output_file, _SHARED["families"],
                              _SHARED["bases"][offset], _SHARED["progress"])
    return part


if __name__ == "__main__":
    import sys

    output_path = sys.argv[1]
    with open(os.path.join(output_path, "results/count.txt"), encoding="utf8") as input_file:
        count = int(input_file.read().strip())
    merge_alignments(os.path.join(output_path, "results/alignments.jsonl.lz4"), count)
