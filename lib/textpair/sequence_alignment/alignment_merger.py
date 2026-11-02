"""Group alignments into passage families: one per reused passage, anchored on its earliest occurrence.

Every alignment has two ends, spans of text in two documents. At each place in a
document, the ends that overlap are carved into sites, each built around the passage
most of them cover. A family is then the earliest site of a passage together with the
sites aligned directly with it. A window running on over several passages counts under
each of them, and anchors no family of its own. A document in a family only through its
partners, not aligned with the family's passage itself, stays in it if it holds at least
half of that passage (family_places).
"""

import contextlib
import os
import re
import sys
from array import array
from typing import Union

import lz4.frame
import msgspec
import numpy as np
import orjson
from numba import njit
from tqdm import tqdm

from textpair.utils import clean_passage

from . import family_places
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
          "projecting passages onto their reuses", "writing group sources", "rewriting results")


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
_LEAD_WORDS = 5  # words a family's passage may start before its core, where one of its reuses starts
_LEAD_BYTES = 4096  # ... read back no further than this
_HOLDS = 0.5     # a document not aligned with a family's passage stays when it holds this much of it
_WORD = re.compile(r"\w+")
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


_DECODE_ENDS = msgspec.json.Decoder(_Ends).decode


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

    Returns the file names, each file's year, and each end's file, start and end.
    """
    names: list[str] = []
    ids: dict[str, int] = {}
    years: list[int] = []
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
                target = ids.get(r.target_filename)
                if target is None:
                    target = ids[r.target_filename] = len(names)
                    names.append(r.target_filename)
                    years.append(_year(r.target_year))
                files.extend((source, target))
                starts.extend((r.source_start_byte, r.target_start_byte))
                ends.extend((r.source_end_byte, r.target_end_byte))
            progress.update(len(batch))
        progress.close()
    if not files:
        return [names, np.empty(0, np.int64), np.empty(0, np.int32), np.empty(0, np.int64), np.empty(0, np.int64)]
    # a list, so the grouping can take the spans over and free them once it has the windows
    return [names, np.array(years, np.int64), np.frombuffer(files, np.int32),
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


def _window_weights(window_of, files, n_windows):
    """Each window's documents: the distinct files at the other ends of the alignments it is
    an end of. Identical spans are one window, so this is what it stands for."""
    partner = files[np.arange(files.shape[0]) ^ 1].astype(np.int64)
    pairs = np.unique(window_of.astype(np.int64) * (int(files.max()) + 1 if files.size else 1) + partner)
    return np.bincount(pairs // (int(files.max()) + 1 if files.size else 1), minlength=n_windows).astype(np.int64)


@njit(nogil=True, cache=True)
def _weighted_at(values, weights, need, descending):
    """The value at which the running weight of values, in order, first reaches `need`."""
    order = np.argsort(-values if descending else values, kind="mergesort")
    total = 0
    for j in order:
        total += weights[j]
        if total >= need:
            return values[j]
    return values[order[-1]]


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
def _chained_without_longest(starts, ends, members):
    """Whether `members` still overlap one another in a chain once the longest is left out."""
    n = members.shape[0]
    s, e = starts[members], ends[members]
    keep = np.arange(n) != np.argmax(e - s)
    s, e = s[keep], e[keep]
    by_start = np.argsort(s)
    reach = e[by_start[0]]
    for j in range(1, n - 1):
        i = by_start[j]
        if s[i] >= reach:
            return False
        reach = max(reach, e[i])
    return True


@njit(nogil=True, cache=True)
def _carve(starts, ends, weights, bounds, inside, floor, contain):
    """Sites for every cluster of windows. Returns each window's site, each site's core and
    first member, and the (site, window) pairs of longer windows listed under a site.

    A window counts for the documents it stands for (`weights`). A site is built at the
    point most of them cover: its core is where at least half of those overlap, and it
    takes the windows lying mostly inside the core. With an even count that core spans the
    two middle ones; if that lets a long window hold separate passages together, the core
    is where more than half of them overlap.
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
        s, e, wt = starts[c0:c1], ends[c0:c1], weights[c0:c1]
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
                _depth_add(mx, add, size, lo[i], hi[i], wt[i])
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
            cover = out[:g].copy()
            weight = wt[cover].sum()
            need = (weight + 1) // 2
            held_start = n_held
            while True:
                cs = _weighted_at(s[cover], wt[cover], need, False)
                ce = _weighted_at(e[cover], wt[cover], need, True)
                core = ce - cs
                m = _ending_past(tree, esize, np.searchsorted(s, ce, side="left"), cs, out, stack)
                n_members = 0
                n_held = held_start
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
                # only the longest window links these members: retry on a strict majority
                if need * 2 == weight and n_members > 2 and not _chained_without_longest(s, e, out[:n_members]):
                    need += 1
                    continue
                break
            if n_members == 0:  # nothing sits inside: the window overlapping it most stands alone
                out[0] = best
                n_members = 1
            for j in range(n_members):
                i = out[j]
                site_of[c0 + i] = n_sites
                _depth_add(mx, add, size, lo[i], hi[i], -wt[i])
                _end_set(tree, esize, i, _NONE)
                alive -= 1
            core_s[n_sites], core_e[n_sites], first[n_sites] = cs, ce, c0 + out[0]
            n_sites += 1
    return (site_of, core_s[:n_sites].copy(), core_e[:n_sites].copy(), first[:n_sites].copy(),
            held_site[:n_held].copy(), held_window[:n_held].copy())


@njit(nogil=True, cache=True)
def _anchor(order, offsets, partners, adopt, run_on, list_offsets, listing, core_s, core_e):
    """Families over sites taken in `order`: a site joins the family most of its placed
    partners are in, or else anchors a new one and takes its unplaced partners, except
    those that most of their own partners have already placed elsewhere.

    A run-on site -- only windows listed under other sites -- joins the family of the
    listing site it overlaps most, and is never taken as a partner."""
    family = np.full(order.shape[0], -1, np.int64)
    anchors = np.empty(order.shape[0], np.int64)
    placed = np.empty(max(partners.shape[0], 1), np.int64)
    n_families = 0
    for sid in order:
        if family[sid] >= 0:
            continue
        if run_on[sid]:
            best, best_overlap = -1, _NONE
            for j in range(list_offsets[sid], list_offsets[sid + 1]):
                x = listing[j]
                if family[x] < 0:
                    continue
                overlap = min(core_e[x], core_e[sid]) - max(core_s[x], core_s[sid])
                if overlap > best_overlap or (overlap == best_overlap and x < best):
                    best, best_overlap = x, overlap
            if best >= 0:
                family[sid] = family[best]
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
            q = partners[j]
            if family[q] >= 0 or run_on[q]:
                continue
            elsewhere = 0  # a partner mostly placed in other families is left to join them
            for x in range(offsets[q], offsets[q + 1]):
                if family[partners[x]] >= 0 and family[partners[x]] != n_families:
                    elsewhere += 1
            if elsewhere and elsewhere >= adopt * (offsets[q + 1] - offsets[q]):
                continue
            family[q] = n_families
        n_families += 1
    return family, anchors[:n_families].copy()


@njit(nogil=True, cache=True)
def _refereed(window_of, wf, site_of, family, is_anchor, run_on, anchor_file, anchor_s, anchor_e):
    """What each family's anchor document says of the windows of its other sites.

    Returns whether each window is aligned there with its own family, and the (window,
    family) pairs where it is aligned there with another family anchored there, apart
    from its own -- once per alignment. A run-on window there, listed under several
    passages, says neither."""
    home = np.zeros(wf.shape[0], np.bool_)
    away_w = np.empty(0, np.int64)
    away_g = np.empty(0, np.int64)
    n = 0
    for step in range(2):  # count, then fill
        if step:
            away_w = np.empty(n, np.int64)
            away_g = np.empty(n, np.int64)
            n = 0
        for e in range(window_of.shape[0]):
            w, p = window_of[e], window_of[e ^ 1]
            s = site_of[w]
            if is_anchor[s]:
                continue
            f = family[s]
            if wf[p] != anchor_file[f] or run_on[site_of[p]]:
                continue
            g = family[site_of[p]]
            if g == f:
                home[w] = True
            elif (anchor_file[g] == anchor_file[f]
                  and min(anchor_e[g], anchor_e[f]) <= max(anchor_s[g], anchor_s[f])):
                if step:
                    away_w[n], away_g[n] = w, g
                n += 1
    return home, away_w, away_g


def _refer(window_of, wf, site_of, family, anchors, run_on, site_file, core_s, core_e):
    """Families after the anchors' documents have refereed: a member window aligned with its
    family's anchor document only at another family's passage there goes to that family.
    A site with no window at home goes whole; otherwise those windows leave it, each a site
    of its own. Returns the families, one per site, and changes `site_of` in place."""
    n_sites, n_families = family.shape[0], anchors.shape[0]
    is_anchor = np.zeros(n_sites, np.bool_)
    is_anchor[anchors] = True
    home, away_w, away_g = _refereed(window_of, wf, site_of, family, is_anchor, run_on,
                                     site_file[anchors], core_s[anchors], core_e[anchors])
    keep = ~home[away_w]
    if not keep.any():
        return family
    # each window goes where most of its alignments there lead, to the earliest family on a tie
    keys, counts = np.unique(away_w[keep] * n_families + away_g[keep], return_counts=True)
    w, g = keys // n_families, keys % n_families
    pick = np.lexsort((g, -counts, w))
    w, g = w[pick], g[pick]
    first_of = np.concatenate(([True], w[1:] != w[:-1]))
    w, g = w[first_of], g[first_of]
    at_home = np.zeros(n_sites, np.bool_)
    at_home[site_of[np.flatnonzero(home)]] = True
    whole = ~at_home[site_of[w]]
    family = family.copy()
    if whole.any():
        # a site goes whole where most of its windows go, to the earliest family on a tie
        keys, counts = np.unique(site_of[w[whole]] * n_families + g[whole], return_counts=True)
        s, t = keys // n_families, keys % n_families
        pick = np.lexsort((t, -counts, s))
        s, t = s[pick], t[pick]
        first_of = np.concatenate(([True], s[1:] != s[:-1]))
        family[s[first_of]] = t[first_of]
    leaving = ~whole
    site_of[w[leaving]] = n_sites + np.arange(int(leaving.sum()))
    return np.concatenate((family, g[leaving]))


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
    """What the output steps need from the grouping, and what settling families on their places needs."""

    __slots__ = ("names", "years", "site_file", "core_s", "core_e", "anchors", "shown_s", "documents", "rep_end",
                 "rep_passage", "rep_family", "rep_side", "list_offsets", "list_members",
                 "window_of", "wf", "ws", "we")

    def group_ids(self, passage_id: int) -> bytes:
        o = self.list_offsets
        if passage_id + 1 >= o.shape[0]:
            return b"[]"
        return orjson.dumps(self.list_members[o[passage_id]:o[passage_id + 1]], option=orjson.OPT_SERIALIZE_NUMPY)

    __getitem__ = group_ids


def _document_keys(list_offsets, list_members, window_of, wf, n_files):
    """(family, file) keys of a family's documents: those of both ends of the alignments
    carrying or listing it, its anchor's too."""
    per_alignment = np.repeat(np.arange(list_offsets.shape[0] - 1), np.diff(list_offsets))
    return np.unique(np.concatenate((list_members * n_files + wf[window_of[2 * per_alignment]],
                                     list_members * n_files + wf[window_of[2 * per_alignment + 1]])))


def _documents(list_offsets, list_members, window_of, wf, n_files, n_families):
    """Each family's document count."""
    keys = _document_keys(list_offsets, list_members, window_of, wf, n_files)
    return np.bincount(keys // n_files, minlength=n_families)


def _order_rows(result: _Families) -> None:
    """Families in the order of the records their rows come from."""
    by_passage = np.argsort(result.rep_end // 2, kind="stable")
    result.rep_passage = (result.rep_end // 2)[by_passage]
    result.rep_family = by_passage
    result.rep_side = (result.rep_end % 2)[by_passage]


def group_passages(corpus: list) -> _Families:
    """Sites, then families, then what each alignment and each family row gets.

    Takes `corpus` from `read_ends` and empties it: the per-end spans are the largest
    thing held, and nothing needs them once the windows are built.
    """
    names, years, files, starts, ends = corpus
    corpus.clear()
    _status(_label(2))
    order = np.lexsort((ends, starts, files))
    window_of, wf, ws, we, first_end = _windows(files, starts, ends, order)
    weights = _window_weights(window_of, files, wf.shape[0])
    del order, files, starts, ends
    site_of, core_s, core_e, first, held_site, held_window = _carve(
        ws, we, weights, _clusters(wf, ws, we), _INSIDE, _FLOOR, _CONTAIN)
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
    # a run-on site comes after the other sites of its document, so the passages it runs over are placed
    run_on = np.bincount(site_of[np.unique(held_window)], minlength=n_sites) == np.bincount(site_of, minlength=n_sites)
    pairs = np.unique(site_of[held_window] * n_sites + held_site) if held_site.size else np.empty(0, np.int64)
    list_offsets = np.concatenate(([0], np.cumsum(np.bincount(pairs // n_sites, minlength=n_sites)))).astype(np.int64)
    visit = np.lexsort((np.arange(n_sites), core_s, run_on, rank[site_file], years[site_file]))
    family, anchors = _anchor(visit, offsets, partners, _ADOPT, run_on, list_offsets, pairs % n_sites, core_s, core_e)
    del pairs, list_offsets
    # two families anchored on the very same passage are one family
    keys = np.stack((site_file[anchors], core_s[anchors], core_e[anchors]), axis=1)
    _, first_of, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    if first_of.shape[0] < anchors.shape[0]:
        kept = np.sort(first_of)
        family = np.searchsorted(kept, first_of[inverse.ravel()])[family]
        anchors = anchors[kept]
    n_families = anchors.shape[0]
    family = _refer(window_of, wf, site_of, family, anchors, run_on, site_file, core_s, core_e)
    del run_on
    # a family's passage starts where the earliest of its reuses does, when that is a few words
    # before its core -- words fewer than half of them share -- and no two families then show
    # the same passage
    head_file, head_s, head_e = site_file[anchors], core_s[anchors], core_e[anchors]
    shown_s = _kept_apart(head_file, head_s, head_e, _lead_starts(names, head_file, head_s, anchors, site_of, ws))
    listed_family = family[held_site]
    pairs = np.unique(held_window * n_families + listed_family) if held_site.size else np.empty(0, np.int64)
    listed_offsets = np.concatenate(([0], np.cumsum(np.bincount(pairs // n_families, minlength=wf.shape[0])))).astype(np.int64)
    listed = pairs % n_families
    list_offsets, list_members = _group_lists(window_of, site_of, family, listed_offsets, listed)

    result = _Families()
    result.names, result.years = names, years
    result.site_file, result.core_s, result.core_e = site_file, core_s, core_e
    result.anchors, result.shown_s = anchors, shown_s
    result.documents = _documents(list_offsets, list_members, window_of, wf, n_files, n_families)
    result.rep_end = first_end[first[anchors]]  # a family's row comes from its anchor's first window
    _order_rows(result)
    result.list_offsets, result.list_members = list_offsets, list_members
    result.window_of, result.wf, result.ws, result.we = window_of, wf, ws, we
    return result


def _lead_starts(names, head_file, head_s, anchors, site_of, ws) -> np.ndarray:
    """Each family's passage start: the earliest start of a window of its anchor site that
    is at most _LEAD_WORDS words before the site's core, or the core's own."""
    shown = head_s.copy()
    family_of = np.full(int(site_of.max()) + 1 if site_of.size else 0, -1, np.int64)
    family_of[anchors] = np.arange(anchors.shape[0])
    owner = family_of[site_of]
    lead = np.flatnonzero(owner >= 0)
    lead = lead[(ws[lead] < head_s[owner[lead]]) & (ws[lead] >= head_s[owner[lead]] - _LEAD_BYTES)]
    if not lead.size:
        return shown
    lead = lead[np.lexsort((ws[lead], owner[lead], head_file[owner[lead]]))]
    name, handle = None, None
    try:
        k = 0
        while k < lead.shape[0]:
            f = owner[lead[k]]
            m = k
            while m < lead.shape[0] and owner[lead[m]] == f:
                m += 1
            if names[head_file[f]] != name:
                if handle is not None:
                    handle.close()
                name = names[head_file[f]]
                handle = open(name, "rb")
            starts = ws[lead[k:m]]
            handle.seek(int(starts[0]))
            raw = handle.read(int(head_s[f] - starts[0]))
            for start in starts:  # earliest first
                if len(_WORD.findall(clean_passage(raw[int(start - starts[0]):]))) <= _LEAD_WORDS:
                    shown[f] = start
                    break
            k = m
    except OSError:
        pass
    finally:
        if handle is not None:
            handle.close()
    return shown


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
    counts = _run_segments(_count_segment, path, segments, count, _label(5, "locating records"))
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


@njit(nogil=True, cache=True)
def _place_keys(window_of, list_offsets, list_members, wf, anchor_file, n_windows):
    """(family, window) keys of the ends of every family's alignments outside its anchor's
    document, sorted: by family, then file and start, the order windows are numbered in."""
    keys = np.empty(2 * list_members.shape[0], np.int64)
    m = 0
    for k in range(list_offsets.shape[0] - 1):
        for j in range(list_offsets[k], list_offsets[k + 1]):
            g = list_members[j]
            for e in (2 * k, 2 * k + 1):
                w = window_of[e]
                if wf[w] != anchor_file[g]:
                    keys[m] = g * n_windows + w
                    m += 1
    return np.unique(keys[:m])


@njit(nogil=True, cache=True)
def _direct(keys, n_windows, window_of, list_offsets, list_members, wf, ws, we, p_file, p_start, p_end):
    """Whether each (family, window) is aligned with the family's passage itself: an
    alignment of that family has the window at one end and, at the other, the passage
    in its anchor's document."""
    found = np.empty(2 * list_members.shape[0], np.int64)
    m = 0
    for k in range(list_offsets.shape[0] - 1):
        a, b = window_of[2 * k], window_of[2 * k + 1]
        for j in range(list_offsets[k], list_offsets[k + 1]):
            g = list_members[j]
            for x, y in ((a, b), (b, a)):
                if wf[x] == p_file[g] and ws[x] < p_end[g] and we[x] > p_start[g]:
                    found[m] = g * n_windows + y
                    m += 1
    found = np.unique(found[:m])
    out = np.empty(keys.shape[0], np.bool_)
    for i in range(keys.shape[0]):
        at = np.searchsorted(found, keys[i])
        out[i] = at < found.shape[0] and found[at] == keys[i]
    return out


@njit(nogil=True, cache=True)
def _without(window_of, list_offsets, list_members, gone, n_windows):
    """Each alignment's families, less those an end of it is gone from (`gone`: sorted
    (family, window) keys)."""
    offsets = np.zeros_like(list_offsets)
    members = np.empty_like(list_members)
    m = 0
    for k in range(list_offsets.shape[0] - 1):
        a, b = window_of[2 * k], window_of[2 * k + 1]
        for j in range(list_offsets[k], list_offsets[k + 1]):
            g = list_members[j]
            ka, kb = g * n_windows + a, g * n_windows + b
            ia, ib = np.searchsorted(gone, ka), np.searchsorted(gone, kb)
            if not (ia < gone.shape[0] and gone[ia] == ka or ib < gone.shape[0] and gone[ib] == kb):
                members[m] = g
                m += 1
        offsets[k + 1] = m
    return offsets, members[:m].copy()


@njit(nogil=True, cache=True)
def _overlaps(qf, qs, qe, tf, ts, te):
    """(query, target) pairs overlapping by half the shorter of the two or more; targets
    sorted by file and start."""
    qi, ti = np.empty(0, np.int64), np.empty(0, np.int64)
    n = 0
    for step in range(2):  # count, then fill
        if step:
            qi, ti = np.empty(n, np.int64), np.empty(n, np.int64)
            n = 0
        for q in range(qf.shape[0]):
            j = np.searchsorted(tf, qf[q])
            while j < tf.shape[0] and tf[j] == qf[q] and ts[j] < qe[q]:
                overlap = min(te[j], qe[q]) - max(ts[j], qs[q])
                if overlap > 0 and 2 * overlap >= min(te[j] - ts[j], qe[q] - qs[q]):
                    if step:
                        qi[n], ti[n] = q, j
                    n += 1
                j += 1
    return qi, ti


@njit(nogil=True, cache=True)
def _relisted(list_offsets, list_members, which, offsets, members):
    """Each alignment's families, those of `which` (sorted) replaced by `offsets`/`members`."""
    out_offsets = np.zeros_like(list_offsets)
    out = np.empty(list_members.shape[0] + members.shape[0], np.int64)
    m = j = 0
    for k in range(list_offsets.shape[0] - 1):
        if j < which.shape[0] and which[j] == k:
            lo, hi, source = offsets[j], offsets[j + 1], members
            j += 1
        else:
            lo, hi, source = list_offsets[k], list_offsets[k + 1], list_members
        for x in range(lo, hi):
            out[m] = source[x]
            m += 1
        out_offsets[k + 1] = m
    return out_offsets, out[:m].copy()


def _homes(f: _Families, orphans, places, alive, passages, workers):
    """For alignments no family of theirs fits, the family both their ends hold at least half
    the passage of, among those whose places or passage overlap them; -1 where there is none.
    In its anchor's document, an end holds a passage it overlaps half of."""
    pg, pf, ps, pe = places
    anchor_file, p_start, p_end = passages
    n_families = anchor_file.shape[0]
    w = f.window_of[np.stack((2 * orphans, 2 * orphans + 1), axis=1).ravel()]
    qf, qs, qe = f.wf[w].astype(np.int64), f.ws[w], f.we[w]
    live = np.flatnonzero(alive)
    tg = np.concatenate((pg[live], np.arange(n_families)))
    tf = np.concatenate((pf[live], anchor_file)).astype(np.int64)
    ts, te = np.concatenate((ps[live], p_start)), np.concatenate((pe[live], p_end))
    order = np.lexsort((ts, tf))
    qi, ti = _overlaps(qf, qs, qe, tf[order], ts[order], te[order])
    found = np.unique(qi // 2 * n_families + tg[order][ti])
    lo, lm = f.list_offsets, f.list_members  # the families they had, already found wanting
    rows = np.repeat(np.arange(orphans.shape[0]), lo[orphans + 1] - lo[orphans])
    had = rows * n_families + lm[np.concatenate([np.arange(lo[k], lo[k + 1]) for k in orphans])]
    found = np.setdiff1d(found, had)
    homes = np.full(orphans.shape[0], -1, np.int64)
    if not found.size:
        return homes
    o, h = found // n_families, found % n_families
    share = np.zeros((found.shape[0], 2))
    elsewhere = []
    for side in range(2):
        e = 2 * o + side
        in_anchor = qf[e] == anchor_file[h]
        overlap = np.minimum(qe[e], p_end[h]) - np.maximum(qs[e], p_start[h])
        share[in_anchor, side] = (np.maximum(overlap, 0) / np.maximum(p_end[h] - p_start[h], 1))[in_anchor]
        elsewhere.append((np.flatnonzero(~in_anchor), side, qf[e][~in_anchor], qs[e][~in_anchor], qe[e][~in_anchor]))
    rows = np.concatenate([x[0] for x in elsewhere])
    sides = np.concatenate([np.full(x[0].shape[0], x[1]) for x in elsewhere])
    rf, rs, re_ = (np.concatenate([x[k] for x in elsewhere]) for k in (2, 3, 4))
    if rows.size:
        by = np.lexsort((re_, rs, rf, h[rows]))
        g_, f_, s_, e_, of = family_places.places(h[rows][by], rf[by], rs[by], re_[by])
        held = family_places.project(f.names, passages, g_, f_, s_, e_, workers,
                                     _label(4, "projecting passages onto alignments no family fits"))[2]
        share[rows[by], sides[by]] = held[of]
    score = share.min(axis=1)
    fits = np.flatnonzero(score >= _HOLDS)
    best = fits[np.lexsort((h[fits], -score[fits], o[fits]))]  # by orphan, best first, earliest family on a tie
    first = best[np.concatenate(([True], o[best][1:] != o[best][:-1]))] if best.size else best
    homes[o[first]] = h[first]
    return homes


def settle_families(families: _Families, places_file: str, workers: int = 1, context_size: int = 300) -> dict:
    """Project each family's passage onto its places, and keep in a family only the documents
    aligned with its passage itself or holding at least half of it: an alignment that brought in
    one that is neither leaves the family. An alignment that leaves every family it had goes to one
    whose passage both its ends hold, or else is grouped with the others left over into families
    of their own; a family no alignment is left in goes. Writes the places' rows; returns counts."""
    f = families
    n_windows = f.wf.shape[0]

    def passages():
        return f.site_file[f.anchors], f.shown_s, f.core_e[f.anchors]

    def placed(list_offsets, list_members):
        keys = _place_keys(f.window_of, list_offsets, list_members, f.wf, f.site_file[f.anchors], n_windows)
        w = keys % n_windows
        return keys, family_places.places(keys // n_windows, f.wf[w], f.ws[w], f.we[w])

    keys, (pg, pf, ps, pe, place_of) = placed(f.list_offsets, f.list_members)
    shown = family_places.project(f.names, passages(), pg, pf, ps, pe, workers, _label(4))
    held = np.zeros(pg.shape[0], np.bool_)
    held[place_of[_direct(keys, n_windows, f.window_of, f.list_offsets, f.list_members,
                          f.wf, f.ws, f.we, *passages())]] = True
    gone = ~held & (shown[2] < _HOLDS)
    counts = dict(left=0, went=0, orphans=0, homed=0, founded=0, joined=0)
    if gone.any():
        n_families = f.anchors.shape[0]
        list_offsets, list_members = _without(f.window_of, f.list_offsets, f.list_members,
                                              keys[gone[place_of]], n_windows)
        orphans = np.flatnonzero(np.diff(list_offsets) == 0)
        if orphans.size:
            homes = _homes(f, orphans, (pg, pf, ps, pe), ~gone, passages(), workers)
            lists = [np.array([h]) for h in homes]
            rest = orphans[homes < 0]
            if rest.size:
                ends = np.stack((2 * rest, 2 * rest + 1), axis=1).ravel()
                w = f.window_of[ends]
                sub = group_passages([f.names, f.years, f.wf[w], f.ws[w], f.we[w]])
                n_sites = f.site_file.shape[0]
                f.site_file = np.concatenate((f.site_file, sub.site_file))
                f.core_s, f.core_e = np.concatenate((f.core_s, sub.core_s)), np.concatenate((f.core_e, sub.core_e))
                f.anchors = np.concatenate((f.anchors, sub.anchors + n_sites))
                f.shown_s = np.concatenate((f.shown_s, sub.shown_s))
                f.rep_end = np.concatenate((f.rep_end, 2 * rest[sub.rep_end // 2] + sub.rep_end % 2))
                # a new family anchored on the very passage of another is that family
                anchor_keys = np.stack((f.site_file[f.anchors], f.core_s[f.anchors], f.core_e[f.anchors]), axis=1)
                _, first_of, inverse = np.unique(anchor_keys, axis=0, return_index=True, return_inverse=True)
                alias = first_of[inverse.ravel()]
                for i, j in enumerate(np.flatnonzero(homes < 0)):
                    lists[j] = alias[sub.list_members[sub.list_offsets[i]:sub.list_offsets[i + 1]] + n_families]
            offsets = np.concatenate(([0], np.cumsum([x.shape[0] for x in lists]))).astype(np.int64)
            list_offsets, list_members = _relisted(list_offsets, list_members, orphans, offsets,
                                                   np.concatenate(lists).astype(np.int64))
            counts.update(orphans=int(orphans.size), homed=int((homes >= 0).sum()))
        # the places left, as projected where they are unchanged
        _, (qg, qf, qs, qe, _) = placed(list_offsets, list_members)
        same = family_places.same_places(pg, pf, ps, pe, qg, qf, qs, qe)
        shown = [values[np.maximum(same, 0)] for values in shown]
        new = same < 0
        if new.any():
            fresh = family_places.project(f.names, passages(), qg[new], qf[new], qs[new], qe[new], workers,
                                          _label(4, "projecting the places that changed"))
            for values, update in zip(shown, fresh):
                values[new] = update
        n_all, n_files = f.anchors.shape[0], len(f.names)
        before = _document_keys(f.list_offsets, f.list_members, f.window_of, f.wf, n_files)
        after = _document_keys(list_offsets, list_members, f.window_of, f.wf, n_files)
        documents = np.bincount(after // n_files, minlength=n_all)
        kept = documents > 0
        counts.update(left=int(np.setdiff1d(before, after).size), went=int((~kept[:n_families]).sum()),
                      founded=int(kept[n_families:].sum()), joined=int(np.setdiff1d(after[after // n_files < n_families], before).size))
        renumber = np.cumsum(kept) - 1
        pg, pf, ps, pe = renumber[qg], qf, qs, qe
        f.anchors, f.shown_s, f.rep_end = f.anchors[kept], f.shown_s[kept], f.rep_end[kept]
        f.list_offsets, f.list_members = list_offsets, renumber[list_members]
        f.documents = documents[kept]
        # no two families show the same passage, the new ones' included
        f.shown_s = _kept_apart(f.site_file[f.anchors], f.core_s[f.anchors], f.core_e[f.anchors], f.shown_s)
        _order_rows(f)
    counts["places"] = family_places.write_places(places_file, f.names, pg, pf, ps, pe, shown, workers,
                                                  context_size, _label(4, "writing places"))
    f.window_of = f.wf = f.ws = f.we = None
    return counts


def merge_alignments(results_file: str, count: int, workers: int = 1, context_size: int = 300):
    """Group alignments into passage families, settle them on the places that reuse their
    passages (family_places), and write the group and places files."""
    corpus = read_ends(results_file, count)
    if not corpus[3].shape[0]:
        _status()
        print("  No passage groups found.")
        return None
    families = group_passages(corpus)
    places_file = os.path.join(os.path.dirname(results_file), "passage_group_places.jsonl")
    settled = settle_families(families, places_file, workers, context_size)

    # Both output steps read the records back, so they share one split of the file
    _status(_label(5))
    segments, bases = _segment_plan(results_file, count, workers)

    groups_file = os.path.join(os.path.dirname(results_file), "passage_group_source.jsonl")
    groups = write_group_sources(results_file, groups_file, families, segments, bases)

    temp_results_file = f"{results_file}.temp_final.lz4"
    rewrite_results(results_file, temp_results_file, families, count, segments, bases)
    os.remove(results_file)
    os.rename(temp_results_file, results_file)

    _status()
    print(f"  {groups:,} passage groups, written to {os.path.basename(groups_file)}")
    print(f"  {settled['places']:,} places reusing them, written to {os.path.basename(places_file)}")
    print(f"  {settled['left']:,} documents left a family they were in only through a partner, and "
          f"{settled['went']:,} families with them; of the {settled['orphans']:,} alignments then in none, "
          f"{settled['homed']:,} joined a family whose passage both their ends hold, bringing "
          f"{settled['joined']:,} documents, and the rest formed {settled['founded']:,} families of their own", flush=True)
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
            bar = stack.enter_context(tqdm(total=count, desc=_label(6), leave=False))
            _rewrite_records(input_file, output_file, group_lists, 0, bar=bar)
        return

    _SHARED.update(bases=bases, group_lists=group_lists)
    try:
        parts = _run_segments(_rewrite_segment, results_file, segments, count, _label(6))
    finally:
        # A module global, so a failed pass would otherwise leave the lists
        # pinned for the rest of the run.
        for key in ("bases", "group_lists"):
            _SHARED.pop(key, None)
    _status(_label(6))
    _join_parts(parts, target)


def write_group_sources(results_file: str, groups_file: str, families: _Families,
                        segments: list[tuple[int, int]], bases: dict[int, int]) -> int:
    """Write one row per family: its anchor's passage, from the earliest of its reuses when
    that is a few words before, with the metadata of the record that first has it, and
    `count`, the documents its alignments span. Returns the rows written.

    Rows come out in the order their records appear; the database load keys on group_id.
    """
    total = families.anchors.shape[0]
    if len(segments) < 2:
        with contextlib.ExitStack() as stack:
            input_file = stack.enter_context(lz4.frame.open(results_file))
            output_file = stack.enter_context(open(groups_file, "wb"))
            bar = stack.enter_context(tqdm(total=total, desc=_label(5), leave=False))
            _group_source_records(input_file, output_file, families, 0, bar=bar)
        return total

    _SHARED.update(bases=bases, families=families)
    try:
        parts = _run_segments(_group_source_segment, results_file, segments, total, _label(5))
    finally:
        for key in ("bases", "families"):
            _SHARED.pop(key, None)
    # Plain jsonl, not lz4, so the parts join as they are.
    _status(_label(5))
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
