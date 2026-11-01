"""Which characters two texts share: seed, chain, align.

Seeds are the runs of words both texts share, tiled longest first and found in any order.
Runs that follow each other in both texts make a chain, and chains close to each other make
one stretch. Inside a stretch words are matched first; what lies between matched words is
then compared letter by letter with diff-match-patch, on text folded as words are compared.
"""

import bisect
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
import regex
from diff_match_patch import diff_match_patch

# A word, or a single character of the scripts written without spaces, as split_words finds them
REUSE_WORD = regex.compile(
    r"[\p{Han}\p{Hiragana}\p{Katakana}]|[[\p{L}\p{N}]--[\p{Han}\p{Hiragana}\p{Katakana}]]+", regex.V1)
REUSE_GAP = 4  # words two shared runs may be apart, on both sides, and still be one chain
BRIDGE = 16  # words two chains may be apart, on both sides, and still be one stretch
MARGIN = 3  # words around a stretch compared too
SEED_BUDGET = 50  # seed runs per word of two texts before they are first cut at words each uses once

UNMARKED = str.maketrans("ſ", "s")  # letters told apart by typography alone, not marked respelt

DMP = diff_match_patch()
DMP.Diff_Timeout = 0


@lru_cache(maxsize=200_000)
def reuse_key(word: str) -> str:
    """A word as compared across spellings: lowercase, no accents, u/v, i/j/y and ſ/s alike."""
    word = "".join(c for c in unicodedata.normalize("NFKD", word.lower()) if not unicodedata.combining(c))
    return word.replace("v", "u").replace("j", "i").replace("y", "i")


class FoldTable(dict):
    """Each character as reuse_key compares it, kept to one character."""

    def __missing__(self, code: int) -> str:
        c = chr(code)
        folded = reuse_key(c)
        if len(folded) != 1:
            folded = c.lower() if len(c.lower()) == 1 else c
        self[code] = folded
        return folded


FOLD = FoldTable()


def codes(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-32-le", "surrogatepass"), np.uint32)


ALONE = regex.compile(r"[\p{Han}\p{Hiragana}\p{Katakana}]")
IN_WORD = regex.compile(r"[\p{L}\p{N}]")
# by code point, filled in as characters are met: each one folded, and its part in
# REUSE_WORD's words (2 a word by itself, 1 in a run, 0 none)
FOLDED = np.arange(0x110000, dtype=np.uint32)
KIND = np.zeros(0x110000, np.uint8)
KNOWN = np.zeros(0x110000, np.bool_)


def read(text: str) -> tuple[np.ndarray, np.ndarray]:
    """A text's characters as folded code points, and their kinds."""
    points = codes(text)
    for code in np.unique(points[~KNOWN[points]]).tolist():
        c = chr(code)
        FOLDED[code] = ord(FOLD[code])
        KIND[code] = 2 if ALONE.match(c) else 1 if IN_WORD.match(c) else 0
        KNOWN[code] = True
    return FOLDED[points], KIND[points]


HASH_BASE = np.uint64(0x100000001B3)
HASH_MIX = np.uint64(0x9E3779B97F4A7C15)


def split_words(text: str) -> tuple[str, list[int], list[int], np.ndarray]:
    """A text folded, where REUSE_WORD's words start and end in it, and a number for each word as folded."""
    if not text:
        return "", [], [], np.zeros(0, np.uint64)
    points, kinds = read(text)
    folded = points.tobytes().decode("utf-32-le", "surrogatepass")
    run, alone = kinds == 1, kinds == 2
    starts = np.flatnonzero((run & ~np.concatenate(([False], run[:-1]))) | alone)
    ends = np.flatnonzero((run & ~np.concatenate((run[1:], [False]))) | alone) + 1
    if not len(starts):
        return folded, [], [], np.zeros(0, np.uint64)
    # each letter weighted by a power of HASH_BASE from its word's start, summed by word
    begun = np.zeros(len(text), np.int64)
    begun[starts] = 1
    offset = np.arange(len(text)) - starts[np.maximum(np.cumsum(begun) - 1, 0)]
    lengths = ends - starts
    powers = np.cumprod(np.concatenate(([1], np.full(lengths.max() - 1, HASH_BASE))).astype(np.uint64))
    letters = points.astype(np.uint64) + np.uint64(1)
    weighted = np.where(run | alone, letters * powers[np.clip(offset, 0, len(powers) - 1)], np.uint64(0))
    keys = np.add.reduceat(weighted, starts) * HASH_MIX + lengths.astype(np.uint64)
    return folded, starts.tolist(), ends.tolist(), keys


def pair_keys(keys: np.ndarray) -> np.ndarray:
    return keys[:-1] * HASH_MIX + keys[1:]


def tiles(words: list[int], pairs: dict, other: list[int], js=None) -> list[tuple[int, int, int]]:
    """Shared runs of two words or more as (i, j, size), tiled longest first, no word on either
    side in two; `js` are where in `other` the word pairs the runs may start with are, if known."""
    runs = []
    for j in range(len(other) - 1) if js is None else js:
        for i in pairs.get((other[j], other[j + 1]), ()):
            if i and j and words[i - 1] == other[j - 1]:
                continue  # inside a longer run
            k = 2
            while i + k < len(words) and j + k < len(other) and words[i + k] == other[j + k]:
                k += 1
            runs.append((k, i, j))
    runs.sort(key=lambda run: (-run[0], run[1]))
    used, used_other = bytearray(len(words)), bytearray(len(other))
    found = []
    for length, i, j in runs:
        # the longest stretch of the run that neither side has used yet
        best, start = (0, 0), None
        for t in range(length + 1):
            free = t < length and not used[i + t] and not used_other[j + t]
            if free and start is None:
                start = t
            elif not free and start is not None:
                best = max(best, (t - start, -start))
                start = None
        size, offset = best[0], -best[1]
        if size < 2:
            continue
        for t in range(offset, offset + size):
            used[i + t] = used_other[j + t] = 1
        found.append((i + offset, j + offset, size))
    return sorted(found)


def pair_index(words: list[str]) -> dict:
    pairs = defaultdict(list)
    for i in range(len(words) - 1):
        pairs[(words[i], words[i + 1])].append(i)
    return pairs


def split_tiles(words: list[int], other: list[int], pairs: dict | None = None, js=None) -> list[tuple[int, int, int]]:
    """tiles(), but where two long texts share frequent word pairs, which would start too many
    runs, they are first cut at words each uses once, kept in an order both share."""
    pairs = pairs if pairs is not None else pair_index(words)
    budget = SEED_BUDGET * (len(words) + len(other))
    most = max(map(len, pairs.values()), default=0)
    starting = zip(other, other[1:]) if js is None else ((other[j], other[j + 1]) for j in js)
    if len(other) * most <= budget or sum(len(pairs.get(pair, ())) for pair in starting) <= budget:
        return tiles(words, pairs, other, js)
    count, count_other = Counter(words), Counter(other)
    where = {w: j for j, w in enumerate(other) if count_other[w] == 1}
    anchors = [(i, where[w]) for i, w in enumerate(words) if count[w] == 1 and w in where]
    tails, at, back = [], [], [None] * len(anchors)  # longest increasing run of anchors
    for k, (_, j) in enumerate(anchors):
        lo = bisect.bisect_left(tails, j)
        back[k] = at[lo - 1] if lo else None
        if lo == len(tails):
            tails.append(j)
            at.append(k)
        else:
            tails[lo] = j
            at[lo] = k
    chain, k = [], at[-1] if at else None
    while k is not None:
        chain.append(anchors[k])
        k = back[k]
    chain.reverse()
    if not chain:
        return []
    found, pi, pj = [], 0, 0
    for i, j in chain:
        if i < pi or j < pj:
            continue
        s = 0  # the anchor grown over the words around it both share
        while i - s - 1 >= pi and j - s - 1 >= pj and words[i - s - 1] == other[j - s - 1]:
            s += 1
        e = 1
        while i + e < len(words) and j + e < len(other) and words[i + e] == other[j + e]:
            e += 1
        found += [(ti + pi, tj + pj, size) for ti, tj, size in split_tiles(words[pi:i - s], other[pj:j - s])]
        if s + e >= 2:
            found.append((i - s, j - s, s + e))
        pi, pj = i + e, j + e
    found += [(ti + pi, tj + pj, size) for ti, tj, size in split_tiles(words[pi:], other[pj:])]
    return sorted(found)


def stretches(found: list, alone: int, chained: int) -> list[list[tuple[int, int, int]]]:
    """Tiles in chains, kept if a tile is `alone` words long or the chain `chained`, and kept chains close to each other joined."""
    chains, chain = [], []
    for tile in found + [None]:
        if chain and tile:
            i, j, size = chain[-1]
            if 0 <= tile[0] - (i + size) <= REUSE_GAP and 0 <= tile[1] - (j + size) <= REUSE_GAP:
                chain.append(tile)
                continue
        if chain and (max(t[2] for t in chain) >= alone or sum(t[2] for t in chain) >= chained):
            chains.append(chain)
        chain = [tile]
    joined = []
    for chain in chains:
        if joined:
            i, j, size = joined[-1][-1]
            if 0 <= chain[0][0] - (i + size) <= BRIDGE and 0 <= chain[0][1] - (j + size) <= BRIDGE:
                joined[-1] = joined[-1] + chain
                continue
        joined.append(chain)
    return joined


def out_of_order(found: list) -> set[int]:
    """Stretches outside the largest set, in words, that both texts have in the same order."""
    ends = sorted({s[-1][1] + s[-1][2] for s in found})
    tree = [(0, -1)] * (len(ends) + 1)  # prefix maxima of (words, stretch) by where stretches end in the other text
    score, back = [0] * len(found), [-1] * len(found)
    for k, stretch in enumerate(found):
        best, pos = (0, -1), bisect.bisect_right(ends, stretch[0][1])
        while pos:
            best = max(best, tree[pos])
            pos -= pos & -pos
        score[k], back[k] = best[0] + sum(t[2] for t in stretch), best[1]
        pos = bisect.bisect_left(ends, stretch[-1][1] + stretch[-1][2]) + 1
        while pos <= len(ends):
            tree[pos] = max(tree[pos], (score[k], k))
            pos += pos & -pos
    keep, k = set(), max(range(len(found)), key=score.__getitem__, default=-1)
    while k != -1:
        keep.add(k)
        k = back[k]
    return set(range(len(found))) - keep


@lru_cache(maxsize=1 << 16)
def matched_words(a: tuple, b: tuple) -> list[tuple[int, int, int]]:
    """Blocks (i, j, size) of the words two short lists share, in order."""
    ids = {}
    sa = "".join(chr(0xE000 + ids.setdefault(w, len(ids))) for w in a)
    sb = "".join(chr(0xE000 + ids.setdefault(w, len(ids))) for w in b)
    blocks, i, j = [], 0, 0
    for op, text in DMP.diff_main(sa, sb, False):
        if op == 0:
            blocks.append((i, j, len(text)))
        if op != 1:
            i += len(text)
        if op != -1:
            j += len(text)
    return blocks


@lru_cache(maxsize=1 << 16)
def matched_letters(a: str, b: str, open_start: bool = False, open_end: bool = False) -> list[tuple[int, int, int]]:
    """Blocks (i, j, size) of the letters two short stretches share, as the app's diff cleans
    them up; at an open end, as the cleanup does inside, a match no longer than the edits next to it is dropped."""
    diffs = DMP.diff_main(a, b, False)
    DMP.diff_cleanupSemantic(diffs)
    for end, inward in ((open_start, range(1, len(diffs))), (open_end, range(len(diffs) - 2, -1, -1))):
        edge = inward.start - inward.step
        if end and len(diffs) > 1 and diffs[edge][0] == 0:
            edited = {-1: 0, 1: 0}
            for k in inward:
                if diffs[k][0] == 0:
                    break
                edited[diffs[k][0]] += len(diffs[k][1])
            if len(diffs[edge][1]) <= max(edited.values()):
                diffs[edge] = (2, diffs[edge][1])  # neither side's
    blocks, i, j = [], 0, 0
    for op, text in diffs:
        if op == 0:
            blocks.append((i, j, len(text)))
        if op != 1:
            i += len(text)
        if op != -1:
            j += len(text)
    return blocks


class Shared:
    """What two texts share: shared[side] by character, with detail, respelt[side] (shared
    but spelt otherwise) and moved[side] (shared in another order)."""

    def __init__(self, la: int, lb: int, detail: bool):
        self.shared = (np.zeros(la, np.bool_), np.zeros(lb, np.bool_))
        if detail:
            self.respelt = (np.zeros(la, np.bool_), np.zeros(lb, np.bool_))
            self.moved = (np.zeros(la, np.bool_), np.zeros(lb, np.bool_))


class Passage:
    """A text to compare others with, its words indexed once."""

    def __init__(self, text: str):
        self.text = text or ""
        self.folded, starts, ends, keys = split_words(self.text)
        self.spans = list(zip(starts, ends))
        self.words = keys.tolist()
        # characters of a script without spaces are more often shared by chance than words
        unspaced = sum(e - s == 1 and not self.folded[s].isascii() for s, e in self.spans) * 2 > len(self.words)
        self.alone, self.chained = (4, 6) if unspaced else (3, 4)
        self.pairs = pair_index(self.words)
        self.pair_keys = np.unique(pair_keys(keys))

    def compare(self, other_text: str, detail: bool = False) -> Shared:
        other_text = other_text or ""
        found = Shared(len(self.text), len(other_text), detail)
        folded, starts, ends, keys = split_words(other_text)
        other = keys.tolist()
        js = np.flatnonzero(np.isin(pair_keys(keys), self.pair_keys)).tolist() if len(keys) > 1 else []
        joined = stretches(split_tiles(self.words, other, self.pairs, js), self.alone, self.chained)
        moved = out_of_order(joined) if detail and len(joined) > 1 else set()
        ma, mb = found.shared

        def mark(a0, b0, size, respelt, move):
            ma[a0:a0 + size] = True
            mb[b0:b0 + size] = True
            if detail:
                if respelt:
                    a, b = self.text[a0:a0 + size].translate(UNMARKED), other_text[b0:b0 + size].translate(UNMARKED)
                    differ = codes(a) != codes(b)
                    found.respelt[0][a0:a0 + size] |= differ
                    found.respelt[1][b0:b0 + size] |= differ
                if move:
                    found.moved[0][a0:a0 + size] = True
                    found.moved[1][b0:b0 + size] = True

        def compare(a0, a1, b0, b1, move, open_start=False, open_end=False):
            if a0 >= a1 or b0 >= b1:
                return
            fa, fb = self.folded[a0:a1], folded[b0:b1]
            if fa == fb:
                mark(a0, b0, a1 - a0, self.text[a0:a1] != other_text[b0:b1], move)
                return
            for i, j, size in matched_letters(fa, fb, open_start, open_end):
                mark(a0 + i, b0 + j, size, self.text[a0 + i:a0 + i + size] != other_text[b0 + j:b0 + j + size], move)

        def word(a0, a1, b0, b1, move):
            """A word matched whole, whichever of its letters the letter comparison missed."""
            if detail:
                for side, s, e in ((0, a0, a1), (1, b0, b1)):
                    found.respelt[side][s:e] |= ~found.shared[side][s:e]
                    if move:
                        found.moved[side][s:e] = True
            ma[a0:a1] = True
            mb[b0:b1] = True

        tiled, tiled_other = bytearray(len(self.words)), bytearray(len(other))
        for stretch in joined:
            for i, j, size in stretch:
                tiled[i:i + size] = tiled_other[j:j + size] = b"\x01" * size

        def margin(used, at, step, end):
            """Up to MARGIN words from `at` towards `end`, none of another stretch's tiles."""
            for _ in range(MARGIN):
                if at == end or used[at + min(step, 0)]:
                    break
                at += step
            return at

        for k, stretch in enumerate(joined):
            move = k in moved
            i0, j0 = stretch[0][0], stretch[0][1]
            i1, j1 = stretch[-1][0] + stretch[-1][2], stretch[-1][1] + stretch[-1][2]
            w0, v0 = margin(tiled, i0, -1, 0), margin(tiled_other, j0, -1, 0)
            w1, v1 = margin(tiled, i1, 1, len(self.words)), margin(tiled_other, j1, 1, len(other))
            # the tiles, and the words of what lies around and between them both share
            gaps = [((w0, i0), (v0, j0))]
            gaps += [((p[0] + p[2], t[0]), (p[1] + p[2], t[1])) for p, t in zip(stretch, stretch[1:])]
            gaps.append(((i1, w1), (j1, v1)))
            blocks = list(stretch)
            for (x0, x1), (y0, y1) in gaps:
                if x0 < x1 and y0 < y1:
                    blocks += [(x0 + i, y0 + j, size)
                               for i, j, size in matched_words(tuple(self.words[x0:x1]), tuple(other[y0:y1]))]
            blocks.sort()
            a_at, b_at = self.spans[w0][0], starts[v0]
            for n, (i, j, size) in enumerate(blocks):
                compare(a_at, self.spans[i][0], b_at, starts[j], move, open_start=not n)  # what lies since the last block
                a0, a1 = self.spans[i][0], self.spans[i + size - 1][1]
                b0, b1 = starts[j], ends[j + size - 1]
                if self.folded[a0:a1] == folded[b0:b1]:
                    mark(a0, b0, a1 - a0, self.text[a0:a1] != other_text[b0:b1], move)
                else:  # word by word, and what lies between them
                    for t in range(size):
                        (s, e), u, v = self.spans[i + t], starts[j + t], ends[j + t]
                        if t:
                            compare(self.spans[i + t - 1][1], s, ends[j + t - 1], u, move)
                        compare(s, e, u, v, move)
                        word(s, e, u, v, move)
                a_at, b_at = a1, b1
            compare(a_at, self.spans[w1 - 1][1], b_at, ends[v1 - 1], move, open_end=True)
        return found


def runs(text: str, values: np.ndarray) -> list[list]:
    """The text cut where values change: [piece, value] in order."""
    if not text:
        return []
    cuts = [0, *(np.flatnonzero(values[1:] != values[:-1]) + 1).tolist(), len(text)]
    return [[text[s:e], int(values[s])] for s, e in zip(cuts, cuts[1:])]


def reuse_depth(passage: str, rows, anchor_doc) -> dict:
    """How many documents other than the anchor's share each character of a group's passage.

    Every span of a document in the group's alignments is compared with the passage.
    `pieces` are the passage cut where that count changes, each with its count.
    """
    passage = passage or ""
    spans = defaultdict(dict)
    for row in rows:
        for side in ("source", "target"):
            if row[f"{side}_doc_id"] != anchor_doc:
                span = (row[f"{side}_start_byte"], row[f"{side}_end_byte"])
                spans[row[f"{side}_filename"]][span] = row[f"{side}_passage"]
    found = Passage(passage)
    depth = np.zeros(len(passage), np.int32)
    for texts in spans.values():
        reused = np.zeros(len(passage), np.bool_)
        for text in texts.values():
            reused |= found.compare(text).shared[0]
        depth += reused
    return {"pieces": runs(passage, depth), "documents": len(spans)}


def differences(source: str, target: str) -> dict:
    """Two passages cut into what they share, spelt alike or otherwise, in the same order or
    not, and what is the one's alone: [piece, "same" | "respelt" | "moved" | "different"]."""
    found = Passage(source).compare(target, detail=True)
    out = {}
    for side, text in enumerate((source or "", target or "")):
        kinds = np.where(found.shared[side], np.where(found.respelt[side], 2, 1), 0)
        kinds[found.moved[side]] = 3
        out["source" if side == 0 else "target"] = [
            [piece, ("different", "same", "respelt", "moved")[kind]] for piece, kind in runs(text, kinds)]
    return out
