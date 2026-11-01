"""Each passage extended past its first and last matching ngrams, as BLAST extends a hit.

A passage runs from the first ngram its two documents share to the last, so words at its
ends that never make an ngram -- stopwords, words too short, words spelt otherwise -- are
left out though both texts have them. Each end is extended outward on both documents at
once: their text as the passage shows it, folded (case, accents, u/v, i/j/y and long s
alike), is aligned letter by letter with affine gaps, and the extension stops where the
score falls X_DROP below the best it reached (BLAST's X-drop). It is then cut back to a
word boundary, and past any last words the two sides do not share.

An extension is computed with the two documents in an order fixed by their text, not by
which is the source, so comparing a pair the other way round extends it the same way.
"""
import unicodedata

import numpy as np
import regex
from numba import njit

MATCH, MISMATCH, GAP_OPEN, GAP_EXTEND = 1, -1, 2, 1
X_DROP = 15
WINDOW = 1024  # bytes read past an end, grown while an extension runs to the window's end
MAX_WINDOW = 1 << 20
ALIKE = 0.6  # share of their letters two last words need in common for an extension to end on them
LONG_WORD = 64

_NONE = -(1 << 40)


def _tables():
    """By code point of the Basic Multilingual Plane: the character folded, and its part in
    a word (0 none, 1 in a run of letters or digits, 2 a word by itself, as in Chinese)."""
    fold = np.arange(0x10000, dtype=np.int32)
    kind = np.zeros(0x10000, np.uint8)
    alone = regex.compile(r"[\p{Han}\p{Hiragana}\p{Katakana}]")
    for code in range(0x10000):
        c = chr(code)
        if c.isalnum():
            kind[code] = 2 if alone.match(c) else 1
        folded = "".join(x for x in unicodedata.normalize("NFKD", c.lower()) if not unicodedata.combining(x))
        folded = folded.replace("v", "u").replace("j", "i").replace("y", "i")
        if len(folded) == 1:
            fold[code] = ord(folded)
        elif len(c.lower()) == 1:
            fold[code] = ord(c.lower())
    return fold, kind


FOLD, KIND = _tables()


@njit(cache=True)
def _entity(raw, i, j):
    """The character of the reference raw[i:j+1] ("&...;"), or -1 if it is not one."""
    k = i + 1
    if k < j and raw[k] == 35:  # '#'
        k += 1
        base = 10
        if k < j and (raw[k] == 120 or raw[k] == 88):  # 'x'
            base = 16
            k += 1
        if k == j:
            return -1
        value = 0
        while k < j:
            b = raw[k]
            if 48 <= b <= 57:
                digit = b - 48
            elif base == 16 and 97 <= (b | 32) <= 102:
                digit = (b | 32) - 87
            else:
                return -1
            value = value * base + digit
            if value > 0x10FFFF:
                return 0xFFFD
            k += 1
        return value if value > 0 else 0xFFFD
    n = j - k
    if n == 3 and raw[k] == 97 and raw[k + 1] == 109 and raw[k + 2] == 112:
        return 38  # amp
    if n == 2 and raw[k + 1] == 116 and (raw[k] == 108 or raw[k] == 103):
        return 60 if raw[k] == 108 else 62  # lt, gt
    if n == 4 and raw[k] == 113 and raw[k + 1] == 117 and raw[k + 2] == 111 and raw[k + 3] == 116:
        return 34  # quot
    if n == 4 and raw[k] == 97 and raw[k + 1] == 112 and raw[k + 2] == 111 and raw[k + 3] == 115:
        return 39  # apos
    if n == 4 and raw[k] == 110 and raw[k + 1] == 98 and raw[k + 2] == 115 and raw[k + 3] == 112:
        return 32  # nbsp
    for m in range(k, j):
        b = raw[m]
        if not (48 <= b <= 57 or 97 <= (b | 32) <= 122):
            return -1
    return 0xFFFD if n > 0 else -1


@njit(cache=True)
def _text(raw, fold, kind, cut):
    """The text of raw bytes as a passage shows it -- tags dropped, references as their
    character, white space as one space -- folded, by character: (codes, kinds, starts,
    ends), starts and ends being byte offsets into raw. With `cut`, raw begins at an
    arbitrary byte, perhaps inside a character or a tag."""
    n = len(raw)
    codes = np.empty(n, np.int32)
    kinds = np.empty(n, np.uint8)
    starts = np.empty(n, np.int64)
    ends = np.empty(n, np.int64)
    k = 0
    i = 0
    if cut:
        while i < n and (raw[i] & 0xC0) == 0x80:
            i += 1
        j = i
        while j < n and raw[j] != 60 and raw[j] != 62:
            j += 1
        if j < n and raw[j] == 62:
            i = j + 1
    while i < n:
        b = raw[i]
        if b == 60:  # a tag
            j = i + 1
            while j < n and raw[j] != 62:
                j += 1
            if j >= n:
                break
            i = j + 1
            continue
        width = 1
        code = -1
        if b == 32 or b == 9 or b == 10 or b == 13 or b == 12:
            code = 32
        elif b == 0xC2 and i + 1 < n and raw[i + 1] == 0xA0:
            code, width = 32, 2
        elif b == 38:  # '&'
            j = i + 1
            while j < n and j - i <= 12 and raw[j] != 59 and raw[j] != 38 and raw[j] != 60:
                j += 1
            if j < n and raw[j] == 59:
                code = _entity(raw, i, j)
                if code >= 0:
                    width = j + 1 - i
            if code < 0:
                code = 38
        elif b < 0x80:
            code = b
        else:
            if b >> 5 == 6:
                width, code = 2, b & 0x1F
            elif b >> 4 == 14:
                width, code = 3, b & 0x0F
            elif b >> 3 == 30:
                width, code = 4, b & 0x07
            else:
                width, code = 1, 0xFFFD
            if i + width > n:
                break
            for m in range(i + 1, i + width):
                if (raw[m] & 0xC0) != 0x80:
                    width, code = 1, 0xFFFD
                    break
                code = (code << 6) | (raw[m] & 0x3F)
        if code == 32:
            if k > 0 and codes[k - 1] == 32:
                ends[k - 1] = i + width
                i += width
                continue
            kinds[k] = 0
        elif code < 0x10000:
            kinds[k] = kind[code]
            code = fold[code]
        else:
            kinds[k] = 1
        codes[k] = code
        starts[k] = i
        ends[k] = i + width
        k += 1
        i += width
    return codes[:k], kinds[:k], starts[:k], ends[:k]


@njit(cache=True)
def _xdrop(a, b, x):
    """The best-scoring alignment of the beginnings of a and b, letters scored MATCH or
    MISMATCH and gaps GAP_OPEN plus GAP_EXTEND a letter, cells more than x below the best
    so far given up: (score, i, j, open), taking a[:i] and b[:j], open when it was still
    going at the end of either."""
    n, m = len(a), len(b)
    H = np.full(m + 1, _NONE, np.int64)
    F = np.full(m + 1, _NONE, np.int64)
    Hn = np.full(m + 1, _NONE, np.int64)
    Fn = np.full(m + 1, _NONE, np.int64)
    best, bi, bj = 0, 0, 0
    H[0] = 0
    lo, hi = 0, 0
    e = _NONE
    for j in range(1, m + 1):  # the first row: gaps in a only
        e = max(H[j - 1] - GAP_OPEN - GAP_EXTEND, e - GAP_EXTEND)
        if e < -x:
            break
        H[j] = e
        hi = j
    still = hi == m
    for i in range(1, n + 1):
        new_lo, new_hi = -1, -1
        e = _NONE
        j = lo
        while j <= m:
            floor = best - x
            d = _NONE
            if j >= 1 and lo <= j - 1 <= hi and H[j - 1] > _NONE:
                d = H[j - 1] + (MATCH if a[i - 1] == b[j - 1] else MISMATCH)
            f = _NONE
            if lo <= j <= hi and H[j] > _NONE:
                f = max(H[j] - GAP_OPEN - GAP_EXTEND, F[j] - GAP_EXTEND)
            h = max(d, f, e)
            if f < floor:
                f = _NONE
            if h < floor:
                h = _NONE
            Hn[j], Fn[j] = h, f
            if h > _NONE:
                if new_lo < 0:
                    new_lo = j
                new_hi = j
                if h > best:
                    best, bi, bj = h, i, j
            e = max(h - GAP_OPEN - GAP_EXTEND, e - GAP_EXTEND) if h > _NONE else (e - GAP_EXTEND if e > _NONE else _NONE)
            if e < floor:
                e = _NONE
            if j > hi and h == _NONE and e == _NONE:
                break
            j += 1
        if new_lo < 0:
            return best, bi, bj, still
        if new_hi == m or i == n:
            still = True
        H, Hn = Hn, H
        F, Fn = Fn, F
        lo, hi = new_lo, new_hi
    return best, bi, bj, still


@njit(cache=True)
def _last_word(kinds, end):
    """(start, end) of the last word before `end`, or (-1, -1)."""
    k = end
    while k > 0 and kinds[k - 1] == 0:
        k -= 1
    if k == 0:
        return -1, -1
    if kinds[k - 1] == 2:
        return k - 1, k
    s = k
    while s > 0 and kinds[s - 1] == 1:
        s -= 1
    return s, k


@njit(cache=True)
def _alike(a, s1, e1, b, s2, e2):
    """Two words with at least ALIKE of their letters in common, in order; past LONG_WORD
    letters, which no word of a language has, only the same letters."""
    n, m = e1 - s1, e2 - s2
    if n > LONG_WORD or m > LONG_WORD:
        if n != m:
            return False
        for k in range(n):
            if a[s1 + k] != b[s2 + k]:
                return False
        return True
    row = np.zeros(m + 1, np.int64)
    for i in range(n):
        diagonal = 0
        for j in range(m):
            above = row[j + 1]
            row[j + 1] = diagonal + 1 if a[s1 + i] == b[s2 + j] else max(above, row[j])
            diagonal = above
    return 2 * row[m] >= ALIKE * (n + m)


@njit(cache=True)
def _trim(a, ka, i, b, kb, j):
    """i and j cut back to the ends of the last two words that are alike: a word the
    extension stops inside does not count, nor what follows the last word."""
    if 0 < i < len(a) and ka[i - 1] == 1 and ka[i] == 1:
        i = _last_word(ka, i)[0]
    if 0 < j < len(b) and kb[j - 1] == 1 and kb[j] == 1:
        j = _last_word(kb, j)[0]
    while True:
        s1, e1 = _last_word(ka, i)
        s2, e2 = _last_word(kb, j)
        if s1 < 0 or s2 < 0:
            return 0, 0
        if _alike(a, s1, e1, b, s2, e2):
            return e1, e2
        i, j = s1, s2


def _side(buf, at, limit, window, before):
    """One document's text from `at` outward: (codes, kinds, starts, ends, base, whole),
    reversed when `before`, offsets relative to `base`; `whole` when it reaches `limit`."""
    if before:
        base = max(limit, at - window)
        raw = np.frombuffer(buf[base:at], np.uint8)
        codes, kinds, starts, ends = _text(raw, FOLD, KIND, base > limit)
        return codes[::-1].copy(), kinds[::-1].copy(), starts[::-1], ends[::-1], base, base == limit
    stop = min(limit, at + window)
    raw = np.frombuffer(buf[at:stop], np.uint8)
    codes, kinds, starts, ends = _text(raw, FOLD, KIND, False)
    return codes, kinds, starts, ends, at, stop == limit


def _reach(buf_a, at_a, limit_a, buf_b, at_b, limit_b, before):
    """How far one end of a passage extends in each document: the new byte offsets."""
    window = WINDOW
    while True:
        a = _side(buf_a, at_a, limit_a, window, before)
        b = _side(buf_b, at_b, limit_b, window, before)
        swap = a[0].tobytes() > b[0].tobytes()
        first, second = (b, a) if swap else (a, b)
        _, i, j, still = _xdrop(first[0], second[0], X_DROP)
        i, j = _trim(first[0], first[1], i, second[0], second[1], j)
        if swap:
            i, j = j, i
        if not still or window >= MAX_WINDOW or (a[5] and b[5]):
            break
        window *= 4
    if before:
        return (a[4] + int(a[2][i - 1]) if i else at_a), (b[4] + int(b[2][j - 1]) if j else at_b)
    return (a[4] + int(a[3][i - 1]) if i else at_a), (b[4] + int(b[3][j - 1]) if j else at_b)


def extend(source_buf, source_limits, target_buf, target_limits, source_start, source_end,
           target_start, target_end):
    """A passage's byte offsets in both documents, each end extended as far as both share it.
    `*_limits` are the (start, end) bytes of each document's text."""
    if not source_buf or not target_buf:
        return source_start, source_end, target_start, target_end
    source_start, target_start = _reach(source_buf, source_start, source_limits[0],
                                        target_buf, target_start, target_limits[0], True)
    source_end, target_end = _reach(source_buf, source_end, source_limits[1],
                                    target_buf, target_end, target_limits[1], False)
    return source_start, source_end, target_start, target_end


def _same(a, b):
    """Two passages of one pair that are one once extended: each side's overlap is at least
    half the shorter of the two -- one extension ran over the other passage -- not passages
    that only touch, as the copies of a phrase repeated in one document do."""
    for lo, hi in ((0, 1), (2, 3)):
        overlap = min(a[hi], b[hi]) - max(a[lo], b[lo])
        if 2 * overlap < min(a[hi] - a[lo], b[hi] - b[lo]) or overlap <= 0:
            return False
    return True


def merged(spans):
    """Passages of one document pair that are one once extended (`_same`), as one: grouped
    with whatever passage they are one with, which gives the same groups in any order."""
    spans = sorted(set(spans))
    parent = list(range(len(spans)))

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for k, span in enumerate(spans):
        for m in range(k + 1, len(spans)):
            if spans[m][0] >= span[1]:
                break  # sorted by start: no later passage overlaps this one in the source
            if _same(span, spans[m]):
                parent[root(m)] = root(k)
    groups = {}
    for k, span in enumerate(spans):
        s0, s1, t0, t1 = groups.get(root(k), span)
        groups[root(k)] = (min(s0, span[0]), max(s1, span[1]), min(t0, span[2]), max(t1, span[3]))
    return sorted(groups.values())
