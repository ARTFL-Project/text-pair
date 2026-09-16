"""Port of Go 1.22 `sort.Slice` (pdqsort, sort/zsortfunc.go) and of compareNgrams'
`getFiles` (main.go:232-297), which fixes the document order the whole run depends on.

The pdqsort port is load-bearing: getFiles' comparator is non-transitive, because when
`strconv.Atoi(year)` fails for either document it falls back to a string compare of
document IDs, inconsistent with the (year, int(docID)) order used otherwise. The resulting
permutation therefore depends on the sort algorithm, and Python's timsort disagrees with
Go's pdqsort. frantext has 34 such documents, eccotcp 1.
"""
import os
import re

_INT_RE = re.compile(r"^[+-]?\d+$")
UNKNOWN_HINT, INCREASING_HINT, DECREASING_HINT = 0, 1, 2
_M64 = (1 << 64) - 1


def go_atoi(s):
    """strconv.Atoi: no leading/trailing space, optional sign, decimal digits only."""
    return int(s) if _INT_RE.match(s) else None


class _Xorshift:
    __slots__ = ("s",)

    def __init__(self, seed):
        self.s = seed & _M64

    def next(self):
        s = self.s
        s ^= (s << 13) & _M64
        s ^= s >> 7
        s ^= (s << 17) & _M64
        self.s = s
        return s


def _bits_len(n):
    return n.bit_length()


def go_sort_slice(data, less):
    """sort.Slice: pdqsort_func(lessSwap, 0, len, bits.Len(uint(len))). Sorts `data` in place."""
    n = len(data)

    def swap(i, j):
        data[i], data[j] = data[j], data[i]

    _pdqsort(less, swap, 0, n, _bits_len(n))
    return data


def _insertion_sort(less, swap, a, b):
    for i in range(a + 1, b):
        j = i
        while j > a and less(j, j - 1):
            swap(j, j - 1)
            j -= 1


def _sift_down(less, swap, lo, hi, first):
    root = lo
    while True:
        child = 2 * root + 1
        if child >= hi:
            break
        if child + 1 < hi and less(first + child, first + child + 1):
            child += 1
        if not less(first + root, first + child):
            return
        swap(first + root, first + child)
        root = child


def _heap_sort(less, swap, a, b):
    first, lo, hi = a, 0, b - a
    for i in range((hi - 1) // 2, -1, -1):
        _sift_down(less, swap, i, hi, first)
    for i in range(hi - 1, -1, -1):
        swap(first, first + i)
        _sift_down(less, swap, lo, i, first)


def _reverse_range(swap, a, b):
    i, j = a, b - 1
    while i < j:
        swap(i, j)
        i += 1
        j -= 1


def _next_power_of_two(length):
    return 1 << _bits_len(length)


def _break_patterns(swap, a, b):
    length = b - a
    if length >= 8:
        random = _Xorshift(length)
        modulus = _next_power_of_two(length)
        idx = a + (length // 4) * 2 - 1
        while idx <= a + (length // 4) * 2 + 1:
            other = random.next() & (modulus - 1)
            if other >= length:
                other -= length
            swap(idx, a + other)
            idx += 1


def _order2(less, box, a, b):
    if less(b, a):
        box[0] += 1
        return b, a
    return a, b


def _median(less, box, a, b, c):
    a, b = _order2(less, box, a, b)
    b, c = _order2(less, box, b, c)
    a, b = _order2(less, box, a, b)
    return b


def _median_adjacent(less, box, a):
    return _median(less, box, a - 1, a, a + 1)


def _choose_pivot(less, a, b):
    SHORTEST_NINTHER = 50
    MAX_SWAPS = 4 * 3
    l = b - a
    box = [0]
    i = a + l // 4 * 1
    j = a + l // 4 * 2
    k = a + l // 4 * 3
    if l >= 8:
        if l >= SHORTEST_NINTHER:
            i = _median_adjacent(less, box, i)
            j = _median_adjacent(less, box, j)
            k = _median_adjacent(less, box, k)
        j = _median(less, box, i, j, k)
    if box[0] == 0:
        return j, INCREASING_HINT
    if box[0] == MAX_SWAPS:
        return j, DECREASING_HINT
    return j, UNKNOWN_HINT


def _partial_insertion_sort(less, swap, a, b):
    MAX_STEPS = 5
    SHORTEST_SHIFTING = 50
    i = a + 1
    for _ in range(MAX_STEPS):
        while i < b and not less(i, i - 1):
            i += 1
        if i == b:
            return True
        if b - a < SHORTEST_SHIFTING:
            return False
        swap(i, i - 1)
        if i - a >= 2:
            for j in range(i - 1, 0, -1):
                if not less(j, j - 1):
                    break
                swap(j, j - 1)
        if b - i >= 2:
            for j in range(i + 1, b):
                if not less(j, j - 1):
                    break
                swap(j, j - 1)
    return False


def _partition(less, swap, a, b, pivot):
    swap(a, pivot)
    i, j = a + 1, b - 1
    while i <= j and less(i, a):
        i += 1
    while i <= j and not less(j, a):
        j -= 1
    if i > j:
        swap(j, a)
        return j, True
    swap(i, j)
    i += 1
    j -= 1
    while True:
        while i <= j and less(i, a):
            i += 1
        while i <= j and not less(j, a):
            j -= 1
        if i > j:
            break
        swap(i, j)
        i += 1
        j -= 1
    swap(j, a)
    return j, False


def _partition_equal(less, swap, a, b, pivot):
    swap(a, pivot)
    i, j = a + 1, b - 1
    while True:
        while i <= j and not less(a, i):
            i += 1
        while i <= j and less(a, j):
            j -= 1
        if i > j:
            break
        swap(i, j)
        i += 1
        j -= 1
    return i


def _pdqsort(less, swap, a, b, limit):
    MAX_INSERTION = 12
    was_balanced = True
    was_partitioned = True
    while True:
        length = b - a
        if length <= MAX_INSERTION:
            _insertion_sort(less, swap, a, b)
            return
        if limit == 0:
            _heap_sort(less, swap, a, b)
            return
        if not was_balanced:
            _break_patterns(swap, a, b)
            limit -= 1
        pivot, hint = _choose_pivot(less, a, b)
        if hint == DECREASING_HINT:
            _reverse_range(swap, a, b)
            pivot = (b - 1) - (pivot - a)
            hint = INCREASING_HINT
        if was_balanced and was_partitioned and hint == INCREASING_HINT:
            if _partial_insertion_sort(less, swap, a, b):
                return
        if a > 0 and not less(a - 1, pivot):
            a = _partition_equal(less, swap, a, b, pivot)
            continue
        mid, already_partitioned = _partition(less, swap, a, b, pivot)
        was_partitioned = already_partitioned
        left_len, right_len = mid - a, b - mid
        balance_threshold = length // 8
        if left_len < right_len:
            was_balanced = left_len >= balance_threshold
            _pdqsort(less, swap, a, mid, limit)
            a = mid + 1
        else:
            was_balanced = right_len >= balance_threshold
            _pdqsort(less, swap, mid + 1, b, limit)
            b = mid


def doc_id_of(name):
    """An ngram file name's document ID, for either index format."""
    if name.endswith(".bin"):
        return name[: -len(".bin")]
    return name.replace(".json", "", 1)


def get_files(ngrams_dir, metadata, sort_field="year"):
    """main.go:232-297. Returns [(doc_id, ngram file path)] in SortID order, so that a
    document's position in the list is its SortID.

    Caveat kept from Go: it reads ONE arbitrary metadata entry to decide whether the sort
    field is numeric (main.go:250-260), and Go's map iteration order is random; we read the
    first entry in the metadata file's order.
    """
    if not ngrams_dir:
        return []
    names = [f for f in os.listdir(ngrams_dir) if not os.path.isdir(os.path.join(ngrams_dir, f))]
    ids = [doc_id_of(n) for n in names]
    paths = dict(zip(ids, (os.path.join(ngrams_dir, n) for n in names)))
    numeric = False
    for fields in metadata.values():
        if sort_field not in fields:
            sort_field = ""
            break
        if go_atoi(fields[sort_field]) is not None:
            numeric = True
        break

    if sort_field == "":
        def raw_less(first, second):
            return (go_atoi(first) or 0) < (go_atoi(second) or 0)
    elif numeric:
        def raw_less(first, second):
            fa = go_atoi(metadata.get(first, {}).get(sort_field, ""))
            if fa is None:
                return first < second
            fb = go_atoi(metadata.get(second, {}).get(sort_field, ""))
            if fb is None:
                return first < second
            if fa < fb:
                return True
            if fa > fb:
                return False
            return (go_atoi(first) or 0) < (go_atoi(second) or 0)
    else:
        def raw_less(first, second):
            return metadata.get(first, {}).get(sort_field, "") < \
                metadata.get(second, {}).get(sort_field, "")

    go_sort_slice(ids, lambda i, j: raw_less(ids[i], ids[j]))
    return [(doc_id, paths[doc_id]) for doc_id in ids]
