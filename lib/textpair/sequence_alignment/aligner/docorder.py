"""Document ordering: a document's position in the list get_files returns is its SortID,
which decides which document of a pair is the source and which is the target.

The order is a strict total order, produced by a sort key. The previous implementation
used a comparator that is non-transitive whenever a sort value fails to parse as an
integer, which left its result dependent on the sort algorithm, on hash iteration order
and on the order the filesystem listed the ngram directory in. Ordering by key therefore
changes results on any corpus that has an unparseable sort value, deliberately: which
document of a pair is the source decides which passages are found at all.
"""
import os
import re

_INT_RE = re.compile(r"^[+-]?\d+$")

NUMERIC, STRING, DOC_ID = "numeric", "string", "doc_id"


def parse_int(value):
    """strconv.Atoi: no surrounding space, optional sign, decimal digits only."""
    return int(value) if isinstance(value, str) and _INT_RE.match(value) else None


def doc_id_of(name):
    """An ngram file name's document ID, for either index format."""
    if name.endswith(".bin"):
        return name[: -len(".bin")]
    return name.replace(".json", "", 1)


def doc_id_key(doc_id):
    """Numeric IDs by value, then the rest by string. Total, since IDs are unique."""
    number = parse_int(doc_id)
    return (1, 0, doc_id) if number is None else (0, number, "")


def sort_mode(metadata, sort_field):
    """NUMERIC when most of the field's values parse as integers, STRING when they do not,
    DOC_ID when no document carries the field (the established behaviour for a missing
    field). Decided once over the whole corpus, so it cannot vary between runs."""
    if not sort_field:
        return DOC_ID
    values = [fields[sort_field] for fields in metadata.values() if sort_field in fields]
    if not values:
        return DOC_ID
    parsed = sum(parse_int(value) is not None for value in values)
    return NUMERIC if 2 * parsed > len(values) else STRING


def sort_key(metadata, sort_field, mode=None):
    """The key function over document IDs, for `mode` or the mode the metadata implies."""
    if mode is None:
        mode = sort_mode(metadata, sort_field)
    if mode == DOC_ID:
        return doc_id_key
    if mode == STRING:
        def key(doc_id):
            return (metadata.get(doc_id, {}).get(sort_field, ""), doc_id_key(doc_id))
        return key

    def key(doc_id):
        value = parse_int(metadata.get(doc_id, {}).get(sort_field, ""))
        if value is None:
            return (1, 0, doc_id_key(doc_id))   # unparseable and missing values go last
        return (0, value, doc_id_key(doc_id))
    return key


def get_files(ngrams_dir, metadata, sort_field="year"):
    """[(doc_id, ngram file path)] in SortID order."""
    if not ngrams_dir:
        return []
    # Sorted so that nothing depends on the filesystem's listing order even incidentally.
    names = sorted(name for name in os.listdir(ngrams_dir)
                   if not os.path.isdir(os.path.join(ngrams_dir, name)))
    paths = {doc_id_of(name): os.path.join(ngrams_dir, name) for name in names}
    key = sort_key(metadata, sort_field)
    return [(doc_id, paths[doc_id]) for doc_id in sorted(paths, key=key)]
