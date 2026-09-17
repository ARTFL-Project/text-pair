"""Text cleaning and metadata handling for the output phase.

The output format is fixed by the corpora already published with it, so three
details below are load-bearing and easy to get wrong: the `$` anchor, the \\s
class, and U+FFFD.
"""
import codecs
import re
from html.entities import html5 as _HTML5

import orjson

tags = re.compile(r"<[^>]*?>")
brokenBeginTags = re.compile(r"^[^<]*?>")
brokenEndTags = re.compile(r"<[^>]*?\Z")     # \Z, not $: a trailing \n must not end it
spaces = re.compile(r" +")
spaceChars = re.compile(r"[\t\n\f\r ]+")   # this class excludes \v, unlike \s
tabEntities = re.compile(r"(&#9;)+")
cleanStart = re.compile(r"^[^\t\n\f\r ]+ ")
cleanEnd = re.compile(r" [^\t\n\f\r ]+\Z")

# The format wants one U+FFFD per invalid byte; Python's default replacement
# emits one per maximal invalid subpart instead.
codecs.register_error("goreplace", lambda e: ("�" * (e.end - e.start), e.end))

# Not Python's html.unescape: it resolves a truncated "&#8" and blanks the C0
# code points, where this format leaves the first alone and emits the second.
# The table is html.entities.html5 minus nGt;/nLt;, which the format never resolved;
# its without-semicolon fallback only consults the single-rune map.
_ENT = {k: v for k, v in _HTML5.items() if k not in ("nGt;", "nLt;")}
_ENT1 = {k: v for k, v in _ENT.items() if len(v) == 1}
_WIN1252 = ("\u20ac\u0081\u201a\u0192\u201e\u2026\u2020\u2021\u02c6\u2030\u0160"
            "\u2039\u0152\u008d\u017d\u008f\u0090\u2018\u2019\u201c\u201d\u2022"
            "\u2013\u2014\u02dc\u2122\u0161\u203a\u0153\u009d\u017e\u0178")   # escape.go:16
_REF = re.compile(r"&(?:#[xX][0-9a-fA-F]*;?|#[0-9]*;?|[0-9A-Za-z]*;?)")


def _ref(m):
    s = m.group(0)
    if len(s) <= 1:
        return s
    if s[1] == "#":
        body = s[2:]
        hexd = body[:1] in ("x", "X")
        if hexd:
            body = body[1:]
        semi = body.endswith(";")
        if semi:
            body = body[:-1]
        if 2 + hexd + len(body) + semi <= 3:       # escape.go:106, no characters matched
            return s
        x = int(body, 16 if hexd else 10) if body else 0
        x = (x & 0xFFFFFFFF) - (0x100000000 if x & 0x80000000 else 0)   # code points wrap as int32
        if 0x80 <= x <= 0x9F:
            return _WIN1252[x - 0x80]
        if x <= 0 or 0xD800 <= x <= 0xDFFF or x > 0x10FFFF:
            return "\ufffd"                      # x < 0 reaches utf8.EncodeRune as RuneError
        return chr(x)
    name = s[1:]
    if not name:
        return s
    v = _ENT.get(name)
    if v is not None:
        return v
    for j in range(min(len(name) - 1, 6), 1, -1):  # longestEntityWithoutSemicolon = 6
        v = _ENT1.get(name[:j])
        if v is not None:
            return v + name[j:]
    return s


def unescape(text):
    """html.UnescapeString (escape.go:38-162)."""
    return _REF.sub(_ref, text)


_unescape = unescape
_tags_sub = tags.sub
_bb_sub = brokenBeginTags.sub
_be_sub = brokenEndTags.sub
_sp_sub = spaces.sub
_tab_sub = tabEntities.sub


def get_text(buf, start, end, is_match):
    """`buf` is the whole document, as an mmap or bytes.

    A passage running past EOF is simply short: the equivalent read would pad with
    NULs and then trim them, which yields the same text as a slice.
    """
    if start < 0:
        start = 0
    passage = buf[start:end]
    if b"\x00" in passage:
        passage = passage.strip(b"\x00")
    if b"\xc2\xa0" in passage:
        passage = passage.replace(b"\xc2\xa0", b" ")
    text = passage.decode("utf-8", "goreplace")
    # Each substitution below is skipped when its pattern provably cannot match, which
    # is the common case for prose: `tags` and `brokenEndTags` both need a '<',
    # `brokenBeginTags` needs a '>', and `spaces` can only change the text where two
    # spaces meet, since replacing one space with one space is a no-op. Between them
    # these are most of the aligner's regex calls.
    if "<" in text:
        text = _tags_sub("", text)
    if not is_match:
        if ">" in text:
            text = _bb_sub("", text)
        if "<" in text:
            text = _be_sub("", text)
    if "&" in text:
        text = _unescape(text)
    if "\\" in text:
        text = text.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
    if "\t" in text:
        text = text.replace("\t", " ")
    if "&#9;" in text:
        text = _tab_sub(" ", text)
    if "\n" in text:
        text = text.replace("\n", " ")
    if "  " in text:
        text = _sp_sub(" ", text)
    return text


def alignment_to_text(buf, start_byte, end_byte, context_size):
    """cleanStart/cleanEnd apply to the contexts only, never the passage."""
    return (cleanStart.sub("", get_text(buf, start_byte - context_size, start_byte, False)),
            get_text(buf, start_byte, end_byte, True),
            cleanEnd.sub("", get_text(buf, end_byte, end_byte + context_size, False)))


def _atoi(s):
    """0 on any parse error, which is what the metadata's consumers expect."""
    try:
        return int(s)
    except (TypeError, ValueError):
        return 0


def rel_pos(start_byte, end_byte, doc_meta):
    """%.2f of byte/((end-start)/100), keeping the published spelling of the
    non-finite results a zero-length document produces."""
    coef = (_atoi(doc_meta.get("end_byte")) - _atoi(doc_meta.get("start_byte"))) / 100.0
    if coef == 0.0:
        return (_nonfinite(start_byte), _nonfinite(end_byte))
    return ("%.2f" % (start_byte / coef), "%.2f" % (end_byte / coef))


def _nonfinite(v):
    if v == 0:
        return "NaN"
    return "+Inf" if v > 0 else "-Inf"


def load_metadata(path):
    """Whitespace-collapse every string, blank out non-strings
    (the metadata is typed as strings throughout, so a non-string blanks out),
    then add the `ngrams` field."""
    # `ngrams` is deliberately `.json`, not `.bin`: it names the document's
    # ngrams_in_order/ file, which generate_ngrams still writes as JSON for the banality
    # filter, not the binary index in ngrams/. banality_finder.NgramDoc opens it by this
    # name and parses it with orjson.
    if not path:
        return {}
    with open(path, "rb") as metadata_file:
        meta = orjson.loads(metadata_file.read())
    sub = spaceChars.sub
    for doc, fields in meta.items():
        for key, value in list(fields.items()):
            fields[key] = sub(" ", value) if isinstance(value, str) else ""
        fields["ngrams"] = doc + ".json"
    return meta
