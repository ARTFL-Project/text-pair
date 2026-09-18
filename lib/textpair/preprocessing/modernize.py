"""Historical spelling modernization.

The maps live in data/<language>.tsv -- plain tab-separated text, sorted, so they
can be edited and reviewed like any other source file. Parsing one costs ~16ms,
so a msgpack cache is built beside it and reused until the .tsv changes.
"""

from __future__ import annotations

import os
import tempfile

import msgspec

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SUPPORTED = ("french", "english")

_DECODER = msgspec.msgpack.Decoder(dict[str, str])
# Keyed by language so repeated construction, and forked workers, share one copy.
_CACHE: dict[str, dict[str, str]] = {}


def data_path(language: str) -> str:
    return os.path.join(DATA_DIR, f"{language}.tsv")


def parse_tsv(path: str) -> dict[str, str]:
    """Read a modernization map. Blank lines and #-comments are ignored."""
    mapping: dict[str, str] = {}
    with open(path, encoding="utf-8") as tsv_file:
        for line in tsv_file:
            if not line.strip() or line.startswith("#"):
                continue
            form, _, modern = line.rstrip("\n").partition("\t")
            if form and modern:
                mapping[form] = modern
    return mapping


def _cache_candidates(language: str) -> list[str]:
    """Where the packed cache may live, best first.

    The package directory is usually root-owned, so fall back to a shared temp
    location rather than failing or silently reparsing on every run.
    """
    return [
        os.path.join(DATA_DIR, f"{language}.mpk"),
        os.path.join(tempfile.gettempdir(), f"textpair-modernize-{language}.mpk"),
    ]


def _read_cache(path: str, source_mtime: float) -> dict[str, str] | None:
    try:
        if os.path.getmtime(path) < source_mtime:
            return None
        with open(path, "rb") as cache_file:
            return _DECODER.decode(cache_file.read())
    except (OSError, msgspec.DecodeError):
        return None


def _write_cache(path: str, mapping: dict[str, str]) -> bool:
    """Write atomically so concurrent workers never read a half-written cache."""
    try:
        directory = os.path.dirname(path)
        with tempfile.NamedTemporaryFile(dir=directory, delete=False, suffix=".tmp") as tmp:
            tmp.write(msgspec.msgpack.encode(mapping))
            temp_path = tmp.name
        os.replace(temp_path, path)
        return True
    except OSError:
        return False


def load_modernizer(language: str) -> dict[str, str] | None:
    """Return the modernization map for a language, or None if unsupported.

    A plain dict rather than a wrapper object: callers bind `.get` once and the
    lookup then costs nothing extra per token.
    """
    language = language.lower()
    if language not in SUPPORTED:
        return None
    if language in _CACHE:
        return _CACHE[language]

    source = data_path(language)
    source_mtime = os.path.getmtime(source)
    candidates = _cache_candidates(language)
    for candidate in candidates:
        cached = _read_cache(candidate, source_mtime)
        if cached is not None:
            _CACHE[language] = cached
            return cached

    mapping = parse_tsv(source)
    for candidate in candidates:
        if _write_cache(candidate, mapping):
            break
    _CACHE[language] = mapping
    return mapping


def pack(language: str) -> str | None:
    """Build the cache ahead of time. Called by install.sh."""
    if language.lower() not in SUPPORTED:
        return None
    mapping = parse_tsv(data_path(language))
    for candidate in _cache_candidates(language):
        if _write_cache(candidate, mapping):
            return candidate
    return None


def pack_all() -> None:
    """Build every cache ahead of time. Called by install.sh."""
    for language in SUPPORTED:
        written = pack(language)
        print(f"  {language}: {'packed to ' + written if written else 'no writable cache location'}")
