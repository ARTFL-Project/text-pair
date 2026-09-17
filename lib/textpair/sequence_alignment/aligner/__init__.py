"""Python sequence aligner.

`align()` is the entry point.

The numba cache directory is chosen here, before the kernel modules import, because
numba binds a function's cache location when `@njit(cache=True)` is applied rather than
when the function is first called. It is set through `numba.config.CACHE_DIR` and not
NUMBA_CACHE_DIR: `philologic.runtime.Query` sets that variable process-wide when textpair
imports it, so by the time this runs numba has already read the environment and the
kernels would cache into philologic's directory.
"""
import os
import tempfile

SHARED_CACHE_DIR = "/var/lib/text-pair/numba_cache"


def _usable(path):
    """True if path exists or can be created, and accepts a write."""
    try:
        os.makedirs(path, exist_ok=True)
        tempfile.TemporaryFile(dir=path).close()
    except OSError:
        return False
    return True


def _cache_candidates():
    override = os.environ.get("TEXTPAIR_NUMBA_CACHE_DIR")
    if override:
        yield override
    yield SHARED_CACHE_DIR
    xdg = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    yield os.path.join(xdg, "textpair", "numba")
    yield os.path.join(tempfile.gettempdir(), f"textpair-numba-{os.getuid()}")


def configure_numba_cache():
    """Point numba at the first usable cache directory, and return it.

    None means none were usable, which leaves numba's own fallback chain in place:
    the package's __pycache__, then ~/.cache/numba.
    """
    from numba import config

    for path in _cache_candidates():
        if _usable(path):
            config.CACHE_DIR = path
            return path
    return None


configure_numba_cache()

from .runner import DEFAULTS, align  # noqa: E402  must follow the cache configuration

__all__ = ["align", "DEFAULTS"]
