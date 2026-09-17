"""Cache directory for the aligner's numba kernels.

Sets numba.config.CACHE_DIR rather than NUMBA_CACHE_DIR: philologic.runtime.Query sets
that variable process-wide when textpair imports it, so by the time this runs numba has
already read the environment and the kernels would cache into philologic's directory.
Must run before the kernel modules import -- numba binds a function's cache location
when @njit(cache=True) is applied, not when the function is first called.
"""
import os
import tempfile

SHARED_DIR = "/var/lib/text-pair/numba_cache"


def _usable(path):
    """True if path exists or can be created, and accepts a write."""
    try:
        os.makedirs(path, exist_ok=True)
        tempfile.TemporaryFile(dir=path).close()
    except OSError:
        return False
    return True


def _candidates():
    override = os.environ.get("TEXTPAIR_NUMBA_CACHE_DIR")
    if override:
        yield override
    yield SHARED_DIR
    xdg = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    yield os.path.join(xdg, "textpair", "numba")
    yield os.path.join(tempfile.gettempdir(), f"textpair-numba-{os.getuid()}")


def configure():
    """Point numba at the first usable cache directory, and return it.

    None means none were usable, which leaves numba's own fallback chain in place:
    the package's __pycache__, then ~/.cache/numba.
    """
    from numba import config

    for path in _candidates():
        if _usable(path):
            config.CACHE_DIR = path
            return path
    return None
