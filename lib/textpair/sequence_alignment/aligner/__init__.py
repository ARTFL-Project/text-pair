"""Python sequence aligner.

`align()` is the entry point.
"""

from .numba_cache import configure as _configure_numba_cache

# Before .runner: numba binds cache locations when @njit is applied.
_configure_numba_cache()

from .runner import DEFAULTS, align  # noqa: E402

__all__ = ["align", "DEFAULTS"]
