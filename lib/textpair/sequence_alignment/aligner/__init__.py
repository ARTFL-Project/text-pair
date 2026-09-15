"""Python sequence aligner, a port of lib/core/src/compareNgrams.

`align()` is the entry point and takes the same parameters as the Go binary's flags.
Selected with `aligner = python` under [MATCHING], or TEXTPAIR_ALIGNER=python.
"""

from .runner import DEFAULTS, align

__all__ = ["align", "DEFAULTS"]
