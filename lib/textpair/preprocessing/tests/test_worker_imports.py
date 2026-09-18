#!/usr/bin/env python3
"""Checks that the alignment path does not import the heavy optional stacks.

    test_worker_imports.py

One stray module-level import undoes this silently, and the cost is not a one-off:
`spawn` re-imports the worker's module in every child process, and macOS defaults
to `spawn`. When `textpair/__init__.py` imported the VSA orchestrator and the
passage classifier eagerly, an 8-worker run held 5GB rather than 0.7GB.

Each case runs in a subprocess, because what matters is what a fresh interpreter
ends up loading.
"""
import subprocess
import sys

FAILURES = []

# Stacks that must stay off the sequence-alignment path. spacy and cupy are
# loaded by preprocessing.spacy_stage, but only once a spacy_model is configured.
HEAVY = (
    "torch",
    "transformers",
    "sentence_transformers",
    "sklearn",
    "faiss",
    "spacy",
    "thinc",
    "cupy",
    "philologic",
    "psycopg2",
)

PROBE = """
import sys
{imports}
heavy = sorted({{name.split('.')[0] for name in sys.modules}} & set({heavy!r}))
print(",".join(heavy))
"""


def loaded_after(imports):
    """Top-level heavy modules present after running `imports` in a fresh process."""
    result = subprocess.run(
        [sys.executable, "-c", PROBE.format(imports=imports, heavy=list(HEAVY))],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        FAILURES.append(f"probe failed for {imports!r}: {result.stderr.strip().splitlines()[-1:]}")
        return None
    return [name for name in result.stdout.strip().split(",") if name]


def check(label, imports, allowed=()):
    present = loaded_after(imports)
    if present is None:
        return
    unexpected = [name for name in present if name not in allowed]
    if unexpected:
        FAILURES.append(f"{label} loaded {', '.join(unexpected)}")


def test_bare_package():
    check("import textpair", "import textpair")


def test_preprocessing():
    check("textpair.preprocessing",
          "from textpair.preprocessing import PreProcessor")


def test_ngram_generation():
    check("Ngrams", "from textpair.sequence_alignment import Ngrams")


def test_the_whole_alignment_surface():
    check("the alignment surface", """
import textpair
from textpair.preprocessing import PreProcessor
from textpair.sequence_alignment import (
    Ngrams, banality_auto_detect, merge_alignments, phrase_matcher, separate_banalities,
)
from textpair.sequence_alignment.aligner import align
from textpair.sequence_alignment import ngram_index
""")


def test_cli_help_stays_light():
    """`textpair --help` should not pay for the VSA stack either."""
    check("the CLI argument parser",
          "from textpair.parse_config import get_config, read_global_config")


def test_deferred_entry_points_still_resolve():
    """Deferring them must not break `from textpair import ...`."""
    result = subprocess.run(
        [sys.executable, "-c",
         "from textpair import parse_files, run_vsa, classify_passages, create_web_app;"
         " print('ok')"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or result.stdout.strip() != "ok":
        FAILURES.append(
            "the deferred entry points no longer import: "
            f"{(result.stderr or result.stdout).strip().splitlines()[-1:]}"
        )


def test_vsa_still_gets_its_stack():
    """The other direction: the VSA path is meant to load torch."""
    present = loaded_after("import textpair.vector_space_alignment")
    if present is not None and "torch" not in present:
        FAILURES.append("textpair.vector_space_alignment did not load torch")


def main():
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    if FAILURES:
        print(f"test_worker_imports: {len(FAILURES)} failure(s)")
        for failure in FAILURES:
            print(f"  {failure}")
        return 1
    print("test_worker_imports: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
