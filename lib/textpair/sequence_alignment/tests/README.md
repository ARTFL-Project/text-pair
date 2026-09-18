# Aligner checks

Two kinds of script live here, and the prefix says which.

**`test_*.py` assert properties and need nothing but the package.** They build their own
corpora or use the fixtures, and they either pass or fail. `run_tests.py` runs all of
them.

**`check_*.py` are tools you point at data**, or checks that guard something other than
the alignments. `check_reference_output.py --fixtures` is the exception: in that mode it
asserts, so `run_tests.py` includes it. The other three are described below.

## Running them

```bash
cd /disk1/shared/text-pair
T=lib/textpair/sequence_alignment/tests

python $T/run_tests.py                       # fixtures, under a minute
python $T/run_tests.py --only match_order    # one check
python $T/run_tests.py --fresh-cache         # compile into an empty numba cache
```

The fixture pass is what to run while working. It is fast because the corpora are three
documents each, and that is also its limitation: several of the properties here only
fail on corpora dense enough to produce the awkward cases. **Before pushing, run the
corpus pass too:**

```bash
V=/disk1/shared/text-pair-validation
python $T/run_tests.py --threads 32 \
  --source-files $V/corpora/cc_clean/ngrams \
  --source-metadata $V/corpora/cc_clean/metadata/metadata.json
```

Every check that accepts a corpus then runs against it as well. It is the pass that has
actually caught things: `MATCHER_SYMMETRY.md` records a session where the fixtures were
green and `cc_clean` was not.

The corpora are under `/disk1/shared/text-pair-validation/corpora`, with a README of their
own: `cc_clean` (62 documents, very dense pairs), `sub3596` (frantext, 3,596 documents),
`ecco_clean` (3,014 documents, 3 million alignments, slow).

Measured on classical_chinese, which is the corpus to reach for: dense enough to be
awkward, small enough to be quick.

| check | fixtures | corpus |
|---|---|---|
| `document_order` | 3.4s | 3.4s |
| `match_order` | 3.7s | 11.1s |
| `chunk_order` | 3.4s | 6.1s |
| `aligner_paths` | 13.7s | 17.3s |
| `matcher_symmetry` | 8.9s | 11.2s |
| `reference_output` | 19.9s | 19.9s |
| `ngram_binary` | skipped | 10.1s |
| | **53s** | **79s** |

Most of each figure is numba compiling the kernels in a fresh subprocess, which is why
almost nothing gets faster on the fixtures. No single check should take minutes; if one
does, something is wrong with it rather than with your machine.

## What each check asserts

| check | |
|---|---|
| `test_document_order.py` | the sort key is a strict total order, and the numeric/string/document-ID mode is decided over the whole corpus rather than per document |
| `test_match_order.py` | the two index properties `align_source` relies on to order a pair's matches without sorting them: a source index is unique within its document, and a key's positions ascend by index. Then that the order it produces equals a full sort of the cross-product |
| `test_chunk_order.py` | chunk file names, and that concatenating them in `sort -V` order keeps each source document's records together — which `alignment_merger.first_step_merge` depends on |
| `test_aligner_paths.py` | batching by `source_batch`/`target_batch` gives the same records as an unbatched run, including counts past the corpus size |
| `test_matcher_symmetry.py` | comparing a pair either way round gives mirrored passages: same record counts, same document pairs, same duplicates, and no passage without a counterpart. Includes a corpus where a phrase occurs once in one document and seven times in another, the case the current matcher exists for |
| `check_reference_output.py --fixtures` | every artifact of a fresh run matches the stored reference tree byte for byte: chunk names, records, `count.txt`, `duplicate_files.csv`, `alignment_config.ini` |
| `check_ngram_binary.py` | the binary ngram index loads to arrays identical to the JSON one. Needs a corpus; `run_tests.py` skips it otherwise |

## The two that are not in `run_tests.py`

**`check_direction_flips.py BEFORE AFTER`** characterises the difference between two
output trees you already have, so there is nothing for it to run against automatically.
Use it when a change to document ordering moves records and you want to know whether every
difference is explained by a pair swapping direction.

**`check_tracing.py`** checks the `--debug` trace rather than the alignments: that
`tracing._walk` still produces exactly what `matching.match_passage` does, that tracing
changes neither the records nor the count, and that the trace accounts for every alignment.

Run it when you change the matcher. `_walk` is a second implementation of the matcher,
written so that tracing costs nothing on runs that do not use it, and this is the only
thing that tells you the two have not drifted — it is what caught a row-ordering mistake
when the matcher was last replaced.

It is not a standing check for two reasons. It guards a developer diagnostic rather than
the output, and it was by a wide margin the slowest thing here: a `--debug` run
re-derives every pair, which `--max-matches` cannot bound, because a trace has to describe
the whole pair. On classical_chinese the drift guard itself is 6.5s with the default cap,
and the `--debug` run around it is minutes. Use `--debug_pairs` to narrow it.

```bash
python $T/check_tracing.py --source-files $V/corpora/cc_clean/ngrams \
  --source-metadata $V/corpora/cc_clean/metadata/metadata.json --threads 32
```

`--max-matches N` (default 1,000) bounds the matches taken from any one pair. Both
implementations get the same list, so the comparison is exact either way, and every pair
is still walked; `0` takes all of them, which on classical_chinese is 20.5 million matches
and about 25 minutes for the two `flex_gap` settings.

## Fixtures and their references

`fixtures/` holds four miniature corpora covering metadata and text variants the real
corpora do not, each of which once produced a wrong record: metadata without
`start_byte`/`end_byte`, non-string metadata values, a missing text file, and an ngram
file with no metadata entry. `fixtures/make_fixtures.py` regenerates the corpora
themselves; they rarely change.

`fixtures/<name>/reference/` is stored **output**, not a second implementation. Any
deliberate change to what the matcher emits makes all four stale, and every fixture fails
until they are recorded again:

```bash
python $T/fixtures/record_references.py
```

Run that only when the change is intended, and say in the commit message why the records
moved. Re-recording to make an unexplained difference go away defeats the point of having
references at all.

References are recorded at a fixed four threads and checked at four, whatever `--threads`
says, because chunk file names carry the per-thread target split that `runner.chunk_ranges`
computes. `FIXTURE_THREADS` in `check_reference_output.py` is the single definition;
`record_references.py` imports it so the two cannot drift.

## The numba cache

The kernels are `@njit(cache=True)`, and a cache directory holding kernels compiled from
different source has twice produced convincing results that were false — a discrepancy
that did not exist, and a residual asymmetry that had already been fixed. Both cost real
time to chase.

`run_tests.py --fresh-cache` compiles into an empty directory and discards it, which rules
that out at the price of a minute or so of compilation. Reach for it whenever a result
surprises you, and when measuring anything you intend to write down.

Otherwise the cache location comes from `TEXTPAIR_NUMBA_CACHE_DIR`, then
`/var/lib/text-pair/numba_cache`, then `$XDG_CACHE_HOME/textpair/numba` — see
`aligner/__init__.py`, which also explains why it is set through `numba.config.CACHE_DIR`
rather than `NUMBA_CACHE_DIR`.

## check_hash_collisions.py

Measures how often a 32-bit hash collision fabricates a passage, by indexing a corpus
twice under different mmh3 seeds and diffing the alignments: collisions under one seed are
almost surely not collisions under the other. It checks the aligner is deterministic
first, since otherwise the diff means nothing.

```bash
python $T/check_hash_collisions.py --philo-db DIR --work DIR [--docs N] [--workers N]
```

Needs room for two n-gram indexes and two alignments of the corpus. On frantext the answer
is about two spurious passages per million, flat across a 19x range of corpus size; see
NGRAM_KEY_COLLISIONS.md for what that means and what it does not cover.
