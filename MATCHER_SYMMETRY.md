# Direction dependence in the sequence matcher

**Scope:** `lib/textpair/sequence_alignment/aligner/matching.py`, `inverted_index.py`,
`tracing.py`
**Corpus:** frantext `sub3596` (3,596 documents), eccotcp `ecco_clean` (3,014) and
classical_chinese `cc_clean` (62), from `/disk1/shared/text-pair-validation/corpora`
**Measured:** 2026-09-17, 64-core machine, 32 threads
**Status:** landed. `match_passage` and `merge_passages` are the symmetric kernels; the
asymmetric ones they replaced are gone. **Superseded in part on 2026-09-23**: measured
against the Go aligner, the chains lost 3.9% of its runs, and `align_pair` now adds the
old walk, run from both documents. See "Losses against the Go aligner" below.

---

## The question

Comparing document A against B does not give the mirror image of comparing B against A.
This note measures how large that difference is, identifies what causes it, and evaluates
a replacement matcher that removes it. The constraint throughout is that the replacement
must not lose passages the current matcher finds.

## How it was measured

A single-corpus run compares each pair once, with the document earlier in `sort_by` order
as the source. Reversing the document order therefore compares every pair the other way
round. `extras/` is not involved; the harness is:

```bash
export NUMBA_CACHE_DIR=/var/tmp/textpair_numba OPENBLAS_NUM_THREADS=1
V=/disk1/shared/text-pair-validation
# metadata with year replaced by (N - rank), which reverses the order with no ties
python -m textpair.sequence_alignment.aligner \
  --source_files=$V/corpora/sub3596/ngrams \
  --source_metadata=<reversed metadata> --output_path=<dir> --threads=32
```

Records from the two runs are compared as *undirected* passage pairs: each record becomes
`(docA, docB, startA, endA, startB, endB)` with `docA` the smaller document ID, so a
passage found with the roles exchanged gets the same key.

## How large the difference is

| | records | |
|---|---|---|
| forward | 46,474 | |
| reversed | 47,598 | |
| same passage both ways | 32,090 | **51.8% agreement** |
| only forward | 14,384 | of which 2,405 overlap a reversed passage |
| only reversed | 15,508 | of which 2,422 overlap a forward passage |

Half the output does not survive a role swap, and the differences are mostly not the same
passage with a wobbly extent: about 11,500 records each way are found in one direction and
missed entirely in the other. 338 document pairs produce alignments one way and none the
other.

## Three independent causes, in order of size

### 1. The matcher reserves the source range of each passage it emits

`matching.py:108`, `last_source_position = last_source_index + 1`. Target positions are
never reserved, so passages are disjoint in the source and may overlap freely in the
target. A phrase occurring once in the source and *n* times in the target therefore yields
one passage; the same pair compared the other way yields *n*.

The clearest instance in the corpus is document `475` (Fontenelle, *De l'origine des
fables*, 5,218 words) against `2368` (Verne, *Vingt mille lieues sous les mers*): **1
passage one way, 105 the other**. The pair shares four ngram indices in `475` against 420
positions in `2368` — one phrase, 105 occurrences. The phrase is edition boilerplate,
`page contient à cet endroit dans l'original papier une illustration`. The documents
dominating the whole diff are long 19th-century novels carrying repeated OCR furniture.

This cause alone accounts for the difference between 52.4% and 99.8% agreement (measured
with merging switched off, to separate it from cause 2).

### 2. The merger derives both byte thresholds from the source passage

`matching.py:166-168`. `merge_with_previous` computes one `distance_value` from the
*source* passage's byte length and applies it to both documents, and it checks that the
candidate does not overlap the previous passage in the target only. Worth 6.7 points of
agreement: with cause 1 fixed but this left alone, agreement is 93.1% rather than 99.9%.

### 3. The duplicate filter divides by the source's ngram count

`inverted_index.py:295`, `pct = count / ns * 100` where `ns` is the source's distinct-key count. A
short document largely contained in a long one is a duplicate one way only. On frantext:
29 duplicate pairs forward, 24 reversed, 18 in common. Small, but it removes pairs from
the output entirely, so it is the last thing standing between 99.9% and 100%.

---

## The replacement matcher

A passage becomes **a chain of matches strictly increasing in both coordinates, with
consecutive steps no longer than `max_gap` in either, dense enough that every
`window_size` stretch of either document holds `min_in_window` of them**. Every test is a
symmetric function of the two coordinates, so exchanging them gives the mirrored set of
chains. Chains are found longest-first by dynamic programming over the matches rather than
by taking whichever continuation comes first, so a passage is never cut short by a nearer
but unextendable match.

The design question that decided everything was **what a chain has to earn to be
reported**:

| reservation rule | frantext records | |
|---|---|---|
| source range (current) | 46,474 | *n* one way, 1 the other |
| nothing | 243,772 | ***n×m*** for a phrase occurring *n* and *m* times |
| **both sides** | **64,208** | **≈ *n+m*** |

Reserving neither side is the obvious symmetric fix and it explodes combinatorially:
5.2x the records, and 172,848 of the 199,896 additions were repetitions of the one
boilerplate line above. Reserving *both* sides is the symmetric rule — a chain is kept
when it covers a stretch of either document that no kept chain covers yet. That is a
minimal cover of both documents rather than a cross-product, and it lands almost exactly
where the union of the two directions does (61,982).

### Results with all three causes fixed

| | |
|---|---|
| agreement between directions | **exact** — 64,208 records each way, none uncovered |
| document pairs, duplicate sets | identical |
| records | 64,208 against 46,474, **+38%** |
| baseline passages not covered | 42 source spans, 20 target spans of 46,474 (**0.13%**) |
| matching-phase CPU, frantext | **0.89x** — 4.94 s to 4.38 s single-threaded, once optimised |
| passages over 10,000 words | 11, against the baseline's 19; longest 23,874 against 48,336 |

The extreme tail improves: requiring non-overlap in both documents and deriving each
side's byte allowance from its own length stops some of the merger's snowballing, where
each merge enlarges `prev` and so enlarges the distance the next merge may span.

### Where the extra 38% comes from

| | |
|---|---|
| records added over the forward baseline | 21,535 |
| already found by the reversed baseline | 12,849 (**60%**) |
| found by neither direction | 8,686 (40%) |

So three fifths of the gain is the current matcher's own output, reported or not depending
on which document happens to be older. The remaining two fifths come from the DP finding
longer chains, from the density walk cutting a thin stretch and keeping both sides rather
than dropping everything past it, and from the cover rule reporting occurrences neither
direction picked.

Coverage holds against both directions, not just the forward one:

| baseline run | records | source spans uncovered | target spans uncovered |
|---|---|---|---|
| forward | 46,474 | 42 | 20 |
| reversed | 47,598 | 18 | 21 |

6,460 records exist in one baseline direction or the other but not in the replacement.
Per-pairing that looks like a loss; per-document it is 62 spans of 46,474 and 39 of
47,598, because the material is covered by a differently paired passage.

Of the 21,535 additions, 11,663 (54%) fall in the 10-to-14-word bucket, and the ones at the
median are the boilerplate line. That is what `banality_finder.py` exists for, and it is
noise the current matcher already emits half the time.

The growth is concentrated in the pairs that were already heavy, not spread over the
corpus — which is what a fix for a repeated-material asymmetry should look like:

| | current | replacement |
|---|---|---|
| document pairs with alignments | 15,530 | 15,998 |
| passages per pair, median | 1 | 1 |
| passages per pair, p99 | 37 | 62 |
| passages per pair, max | 242 | 391 |
| pairs with over 50 passages | 90 | 236 |

### The two residuals, both characterised

**Mirror ties.** Chaining prefers the nearest predecessor, by (Δi + Δj, min(Δi, Δj)). Both
terms are symmetric in the two documents, and together they fix the pair {Δi, Δj}
completely, so what survives is a predecessor at (Δi, Δj) against one at (Δj, Δi) — "extend
back in A, or in B", exact mirror images that nothing symmetric can separate. Adding the
`min` term cut the ties in the densest classical_chinese pair from 19 to **7**, and made
that count independent of `flex_gap` (it was 37 with flex_gap on), which is what a
geometric property of the match set should look like.

A mirror tie moves a passage's start by a few ngrams, and in a pair dense enough that
chains compete for the same matches it can cascade, because the passage extracted first
consumes matches the other would have used. With the `min` term in place, none of that
survives into the output on either corpus measured:

| corpus | flex_gap | records each way | with no overlapping counterpart |
|---|---|---|---|
| frantext, 3,596 docs | off | 64,208 | **0** |
| frantext | on | 63,015 | **0** |
| classical_chinese, 62 docs | off | 46,679 | **0** |
| classical_chinese | on | 45,109 | **0** |

Record counts, document pairs and duplicate sets are identical in every row too. The
tolerance in `test_matcher_symmetry.py` is slack against the 7 ties that remain reachable
in principle, not against a residue anything currently produces.

Before the `min` term, classical_chinese with flex_gap on left 1 record in 45,109 without
a counterpart, on the pair with 1,063,974 matches between two documents. Should a corpus
ever reintroduce one, the fix is to emit both alternatives at a mirror tie, which is
symmetric and adds records.

**The 1,173 old-matcher records with no counterpart** are the same repeated material paired
differently: both matchers cover both documents, they differ on which occurrence pairs with
which. Testing coverage per document rather than per pairing gives the 42-and-20 figure
above.

---

## Cost

The straightforward implementation (matcher 7) costs **2.6x the matching phase on
eccotcp**, all of it the chaining matcher — the merger and duplicate fixes are free
(matcher 4, chaining with the old merger and old duplicate rule, already costs 2.64x).
That version allocates seven buffers per pair and comparison-sorts the whole match list,
against a current matcher that allocates nothing and sorts nothing.

Two changes remove almost all of it, neither touching the algorithm:

- **Hoist the buffers.** `align_source` already keeps its match arrays across a source's
  targets, for the reason given in its own comments; the chaining buffers join them, sized
  once to the biggest pair a source has seen.
- **Counting-sort the extraction order** over only the matches that can end a chain, with
  each length's block ordered by its coordinate pair, instead of `argsort` over every
  match. Plus an early return when no chain reaches `min_matching`, which is most pairs.

The result is byte-identical to the unoptimised version, and **the cost depends on
`flex_gap`**, which the shipped config turns on. Matching phase only, 32 threads, the
retired and shipped kernels run alternately over the same preloaded corpora:

| corpus | flex_gap | retired | shipped | records |
|---|---|---|---|---|
| frantext, 1.5M pairs | off | 0.217 s / 6.05 s CPU | **0.203 s / 5.54 s — 0.94x, 0.92x** | 46,474 → 64,208 |
| frantext | **on** | 0.220 s / 6.14 s | **0.205 s / 5.62 s — 0.93x, 0.92x** | 45,803 → 63,015 |
| eccotcp, 4.3M pairs | off | 13.07 s / 418 s | 17.54 s / 496 s — 1.34x, 1.19x | 3,053,095 → 6,365,291 |
| eccotcp | **on** | 13.39 s / 428 s | 25.41 s / 627 s — **1.90x, 1.47x** | 3,049,800 → 6,340,308 |

On frantext it is *faster* than the matcher it replaces, on both settings, because the
early return bails out of most pairs after one DP pass and that is cheaper than the old
greedy scan with its window bookkeeping.

On eccotcp it is not. flex_gap raises `link_bound` from 15 to 30, which doubles the
candidate window the chaining DP scans for every match, and eccotcp is dense enough for
that to dominate: 1.34x becomes 1.90x. The matching phase is around 13 s of a 21 s full
run, so end to end that is roughly +40% on eccotcp and free on frantext.

**The obvious way to get it back** is that the flexed allowance only exceeds `max_gap`
once a run has reached `min_matching` matches — and `best[a]`, already computed, *is* how
many matches the chain ending at `a` has. So the allowance for a candidate link could be
derived from `best[a]` instead of the global ceiling, which is both tighter and a more
faithful reproduction of the old schedule than linking loosely and cutting afterwards.
Not done here: it is a correctness-sensitive change to the hot loop and wanted more care
than the end of this session had.

### An exact prune, measured but not needed

Matches that belong to one passage are all transitively gap-linked, so **a gap-linked
component holding fewer than `minimum_matching_ngrams` matches cannot contain a passage**.
That prune is exact, and it was measured as the basis for a two-stage matcher before the
optimisation above made two stages unnecessary. It is kept here because it bounds how much
candidate work the problem really contains. Measured over every compared pair on frantext:

| | |
|---|---|
| compared pairs | 1,515,752 |
| matches | 43,458,251 |
| gap-linked components | 40,426,108 |
| components with ≥ `min_matching` matches | **274,920** |
| of those, simple monotone chains | 227,445 (82.7%) |
| of those, needing a real chaining rule | 47,475 (17.3%) |

A 147x reduction in candidates, exactly. Four fifths of the survivors are forced chains
where the DP and a single forward walk agree — which is why the DP's early return is
enough on its own, and why chaining every pair turned out not to be expensive after all.

---

## Other families considered

The current matcher and the replacement are both **exact-ngram seeds plus monotone 2-D
chaining**. Holding to "no losses against the baseline" excludes most alternatives, because
the baseline's chaining rule is permissive: `max_gap` of 15 in both coordinates lets the
diagonal drift 14 per step.

| family | symmetric | superset | note |
|---|---|---|---|
| Diagonal / offset voting (BLAST two-hit, MUMmer) | yes, `d → -d` | **no** | O(n) per pair, no DP. Any fixed band loses interpolated quotation. Belongs as stage 1, not as a replacement |
| Banded Smith-Waterman on token streams (`passim`) | yes | **no**, but dominates in practice | Gap penalties, score-chosen boundaries, aligns through stretches with no shared ngram. Changes the output contract: extents stop landing on ngram boundaries, so `context_size`, the merger and the banality filter all shift |
| Maximal exact matches over a suffix array / FM-index | by definition | **yes** — every k-gram hit lies inside a MEM | Removes hash collisions and the arbitrariness of `minimum_matching_ngrams`; chaining gets *cheaper* because anchors are few and long. Replaces `generate_ngrams.py` and the inverted index, and inverts the execution model to one global computation. Separate project |
| MinHash / LSH / winnowing | yes | no | Scales far better, but window-quantised boundaries destroy the precise extents that are this tool's output. The VSA path already occupies this niche |

---

## flex_gap

`config/sa_config.ini` ships **`flex_gap = true`**, so it is the setting real runs use,
even though the aligner module's own `DEFAULTS` have it off. Every measurement above was
taken with it off; the numbers below are with it on, and the conclusions do not move.

flex_gap lets a run that has already reached `minimum_matching_ngrams` tolerate larger
gaps: the allowance jumps by `min_matching`, then climbs by one per match while it is
below `window_size`. It was never a source of asymmetry — the old kernel applied it to
both documents alike — but it interacts with chaining, because the allowance depends on
how far into a run a match sits and the chaining DP does not know that yet.

Linking with the base `max_gap` would lose runs the old kernel could build. Instead
`link_bound` returns the ceiling the allowance can reach, `max(max_gap + min_matching,
window_size)`, the DP links to that, and the walk down each chain enforces the real
schedule, cutting where a step exceeds the allowance in force at that point. Linking
loosely and cutting afterwards is what keeps the old kernel's runs reachable.

| frantext, flex_gap = true | |
|---|---|
| old matcher, agreement between directions | 52.2% (45,803 forward, 47,119 reversed) |
| new matcher, agreement between directions | **exact** (63,015 each way, none uncovered) |
| coverage against the old forward run | 50 source spans, 33 target spans of 45,803 uncovered |
| coverage against the old reversed run | 32 source, 44 target of 47,119 |

The uncovered count is a little worse than with flex_gap off (42 and 20), which is the
price of loose linking: the DP maximises chain length under the loose bound, and a chain
that is longest there can fragment where a tighter one would not have. 0.18% against
0.13%.

`check_tracing.py` and `test_matcher_symmetry.py` both run each setting, since
flex_gap changes how far the matcher links and what the walk down each chain enforces.
The trace test previously only ever ran the default, which is why this path went
unchecked until now.

## What landed

| file | |
|---|---|
| `matching.py` | `match_passage` and `merge_passages` are the symmetric kernels. The asymmetric `match_passage` and `merge_with_previous` are deleted, not switchable: nobody would choose a matcher whose answer depends on which document is older, and keeping it would mean maintaining two `tracing._walk` mirrors. Old results reproduce from a tag, the way `2042c06` retired the Go aligner |
| `inverted_index.py` | chaining buffers hoisted into `align_source`; duplicate share taken against the smaller document |
| module layout | `kernels.py` became `matching.py`; `pipeline.py` merged into `inverted_index.py`, whose kernels it existed to drive, following `ngram_loader.py`'s pattern; `gotext.py` split, its passage-text half into `output.py` and `load_metadata` into `documents.py` (was `docorder.py`); `numba_cache.py` folded into `__init__.py`, whose import order is the reason it existed. 12 modules to 9, all renamed away from abbreviations. Output is byte-identical on frantext and classical_chinese, both `flex_gap` settings |
| `tracing.py` | `_walk` rewritten to mirror the new matcher, with reasons for where a passage stops and why a rejected one was rejected |
| `tests/test_matcher_symmetry.py` | new: runs a corpus forward and with the document order reversed and compares undirected passage pairs, including a corpus where a phrase occurs once in one document and seven times in another. Fails against the retired matcher with 1 occurrence against 7 |
| `tests/check_reference_output.py` | was a Go-parity harness; now compares against a stored reference tree. `fixtures/record_references.py` re-records them |
| `tests/check_tracing.py` | was `test_debug_trace.py`: rewritten to mirror the new matcher, and now runs both `flex_gap` settings, having only ever run the default. Renamed out of the automated suite because it checks the `--debug` trace rather than the alignments |
| `tests/check_direction_flips.py` | docstring: a direction flip now mirrors |

All of `test_match_order`, `test_chunk_order`, `test_document_order`, `test_aligner_paths`,
`test_debug_trace`, `test_matcher_symmetry` and `check_reference_output --fixtures` pass, and
`_walk` matches the kernel on all 1,891 classical_chinese pairs, for both `flex_gap`
settings, with the trace accounting for every alignment. The fixture references re-record
byte for byte. The aligner is deterministic across runs and thread counts: five runs of
classical_chinese, three at 32 threads and two at 1, all gave 45,109.

## Losses against the Go aligner

Measured 2026-09-23 on full frantext (3,630 documents) with
`/shared/alignments/frantext/sa_config.ini` (`flex_gap = true`), against `master`
end to end: `text_preprocessing` ngrams, `compareNgrams`. Comparisons are of the raw
aligner output, before the phrase filter and banality detection.

| | records |
|---|---|
| Go | 46,452 |
| Python, chains only | 63,689 |
| Python, chains + anchored scan | 65,866 |

Running the Python aligner on `master`'s own JSON ngrams gives 63,692 records and the
same losses, so preprocessing and the 64-bit keys account for almost none of it. A port
of Go's `matchPassage` and `mergeWithPrevious`, statement for statement, reproduces
`compareNgrams` on all 15,809 frantext pairs (`flex_gap` on) and all 1,329 `cc_clean`
pairs (off); everything below uses it to re-derive Go's runs, before merging.

**The chains missed 1,904 of Go's 48,202 runs (3.9%)**, and 1,684 of the 48,995 it
finds with the documents' roles exchanged: runs with less than 90% of their bytes
covered by a record overlapping them in both documents. By mechanism, over the 778
pairs holding one:

| | runs | |
|---|---|---|
| cover rule | 1,679 | a valid chain, rejected because both stretches overlap kept passages |
| gap exception | 125 | Go accepts a step past `max_gap` while the window is dense |
| DP fragmentation | 55 | the matches went to a longer chain, cut into pieces too short to keep |
| window test | 4 | tail differences |

Plus 15 records in 6 pairs lost to the duplicate rule (40 duplicate pairs against Go's
34, a strict superset).

- **The cover rule** is the per-pairing residue this note set aside as "the same
  repeated material paired differently". Go pairs each stretch of the source with the
  first stretch of the target that continues it, so for a citation occurring *n* and *m*
  times it reports a star to the first occurrence; the cover picks other pairings.
  Both documents stay covered, but the pairings Go reported are gone.
- **Fragmentation** is the price of loose linking noted under flex_gap, and it loses
  whole passages. At 2049/2798 the matches (159806,8123) (159809,8112) (159810,8113)
  (159811,8114) (159816,8131) (159817,8132) (159821,8138) hold a 4-match run anchored at
  (159806,8123). The DP's longest chain instead goes through 159811, the walk cuts it at
  the 17-step target gap into two 3-match pieces, and both are dropped with their
  matches marked used.
- **The gap exception** is a rule of Go's: exceeding `max_gap` in the source only ends
  a run when the window is also sparse or the match leaves it. At 1509/2054 Go carries a
  proclamation across a 26-ngram insertion, 500 bytes; the chains stop at 92.

Patching the chains does not converge: an allowance derived from `best[a]` brings the
lost runs from 1,863 to 1,650 on those pairs, a symmetric gap exception makes it worse
(1,751), and adding star pairings to the cover leaves 742.

### What replaced it

`align_pair` keeps the chains and adds **the anchored scan**, Go's walk run once through
each document. Each of the three sets is merged on its own, then passages overlapping
in both documents are coalesced. The union is symmetric by construction, and merging
before coalescing keeps every passage the chains give inside a result.

| frantext | |
|---|---|
| Go runs not covered | **1** of 48,202, and **0** of 48,995 reversed |
| chains-only records not fully contained | **0** of 63,689 (62,458 identical) |
| records | 65,866, +3.4% |
| symmetry, both flex_gap settings | exact: same counts, 0 of 65,866 without a counterpart |

The one run left is `aim_aim_aim` four times in a row at 2668/2848, a pair Go only
compared because a 32-bit collision (`repercut_etat_conscienc` / `pet_cimeti_aim`) gave
it a fourth shared key. At the record level 101 Go records are under 90% covered; all
of their runs are covered, and the rest is Go's merger bridging unmatched text, up to
152 KB on 141/2946, which the symmetric merger does not do.

On `ecco_clean` every one of the chains' 6,340,308 records is identical (6,311,153) or
fully contained, with 6,424,597 records in all.

The duplicate rule divides by the **smaller** document's count, as before. Dividing by the
larger was tried, to flag only what Go flagged either way round, and reverted: on frantext
the 17 pairs it newly aligned were all one text twice -- two editions of a work, or a work
contained whole in a collected volume -- each giving one record over 97-100% of the
smaller document. That is what the duplicate rule is for, and Go aligned those pairs only
when the larger document happened to be the source.

**Cost**, matching phase only, 32 threads, warm cache:

| | chains only | + anchored scan |
|---|---|---|
| frantext | 0.24 s, 6.0 CPU-s | 0.24 s, 6.0 CPU-s |
| ecco_clean | 18.9 s, 605 CPU-s | 25.9 s, 826 CPU-s (1.36x) |

Two exact skips keep it there. The scan only runs where the DP's longest chain reaches
`min(minimum_matching_ngrams, minimum_matching_ngrams_in_window)`: a run either is a
chain or has that many normally linked matches before its first step past the gap.
Within a pair, stretches of the walked document separated by gaps wider than the window
can grow are independent, and a stretch holding no match with `best` at that length has
no run in it. The largest remaining cost is ordering every match by target for the
reverse walk (about 90 CPU-s on ecco_clean); building the mirrored match arrays in
`align_targets` instead would remove it. Not done: the skip needs `best` for each match,
which the mirrored arrays would have to map back to.

## Open items

- The duplicate rule divides by the smaller document's ngram count, which skips a text
  contained in another; see "Losses against the Go aligner" for why that stays.
- **ecco_clean is not exactly symmetric**, `flex_gap` on: the chains alone report
  6,340,308 records one way and 6,340,303 the other, 9 and 3 of them without a
  counterpart, and with the anchored scan 6,424,597 against 6,424,594, 6 and 1. Within
  the test's tolerance, and consistent with the mirror ties below reaching the output
  on a dense enough corpus.
- The mirror tie could be closed outright by emitting both alternatives, which is
  symmetric and adds records. Not done: nothing currently produces one in the output.
- Nothing downstream was re-tuned. 38% more records changes what `banality_finder.py`,
  `alignment_merger.py` and the graph pipeline see, and any existing alignment database
  will differ from a fresh run.
- **eccotcp costs 1.90x the matching phase with the shipped `flex_gap = true`**, against
  0.93x on frantext. Deriving each link's allowance from `best[a]` should recover most of
  it; see the Cost section.
- The chaining DP still has **7 reachable mirror ties** on the densest classical_chinese
  pair, down from 19. None of them reach the output on the corpora measured, but a corpus
  could surface one.

### A caution about the numba cache

Editing a kernel in place while `NUMBA_CACHE_DIR` holds an entry for it served a stale
compiled version during this work, and produced a plausible-looking 0.28% discrepancy that
did not exist: matcher 8 measured 64,204 records where it actually produces 64,208. Two
hours went into bisecting a bug in the wrong place. Clear the cache directory, or point
`NUMBA_CACHE_DIR` somewhere fresh, after editing a `cache=True` kernel.

## Where the exploration lives

The tests above are the maintained artifacts. What follows is the scratch code the
evaluation was done with, kept because it measures things the tests do not: how far apart
two arbitrary output trees are, which passages one has that the other lacks, and the
matching phase timed on its own.

`/disk1/shared/text-pair-validation/matcher_symmetry_experiment`, alongside the corpora it
runs against. Not in the repository: none of it is production code, the paths inside it
are absolute, and `align.py` carries copies of the rejected matchers.

The scripts import `textpair` from the editable install, so they pick up the working tree.
`corpus.py` caches the parsed corpus arrays as `.npy` on first use, which is what makes an
iteration 3 seconds instead of 13.

```bash
cd /disk1/shared/text-pair
export NUMBA_CACHE_DIR=/var/tmp/textpair_numba OPENBLAS_NUM_THREADS=1
E=/disk1/shared/text-pair-validation/matcher_symmetry_experiment
python $E/verify_extractor.py                      # harness against the real aligner
python $E/run_variant.py --matcher 8 --output /tmp/fwd --threads 32
python $E/run_variant.py --matcher 8 --output /tmp/rev --threads 32 \
  --metadata $E/rev_metadata/metadata.json         # every pair the other way round
python $E/census.py /tmp/fwd /tmp/rev              # symmetry
python $E/bench.py --reps 9 --threads 1 --matchers 0,8
```

| file | |
|---|---|
| `variants.py` | the candidate kernels: `match_v1`, `match_v3`, `match_v4`, `match_v8`, `merge_symmetric`, `component_stats` |
| `align.py` | `inverted_index.align_source` with a `matcher` argument; `matcher == 0` reproduces the shipped output byte for byte |
| `run_variant.py` | whole-corpus run under one matcher, by monkeypatching `inverted_index.align_source` |
| `bench.py` | matching phase alone, corpus and postings preloaded, matchers interleaved across repetitions |
| `census.py` | two trees as undirected passage sets: agreement, only-in-one, overlap |
| `losses.py` | records of one tree with no counterpart, and per-document coverage |
| `pairs.py`, `verify_extractor.py` | per-pair match extraction, checked against `align_source` on 142 compared pairs |
| `inspect_pair.py`, `diff_kernels.py`, `diff_source.py`, `find_reuse_bug.py` | narrowing tools |
| `profile_tree.py`, `tail.py`, `pairdiff.py`, `show_diff.py` | passage-length profiles, per-pair counts, examples |
| `provenance.py` | whether an addition was already found by the reversed baseline |
| `rev_metadata/` | frantext metadata with `year` replaced by `(N - rank)`, which reverses the document order with no ties |

Matcher numbering: `0` shipped, `1` match consumption instead of source reservation,
`2` component statistics, `3` chaining with nothing reserved, `4` chaining with both sides
reserved, `5` `4` + symmetric merger, `6` shipped matcher + symmetric merger, `7` `5` +
symmetric duplicate rule, `8` `7` with no per-pair allocation, `9`-`12` scaffolding that
bisected which buffer reuse looked wrong before the numba cache turned out to be the cause.
`0` and `8` are the two that matter; `8` is the candidate.
