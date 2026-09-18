# N-gram key collisions

**Status:** fixed. Keys are 64-bit as of this branch; §6 records what that cost.
The analysis below is kept because it is why, and because the measurements outlive the decision.
**Date:** 2026-09-18

The aligner matches on a 32-bit MurmurHash3 of each normalized n-gram, not on the text.
Distinct n-grams therefore sometimes share a key, and the aligner cannot tell them apart.
This records how often that manufactures a false passage — measured on two corpora, which
disagree by three orders of magnitude for a reason that turns out to be the input data
rather than the hash —
what such a passage actually looks like, and the whole space of things that could be done
about it, with the cost of each one priced against the code as it stands.

---

## 1. What a key is

```
ngram:  "qu_il_le"                  three stemmed tokens joined with _
key:    -6704081154880228844        mmh3.hash64(ngram)[0], signed 64-bit
```

It was `mmh3.hash32`, a signed int32, until §6; the numbers in sections 2 to 5 are that
key's.

The text survives in exactly one file, and only for humans:

| file | holds | read by |
|---|---|---|
| `ngrams/{doc}.bin` | keys, CSR offsets, byte positions | the aligner — its inverted index is over keys |
| `ngrams_in_order/{doc}.bin` | keys, start bytes | the banality filter |
| `index/most_common_ngrams.txt` | keys, frequency-ordered | the banality filter, top N% as a set |
| `index/index.tab` | `ngram<TAB>key` | only the `--debug` tracer; written only under `debug` |

## 2. How often keys collide

All of section 2 describes the **32-bit** key, which is what made the case for widening
it; §6 has what replaced it.

Full frantext: 3,630 documents, 6.87GB, 200.7M n-gram occurrences, 70,120,689 distinct
n-grams, **69,550,968 distinct keys**.

```
key group sizes:  1 ngram: 68,984,333   2: 563,555   3: 3,074   4: 6
same-key occurrence pairs:  153,228,084,234
of which different ngram:         9,777,766
P(a same-key match is spurious) = 6.4e-05  (1 in 15,671)
```

That matches the birthday estimate, n²/2·2³², to three figures — so it scales as the
square of the distinct n-gram count, and is a property of the corpus size rather than of
the language.

## 3. How often that fabricates a passage

Measured, not modelled. The same corpus was indexed twice under different mmh3 seeds:
collisions under seed 0 are almost surely not collisions under seed 42, so any
disagreement between the two alignments is attributable to them. The aligner was first
confirmed deterministic — identical output across re-runs and across 32 vs 7 threads —
so the differences are real.

| frantext docs | distinct n-grams | passages | discordant | rate |
|---|---|---|---|---|
| 150 | 3.6M | 31,863 | 0 | < 3.1e-05 |
| 600 | 15.1M | 929,066 | 2 | 2.2e-06 |
| 1,200 | 28.7M | 3,530,582 | 4 | 1.1e-06 |
| 3,630 | 70.1M | 25,757,954 | 60 | 2.3e-06 |

**About two spurious passages per million on frantext, flat across a 19x range of
corpus size.**
Collisions grow as n², but passages grow at least as fast, so the per-passage rate does
not worsen over this range. The count is the symmetric difference of the two runs, so a
single run's false-positive rate is roughly half: ~1 in a million.

### The mechanism is one collision, not four

A passage needs `minimum_matching_ngrams` (4) matches. Four independent collisions
aligning is genuinely impossible — p⁴ ≈ 1.7e-17. What actually happens is that **one
collision extends a genuine three-n-gram run to the threshold.** Both artifacts at 600
documents are that shape:

```
Voltaire:     "par les lois de la nature.  Augure Ne faut-il pas..."
Saint-Pierre: "par les lois de la physique, par l'amour de l'ordre..."
              par_le_loi ✓  le_loi_de ✓  loi_de_la ✓  de_la_natur ≡ de_la_physiqu ✗

Saint-Pierre: "hommes, et qui devint la base de la religion et des"
Burke:        "...confirmé par les consécrations de la religion et de"
              de_la_religion ✓  la_religion_et ✓  religion_et_des ≡ religion_et_de ✗
```

Both are three genuine matches plus one collision. This is why the rate is ~1e-06 rather
than ~1e-17: it is first-order in p, not fourth.

### Measured directly, not by proxy

The two-seed method above infers the rate from disagreement between seeds. Once 64-bit
keys existed, the thing itself became measurable: index the same corpus under a 32-bit
hash and a 64-bit one, and every passage in the difference is one that depended on a
collision. Full frantext, same configuration, same 3,630 documents:

| | passages |
|---|---|
| 32-bit keys | 25,757,954 |
| 64-bit keys | 25,757,936 |
| only with 32-bit keys | **20** (7.8e-07) |
| only with 64-bit keys | 2 |

**The proxy was sound.** Section 3 estimated ~1.2e-06 per passage for a single run; the
direct answer is 7.8e-07 over 25.8M passages.

The two passages that exist *only* with 64-bit keys are a case this document did not
anticipate: a collision can also **suppress** a real passage, by inflating the shared-key
count of a document pair past `duplicate_threshold` so the pair is dropped whole, or by
extending a neighbouring passage until the two merge. Collisions do not only add.

One caveat for anyone re-running section 3's table: its 600-document row records 929,066
passages, and `--docs 600` today gives 928,548 at 32 bits and 928,546 at 64. The
difference is which 600 documents, not the keys -- see §7 on byte order against locale
collation. The discordant *counts* reproduce exactly: 2 at 600 documents, by both methods.

### The rate is a property of the corpus's degenerate text, not of the key space

Same method on `eebo_ecco_combo` — 56,846 English EEBO/ECCO transcriptions, 37GB — with
an English stopword list and the Porter stemmer, against frantext for reference:

| corpus | docs | stopwords | distinct keys | passages | discordant | rate |
|---|---|---|---|---|---|---|
| frantext | 600 | no | 15.1M | 929,066 | 2 | 2.2e-06 |
| frantext | 3,630 | no | 69.6M | 25,757,954 | 60 | 2.3e-06 |
| eebo | 2,000 | no | 26.7M | 4,527,541 | 1,356 | 3.0e-04 |
| eebo | 2,000 | yes | 23.9M | 462,770 | 910 | 2.0e-03 |
| eebo | 5,000 | yes | 57.7M | 3,170,053 | 7,728 | 2.4e-03 |

A thousand times frantext's rate, at a *smaller* key population — so it is not the hash
that changed. The artifacts are not spread over the corpus. Every one of the 3,715
passages seed 0 produced and seed 42 did not, at 5,000 documents, is a TCP transcription
placeholder:

```
SOURCE: "〈 in non-Latin alphabet 〉 . 〈 in non-Latin alphabet 〉"
TARGET: "〈 in non-Latin alphabet 〉 , from 〈 in non-Latin alphabet 〉"

SOURCE: "page duplicate〉 〈1 page duplicate〉 〈1 page duplicate"
TARGET: "page duplicate〉 〈1 page duplicate〉 〈1 page duplicate"
```

Classifying the whole run by whether either side contains such a placeholder:

| | passages | discordant |
|---|---|---|
| placeholder boilerplate | 2,653,221 (83.7%) | 3,715 (100%) |
| real text | 516,832 | **0** |

**On real text the eebo rate is 0 of 516,832 passages** — a one-sided bound of 1.9e-06,
which is frantext's number rather than a thousand times it. The elevated rate is entirely
degenerate repeated text: placeholders manufacture an enormous population of
threshold-sized candidate matches that differ only in punctuation, and there a single
collision flips the outcome constantly. Two things follow, and the second is not about
collisions at all:

- The collision rate tracks how much degenerate repetition a corpus contains, not how
  many documents or distinct n-grams it has.
- **84% of the raw aligner output on this corpus is placeholder boilerplate.** The
  banality filter runs after alignment and is meant for exactly this; whether it removes
  these passages at its default `proportion` and `threshold` was not tested here.

### What the stopword list does

Measured on the same 2,000 documents, with and without: the list removes **10% of the
distinct n-grams** (23.9M against 26.7M) and **90% of the passages** (463k against
4.53M), while the fabricated count barely moves (910 against 1,356). Content-word
trigrams are nearly all rare whether or not function words are stripped, so filtering
removes genuine matches far more effectively than manufactured ones: the *rate* rises
6.6x while the absolute problem shrinks. The earlier caveat that "the direction of the net
effect is not obvious" is answered — filtering makes the rate look worse and the output
better.

### What this all looks like under a real corpus config

Everything above uses the no-stopword defaults. frantext as it is actually run
(`/disk1/shared/alignments/frantext/config.ini`: the 476-entry
`FrenchStopwords.txt`, `ascii = yes`, `flex_gap = true`, otherwise the aligner's
defaults) is a different scale of output entirely:

| frantext, 3,630 documents | distinct keys | passages | alignments | per seed |
|---|---|---|---|---|
| §3's defaults, no stopwords | 70,120,689 | 25,757,936 | 15.6GB | 269s |
| the corpus config | 77,006,433 | **63,689** | 0.1GB | 53s |

**404x fewer passages**, from a *larger* distinct-n-gram population — content-word
trigrams are nearly all unique where function-word trigrams repeat endlessly, so the
population barely moves (−10% on eebo, +10% here) while the passage count collapses.

Which puts the whole thing in proportion: at 63,689 passages and ~8e-07 per passage, the
20 collision-dependent passages of §3 become **about 0.05 expected**. A 32-bit key was
never going to visibly damage frantext as it is run. §6's case for widening it rests on
the corpora ahead, not on this one.

The run is still worth doing as a check on the implementation — two seeds, identical
output, identical key counts — but 63,689 passages cannot resolve 1e-06, so it bounds the
rate at 1.6e-05 and no tighter.

### Caveats that remain

- **Two corpora, two languages, neither of them noisy in the usual sense.** frantext is
  clean French; eebo is clean transcription plus placeholder markup. A corpus with real
  OCR noise has a different n-gram population again.
- **The eebo real-text figure is a bound, not a rate.** Zero events over 516,832 passages
  cannot resolve 1e-06; it only says eebo is not visibly worse than frantext.
- **The full eebo corpus is still unmeasured.** 5,000 of 56,846 documents took 230s per
  seed, and the cost grows with the document count squared, so the whole corpus is roughly
  7 to 14 hours per seed. What it would buy is a tighter real-text bound at 11x the
  distinct-n-gram population, where P(spurious per match) is ~10x higher.

## 4. What a spurious passage actually is

Worth being precise about, because it sets the bar every fix has to clear. There are two
classes, and they are nothing alike.

**On frantext: a real echo, one match short.** Each artifact is a **genuine shared phrase
that missed the threshold and was promoted over it by a fake match at its boundary** —
three real n-gram matches, five real shared words, plus one collision. The genuine core is
real reuse, just reuse the configuration decided was too weak to report. So the artifact
is qualitatively the same object as the weakest passage the threshold already admits, and
indistinguishable from one by eye: "par les lois de la nature" against "par les lois de la
physique" is a real five-word echo either way. The harm is a passage that exists on the
strength of one match that is not there.

**On eebo: degenerate repeated markup**, and much the larger class — `〈 in non-Latin
alphabet 〉`, `〈1 page duplicate〉`, matched against itself across thousands of documents.
These are threshold-sized matches between stretches identical apart from punctuation, in
text carrying no authorial content, and a collision flips them constantly because there
are millions of near-identical candidates. They are worthless output with or without the
collision — banality-class passages the downstream filter is meant to remove — so they set
no bar for anything here. They matter to this document only as the explanation for eebo's
raw rate.

There is no second *effect* to worry about in either case. A long legitimate passage
containing a colliding match has one match too many in the matcher's own count, but that
count never leaves the aligner: `aligner/output.py` writes byte ranges, positions and
metadata, and no stage downstream carries a match count at all. A collision's only
observable consequence is a passage existing, or its extent.

Two consequences for what follows:

- **The value of a fix is bounded by the value of marginal passages.** Raising
  `minimum_matching_ngrams` to 5 removes every artifact and 85% of the passages. Whatever
  a fix costs, it is being spent to keep four-match passages honest — which only matters
  if four-match passages matter.
- **Filtering cannot be targeted at "the weak ones".** The threshold-5 experiment says
  most passages sit at the threshold, so "only verify the marginal passages" is not a
  saving: the marginal passages are the corpus.

## 5. Where you can intervene

Three places, in the order the data flows: the key (make collisions impossible or
detectable), the matcher (stop one match from carrying a passage), the output (find the
artifacts after the fact). Then two that sit outside that frame: run the whole thing
twice, or do nothing and say so. Summary first, reasoning after.

| Option | Removes | Costs | Format change |
|---|---|---|---|
| 5.1 Tag column: sort narrow, verify wide (not taken) | all of it | <1% generation, **<0.2% aligner (measured)**, +4.5% to +19% index | yes |
| 5.2 Dictionary-encoded keys | all of it | a global two-pass build, and cross-corpus coordination | yes |
| 5.3 A 64-bit key sorted as 64 bits (**taken**) | all of it | ~0.3% aligner, +19% index, +1.4GB RSS; no new kernel | yes |
| 5.4 `minimum_matching_ngrams = 5` | all of it | 85% of passages | no |
| 5.5 Post-hoc detection from the index | flags most of it | a colliding-key list, and a match set the output does not carry | maybe |
| 5.6 Post-hoc verification from the output text | all of it, exactly | a re-normalization pass over passages | no |
| 5.7 Two-seed consensus | all of it | 2x everything | no |
| 5.8 Nothing, with the rate documented | none | nothing | no |

### 5.1 Sort narrow, verify wide: a tag column

**The idea.** Keep the 32-bit key exactly as it is, and store next to it a second,
independent hash of the same n-gram — a *tag* of 8, 16 or 32 bits. The key does the
indexing and the sorting, unchanged. The tag is only ever compared for equality: two
n-grams match if their keys *and* their tags agree. The effective key width is 32 + tag
bits, and the collision rate falls by 2^tag.

**Why the sort does not have to widen.** The aligner's sort is 32 bits wide because that
is what a key is (`aligner/inverted_index.py`: MSD 12 plus two 10-bit LSD passes). Nothing
downstream needs the postings globally sorted by a wider key. (5.3 records that widening
it would have been affordable anyway — the sort is 0.14% of an alignment — so what this
buys is half the space growth, not speed.) What `index_key_groups` and `align_source` need
is:

1. postings sharing a key are contiguous,
2. within a group they stay in ascending slot order, because a source is compared against
   the suffix of its group (documents are slot-ordered),
3. a group holds one n-gram.

Points 1 and 2 are what the existing sort already delivers. Only point 3 is new, and it
is a *partition*, not a sort: within each key group, separate the postings by tag,
stably. Order between the resulting subgroups is immaterial, since groups are visited
independently. And 566,635 of 69,550,968 keys carry more than one n-gram — **0.8%** —
so for 99.2% of groups the partition is a scan that finds every tag equal and does
nothing.

That work folds into a pass that already exists. `index_key_groups` already walks every
group and already does an indirect load plus a binary search per posting (`_doc_of` over
`key_offsets`); fetching `tags[posting_slots[at]]` in the same loop is a marginal
addition to it, and it is the only place the tag is read.

**The sweep can then be made cheaper than it is now.** After grouping, a posting's key
value is dead — `align_source` uses `posting_keys[q] == u` only as a group-membership
test. So the grouping pass can renumber: write a subgroup id over `posting_keys` in
place, and the same id over `ngram_keys[i]` for each source slot. The sweep condition
becomes `posting_keys[q] == ngram_keys[i]`, dropping the bias arithmetic
(`uint32(int64(ngram_keys[i]) + BIAS)`) it does today — cosmetic, since that is once per
key slot rather than per posting, but it is not a cost either. **No new arrays, and
nothing extra carried through the sort.** The ids need no coordination between the threads
that own different bucket ranges: a subgroup's own start offset in the postings array is
already unique. The one caller that
wants the original keys is `tracing.py`, which names n-grams from `index.tab` under
`--debug`; it keeps a copy, which debug can afford, and would in fact then be able to
name the right n-gram instead of an arbitrary one of a colliding pair.

**What a tag costs.**

| Tag | Index growth (frantext) | Fabricated passages per frantext run |
|---|---|---|
| none | — | ~30 |
| 8 bits | +171 MB (+4.5%) | ~0.1 |
| 16 bits | +342 MB (+9%) | ~5e-04 |
| 32 bits | +684 MB (+19%) | ~7e-09 |

(5.3's 684 MB for widening the keys column by four bytes puts frantext at ~171M
`(document, key)` slots, so a byte of tag per slot is 171 MB. Passage counts are half the
discordant totals of §3, since those are symmetric differences.)

**Measured, on the full frantext index.** By a benchmark that built the postings, grouped
them and ran the real `run_match` twice over the same corpus arrays — once as the aligner stands, once with a
synthesized 16-bit tag carrying §2's measured collision structure. 3,630 documents,
171,052,315 key slots, 200,710,954 positions, 32 threads, four interleaved repeats:

| arm | sort | grouping | match | total |
|---|---|---|---|---|
| baseline | 0.14s | 0.26s | 102.90s | 103.31s |
| tag, no splits | 0.20s | 0.31s | 103.02s | 103.54s |
| tag, real split rate | 0.14s | 0.29s | 102.97s | 103.40s |

Paired within each repeat, so machine drift cancels: grouping **+0.025s**, match
**+0.071s**, total **+0.085s — 0.08% of the alignment**, and 0.20% for the no-split arm.
Both are at the measurement floor: the `sort` column runs identical code in all three arms
and still varies by ±23% of its 0.14s, so per-phase noise is ~±0.03s, the same size as the
effect. The defensible claim is **under 0.2% of a full alignment**, not a specific figure.

Two things the benchmark establishes beyond the cost:

- **The match kernels need no change at all.** After grouping, a posting's key is only
  compared for equality, so the grouping pass can renumber every posting and slot with its
  subgroup id. Store it the way the sweep already expects — `ngram_keys[slot] =
  int32(id - 2**31)`, `posting_keys[at] = uint32(id)` — and `align_source`'s existing
  `uint32(int64(ngram_keys[i]) + BIAS)` reconstructs it exactly. The benchmark's tagged arm
  calls the unmodified `run_match`, `align_source` and `match_passage`.
- **The tagged grouping kernel is correct.** Given a uniform tag, where no group can split,
  it reproduces the baseline's output exactly: 25,755,871 passages and 7,255,173,615
  emissions, identical in both arms.

The tag values are synthesized, not read from an index, since generation does not write
them yet; and which n-gram of a colliding pair a posting belongs to is drawn uniformly,
which is the pessimistic end — real colliding pairs are one frequent n-gram and one rare
one, so real groups split into a large part and a tiny one and move fewer elements. What
the benchmark does not cover is generation, the column on disk, or correctness of the
output, which needs the real generation path and §7's two-seed test.

Generation pays for a second hash. **Keeping the existing key bit-identical matters**: it
keeps every index already built, every `most_common_ngrams.txt` and every reference output
valid, and it means the tag can be added to a corpus without renumbering anything. That
rules out taking key and tag from one `hash128` — mmh3's 32-bit and 128-bit functions are
different algorithms, so the low 32 bits of `hash128` are not `hash32`. Two `hash32` calls
with different seeds is the honest price: 2x hashing, and hashing is 1.9% of per-document
time, so **+1.9% of per-document time, ~0.7% of the generation stage** — the same
conversion 5.3 uses.

**16 bits is the recommendation.** Not for the collision rate — 8 bits already puts the
expected artifact count on frantext below one — but because it makes the acceptance test
clean. Under §7's harness, an index with a 16-bit tag should produce *byte-identical*
alignments under two different seeds; the expected discordant count is 9e-04, so a single
discordant passage is a bug rather than noise. With 8 bits the expectation is 0.23 and the
test has no sharp edge.

**The parts that would have to change**, in rough order of risk:

- `ngram_binary.py`: a sixth column, and a format bump to `TPNG0002`. Append it *last* so
  a `uint8` or `uint16` column cannot misalign the int32 columns that follow — the format
  deliberately has no padding, and keeping the tag last keeps it that way. `_csr` sorts by
  `(key, tag)` instead of key; keys stay ascending signed int32, so the loader's existing
  ordering guarantees hold.
- `ngram_loader.py`: one more column into one more array. Files without a tag column —
  every index built to date, and the whole JSON path — load as tag 0 everywhere, which
  makes every partition a no-op and reproduces today's behaviour exactly, with no branch
  anywhere else.
- `inverted_index.py`: the partition in `index_key_groups`, the renumbering, and the
  `repeats` check. **That check needs attention:** it rejects "the same key twice in one
  document", which today cannot happen because the CSR merges a document's colliding
  n-grams into one key slot. With a tag they become two slots with equal keys, and the
  existing check would fire on them. On frantext, a 55k-distinct-n-gram document has about
  a 0.35 chance of holding an intra-document collision, so this is roughly a thousand
  documents, not a curiosity. The check has to compare subgroups, not key groups.
- Generation: the second hash, and the tag through `write_positions`.
- numba cache invalidation, and reference outputs regenerated — by a known delta: the
  ~30 passages the seed diff already identifies.

**What would not have to change.** The banality filter, `most_common_ngrams.txt` and
`ngrams_in_order/*.json` can all stay keyed on the bare 32-bit key. Their question is
statistical ("is this n-gram in the most common N%"), and a colliding key's frequency
being the sum of two n-grams' frequencies moves nothing at that granularity. Adding the
tag to `ngrams_in_order` would grow a 4 GB pile of JSON for no gain. Worth noting the
inverse, though: per-`(key, tag)` frequencies would be the *more* correct ones —
PREPROCESSING_REWRITE.md §4 records that the current file has 69,550,968 lines where the
n-gram population is 70,120,689, and the tag is exactly what closes that gap if it is ever
wanted.

### 5.2 Dictionary-encoded keys

70.1M distinct n-grams fit in 27 bits, so an exact, collision-free key that is still 32
bits wide exists: number the distinct n-grams. Zero collisions by construction, no format
growth, no tag, the sort untouched.

It founders on coordination. A hash is a pure function of the text, which is why a source
corpus and a target corpus can be indexed independently, years apart, on different
machines, and still align. A dictionary id is a property of the corpus it was built from.
Aligning A against B would need a merged dictionary, or a remap of one side's keys, and
every existing index would be unreadable against a new one. It also needs a global pass
over n-gram *text* — 200.7M strings sorted and deduplicated — which is precisely the
work `ngram_index.build` removed (§4 of PREPROCESSING_REWRITE.md: 83.8s against 16.6s on
frantext, plus 23% of per-document work for the text write).

The same objection applies to the neater variant of it: dictionary-encode *tokens*
instead of n-grams (a far smaller dictionary — ~1M stemmed types) and pack a trigram as
three 21-bit ids, exact in 63 bits. Smaller shared state, but still shared state, and
still 64 bits. Dominated by 5.1, which needs no shared state at all.

### 5.3 A 64-bit key sorted as a 64-bit key

The measurement that started this: at 64 bits the expected collision count over 70.1M
distinct n-grams is 1.3e-04, i.e. none. Hashing is not the problem — `mmh3.hash128 >> 64`
costs 1.56x `hash32` (72.6ns against 46.5ns per n-gram), and hashing is 1.9% of
per-document time, so ~0.7% of the generation stage.

The objection used to be the radix sort: MSD 12 plus two 10-bit LSD passes is exactly 32
bits, 64 bits leaves 52 low bits, so six LSD passes rather than two, in what this document
called the hottest kernel. **That was wrong, and 5.1's benchmark is what showed it.**
`build_postings` is **0.14s of a 103s frantext alignment** — 0.14% — against 102.9s in
`run_match`. Tripling the sort's passes costs ~0.3s, or 0.3% of the alignment. The sort is
not hot; the matcher is.

So the real difference between this and 5.1 is space and code, not speed:

| | index growth | aligner RSS growth | new kernel code |
|---|---|---|---|
| 64-bit key | +19% (684MB) | ~+1.4GB (`ngram_keys` and `posting_keys` both double) | none |
| 16-bit tag | +9% (342MB) | +342MB | the partition and renumber in `index_key_groups` |

Measured on the generation side too, since this is the one place a wide key is *not*
free. `ngram_index.build` buckets by key range over the 32-bit space, so it has to widen.
Running the real aggregation over the full frantext index against a variant whose keys are
int64 (counts left at int32, keys remapped monotonically so they span the 64-bit range):

| | aggregation | most_common_ngrams.txt |
|---|---|---|
| int32 keys | 15.90s, 16.00s | 764MB |
| int64 keys | 18.41s, 18.51s | 1,417MB |

**+2.5s, +15.7% of the aggregation** — which is 16s of a 69s generation stage, so **+3.6%
of generation**, or ~4% with `hash64`'s extra cost. One time per corpus, and nothing on the
aligner side. The spill grows to 8+4 bytes per key slot from 4+4, the argsort runs on
int64, and the decimal text is 1.9x the bytes. `ngrams_in_order/*.json` grows in the same
proportion, 3.8GB to ~5.5GB on frantext, which is the concrete argument for making that
file binary columnar like `ngrams/*.bin`.

A 64-bit key is the *simpler* option — no partition, no renumbering, no subgroup-aware
repeats check, just a wider column and a wider radix. It costs twice the space for
identity bits nobody needs, since 48 bits already gives ~350k colliding pairs over 10
billion n-grams. **This is what was built**, and §6 says why: the space it costs came back
out of `ngrams_in_order` in the same change, and an exact key leaves the frequency
semantics exact too.

### 5.4 Raising `minimum_matching_ngrams` to 5

Measured. It does remove the artifacts: at 600 documents the two seeds become identical.
But **passages drop from 929,066 to 134,328** — 85% of matches lost to remove two false
positives. Not a trade worth making.

**Structural variants collapse into it.** The tempting move is to demand corroboration
rather than count: two matched n-grams at consecutive indices on both sides mutually
confirm each other, and a single collision cannot produce both. But the artifacts already
*have* three mutually corroborating matches — the collision sits at the boundary, where
the run has no neighbour to be checked against. Any rule that trims an uncorroborated
boundary match off each end is arithmetically the same as asking for
`minimum_matching_ngrams + 2`, and costs more than 5 does. A boundary match is where a
fake match is undetectable without the text, and that is not fixable by counting.

### 5.5 Post-hoc detection from the index

The original sketch here was to publish the colliding keys and flag the passages that
depend on one. Reading the code makes both halves more expensive than they looked.

**Publishing the colliding keys is not free.** `ngram_index.build` aggregates over
`ngrams/*.bin`, which holds keys and offsets — *not* n-gram text. Two n-grams sharing a
key are indistinguishable in that input, so collisions are invisible to the pass that
would supposedly reveal them. Finding them means the `(ngram, key)` pairs, i.e. restoring
the per-document text write (23% of per-document work) and the `sort -m` merge over text
(83.8s against 16.6s on frantext) that the preprocessing rewrite deleted. The one cheap
route to a colliding-key list is a key-only numpy pass over `(key, tag)` columns — which
means doing 5.1 first, after which nothing needs flagging.

**"Load-bearing" is not a lookup either.** The precise condition — would this passage
still reach `minimum_matching_ngrams` without its colliding matches — needs the passage's
*match set*. The output carries index ranges and a count, not the matched pairs.
`ngrams_in_order` gives every key in a byte span, not which of them matched. So the test
has to intersect the two spans' keys and re-chain them under the same window, gap and
merge rules — a second implementation of `match_passage`, which is the classic way to
build a checker that disagrees with the kernel for reasons having nothing to do with
collisions.

Both problems are avoidable, and the way to avoid the second is what 5.6 does.

### 5.6 Post-hoc verification from the output text

The one post-hoc option that is both exact and free of format change. Alignment records
already carry `source_passage` and `target_passage`, and the generation config is dumped
alongside the output. So a pass over the results can, for each passage:

1. re-normalize and re-tokenize both passage texts under the dumped config,
2. build their n-grams as *strings* and intersect on the text, giving exact keys,
3. call **the same** `match_passage` kernel with the same parameters,
4. compare: a passage that no longer reaches `minimum_matching_ngrams` was fabricated by
   a collision; one that reaches it with fewer matches had its count inflated.

Step 3 is what makes this worth more than 5.5 — the chaining rules are not reimplemented,
they are imported, so the verifier cannot drift from the matcher. It needs no colliding-key
list, no index change and no format bump, and it can *correct* the match counts rather
than only flag them.

Its cost is a Python re-normalization of two short spans per passage, which is
embarrassingly parallel but not cheap at 25.8M passages, and much less cheap if the config
uses spaCy (`pos_to_keep`, `lemmatizer`). Its risk is fidelity: the re-normalization has
to reproduce generation exactly — same modernization maps, same stemmer, same
`minimum_word_length` — or it reports false alarms. The saving grace is that it is
verifiable against the seed-diff oracle: it should flag the ~30 passages the two-seed run
disagrees on, and nothing else.

A cheaper deployment of the same machinery: run it **on demand**, for the one passage a
scholar is looking at, from the web app. Near-zero cost, exact where it matters, and it
answers "is this real?" rather than annotating a million passages that are.

### 5.7 Two-seed consensus

Index and align twice under different seeds and keep only the passages both runs produce.
A fabricated passage would have to be fabricated twice, by two different collisions over
the same pair of n-grams, so the residual is second-order in the collision probability —
effectively zero.

Almost free to build, because §7's harness is most of it: it already loads both runs'
passages keyed on `(source_doc_id, target_doc_id, byte ranges)` and takes their symmetric
difference, so the intersection is the same set operation. What it does not do is write
anything out; a consensus run also needs a filter pass over the `result_batches` lz4
files, which is an afternoon rather than a project.

Keeping the intersection also drops genuine passages whose *extent* a collision changed
under one seed, since both variants of such a passage miss the intersection. That loss is
bounded by the same ~1e-06 and is the conservative direction to err in.

The cost is 2x everything: two indexes, two alignments, ~30GB of intermediate alignments
on frantext. A bad permanent policy, and a perfectly reasonable one-off for a run that is
about to be published.

### 5.8 Nothing, with the rate documented

The null option deserves stating plainly, because §4 makes it stronger than it sounds. One
passage in a million, each one a real five-word echo that was one match short, in output
that also contains banalities, OCR noise and boilerplate. Recording the measured rate
where the corpus is described — a noise floor — may serve a scholar better than a column
that is empty 999,999 times out of a million.

This is also the answer to "is a flag useful?": a per-passage flag at this rate is mostly
a column nobody reads. What is actionable is either a correction (5.1, 5.6) or a stated
noise floor (5.8), not an annotation.

One caution about stating it: the noise floor is a property of the corpus, not of the
aligner. It is ~2e-06 on frantext, below 1.9e-06 on eebo's real text, and 1.2e-03 on
eebo's raw output including placeholders. Publishing any one of those as "the" rate would
be wrong; what travels between corpora is the method, not the number.

## 6. What was done

**64-bit keys (5.3), not the tag (5.1).** The rate on real text never justified this: 20
passages in 25.8M, and the artifacts are five-word echoes one match short of the
threshold. What justified it is scale. The n-gram population grows almost linearly in
documents — 23.9M distinct keys at 2,000 eebo documents, 57.7M at 5,000, β ≈ 0.96 — which
puts the full eebo corpus near 600M distinct n-grams, where 7% of n-grams share a key
against frantext's 0.8%, and a million documents of that size near 10 billion, past the
32-bit space entirely. At the largest index the aligner can load, 2.07e9 n-grams, a
32-bit key has 24% of them sharing a key; 64 bits expects 0.1 collisions. A key width is
decided before the indexes are built, not after, which is the whole argument for doing it
now rather than when it bites.

5.1 would have cost half the space for identity bits nobody needs. 5.3 costs no new kernel
and leaves the frequency semantics exact, which the aggregation and any IDF-style
weighting want anyway.

### What it cost, measured on full frantext

| | 32-bit | 64-bit | |
|---|---|---|---|
| generation, 60 workers | 58.0s | 56.5s | no measurable change |
| alignment, 60 workers | 212s | 203s | no measurable change |
| passages | 25,757,954 | 25,757,936 | 20 collision-dependent, 2 collision-suppressed |
| `ngrams/` | 3.78GB | 4.46GB | +18% |
| `ngrams_in_order/` | 3.99GB | **2.41GB** | −40%, from the binary rewrite |
| `most_common_ngrams.txt` | 0.76GB | 1.43GB | +87%, longer decimals |
| **index total** | **8.53GB** | **8.30GB** | **−2.7%** |

The wide key paid for itself on disk and in time, because `ngrams_in_order` moved to binary
in the same change. Generation gains 2.5s in the aggregation, which has to bucket over a
64-bit key range, and gives back ~11.6 CPU-s of orjson across the workers; the two cancel
to within run-to-run noise. Resident arrays in the aligner are the one unambiguous cost,
computed rather than measured: +21% on frantext (6.5 → 7.9GB), +22% at eebo scale
(27.9 → 34.1GB).

**Acceptance test: zero.** Two seeds must now produce byte-identical alignments, and on
the full corpus §7's harness reports `IDENTICAL over 25,757,936 passages`, a bound of
3.9e-08, where a 32-bit key gave 60 discordant. Both seeds also report exactly 70,120,689
distinct keys — §2's distinct *n-gram* count, against 69,550,968 distinct 32-bit keys. The
key count and the n-gram count are now the same number.

### What changed

- `ngram_binary.py`: `TPNG0002`, keys int64. `TPNG0001` still loads — `columns` returns a
  file's keys in their own width and the loader widens them on assignment — but an index
  can only be aligned against another built with the same hash, since a 32-bit and a
  64-bit key of the same n-gram are unrelated values. Regenerating is ~1 minute for a
  frantext-sized corpus. Also holds the new `TPIO0001` for n-grams in order.
- `aligner/inverted_index.py`: the bias is `uint64(key) ^ 0x8000000000000000` rather than
  `+ 2^31`, since int64 cannot hold 2^63 and the sign-bit flip is the same monotone map.
  `posting_keys` is uint64, and the radix is MSD 12 plus **four 13-bit LSD passes**, which
  covers the low 52 bits exactly. The sort being 0.14% of an alignment, the passes were
  chosen to divide evenly rather than to be few.
- `aligner/ngram_loader.py`: int64 key arrays, including the JSON path.
- `generate_ngrams.py`: `hash64(form)[0]`, the low half of MurmurHash3-128.
- `ngram_index.py`: keys int64, counts still int32 — widening per-document occurrence
  counts would inflate the spill for nothing — and key-range edges over the 64-bit space.
- `ngrams_in_order/{doc}.bin`: two binary columns replacing a JSON array of
  `[start_byte, key]` pairs, 12 bytes an n-gram against 30. `banality_finder.NgramDoc`
  reads it with two `np.frombuffer` views and `np.searchsorted`, where it used to parse
  1.6MB of text per document with orjson (4.7ms, once per source document in the
  results). `documents.py` names it in the metadata `ngrams` field, now `.bin`.
- `tests/test_ngram_order.py`: the new format, and that the range lookup answers exactly
  what the old bisect answered — random ranges, every exact boundary, repeated start
  bytes, an empty document, keys at both int64 extremes.

### Still open

- **The full eebo corpus is unmeasured**, at ~7 to 14 hours per seed. With 64-bit keys the
  two-seed test there is a check on the implementation rather than a rate measurement,
  which makes it less urgent than it was.
- **Positions are still int32, and this is not a corpus-size limit — nor a manual
  concern any more.** `ngram_loader` refuses past 2^31 n-gram positions, but that applies
  to what one `runner._run_combination` loads, and a combination loads **at most two
  batches**: the diagonal of the batch triangle sweeps one batch against itself, the
  off-diagonal loads a source batch plus a target batch. So the ceiling is on
  `source_batch + target_batch`, roughly 153,000 eebo-sized documents or 39,000
  frantext-sized ones, and a corpus larger than that is aligned by raising `source_batch`
  — `k` batches give `k(k+1)/2` combinations covering every unordered document pair
  exactly once.

  `runner._size_batches` now does that raising itself, from the `max_positions`
  parameter: positions per document come from the binary headers, which the loader's
  sizing pass reads anyway, so a combination's load is known before anything is
  allocated. It only ever raises an explicit `source_batch`, and says so when it does.
  Measured on 600 frantext documents — 33,228,373 positions, every header read in under
  10ms — budgets of half and a fifth of the corpus raised the count to 5 and 14, wrote the
  15 and 105 batch files the triangle predicts, and found the same 928,546 passages as a
  single batch. The cost is per-combination overhead: 7s, 38s and 209s for 1, 5 and 14
  batches at that size, which is why the sizing raises the count only when the alternative
  is failing. At a scale where batching is actually required, matching dominates and the
  overhead disappears.

  Verified rather than assumed: the same 600 documents at `source_batch=1` and
  `source_batch=3` give **identical** passages, 928,546 either way, the batched run writing
  the 6 batch files the triangle predicts. It costs k-fold index loading — each batch is
  loaded once per combination it appears in — which at 600 documents turned 6s into 22s
  because loading is the whole run at that size, and which at a million documents is about
  an hour against weeks of matching.

  Widening the offsets would therefore buy larger *batches*, not a larger corpus: fewer,
  bigger combinations and less re-loading. It means int64 `position_offsets`,
  `posting_slots` and `sweep_starts` — all three hold global indices — and reworking
  `align_source`'s `packed_positions`, which packs a global position index into the high
  half of an int64 and so has its own 2^32 wall; the fix there is to pack
  document-relative positions and add the base inside `match_passage`. The trade, from the
  measured eebo figures:

  | | per document | ceiling per combination |
  |---|---|---|
  | int32 offsets | 599.6 KB | 153,414 documents, the guard |
  | int64 offsets | 761.4 KB | 262,673 documents, RAM at 200GB usable |

  1.7x per combination for +27% resident arrays, which itself drops the RAM-bound ceiling
  from 333,578 documents to 262,673. For a million documents it turns 14 batches and 105
  combinations into 8 and 36, with identical total comparisons: it saves index loading, not
  matching. Not worth doing until re-loading shows up in a profile.

- **`most_common_ngrams.txt` is now 1.4GB of decimal text**, and `ngrams_in_order` showed
  what binary is worth. The same argument applies to it.

## 7. Reproducing

`lib/textpair/sequence_alignment/tests/check_hash_collisions.py` runs the whole
experiment: index the corpus under two seeds, align both, confirm the aligner is
deterministic, diff the passages. The preprocessing options are arguments, since §3 shows
the normalization changes the answer; the defaults are frantext's.

```bash
cd /disk1/shared/text-pair/lib
T=textpair/sequence_alignment/tests

# frantext, the §3 defaults: French, stemmer, modernize, no stopword list
python $T/check_hash_collisions.py --philo-db /var/www/html/philologic5/frantext \
    --work /disk1/tmp-textpair/hc --docs 600 --workers 32

# eebo, English with a stopword list
python $T/check_hash_collisions.py \
    --philo-db /var/www/html/philologic5/eebo_ecco_combo \
    --work /disk1/tmp-textpair/hc-eebo --docs 5000 --workers 60 \
    --language english --stopwords /home/clovis/english_stopwords.txt --keep
```

It prints the discordant rate, or the bound when the two runs agree. Each run's passages
are reduced to a sorted key file and its alignment is deleted before the next run starts,
so the peak is one alignment rather than three; `--keep` keeps the indexes and alignments,
which is what the boilerplate classification in §3 needed. The diff is `sort`/`comm` over
those key files rather than Python sets, because at eebo scale a set of every passage does
not fit in memory.

Note that `--docs N` takes Python's sorted prefix, which is byte order; `ls | head -N` uses
locale collation and picks a slightly different set, so the two are not interchangeable
when comparing runs.

Costs, measured: frantext 600 documents 40s, frantext 3,630 documents 25 minutes and
~30GB of alignments; eebo 2,000 documents 64s per seed, eebo 5,000 documents 230s per seed
and 1.1GB of alignments.
