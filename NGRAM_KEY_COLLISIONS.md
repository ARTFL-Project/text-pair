# N-gram key collisions

**Status:** measured, not acted on
**Date:** 2026-09-18

The aligner matches on a 32-bit MurmurHash3 of each normalized n-gram, not on the text.
Distinct n-grams therefore sometimes share a key, and the aligner cannot tell them apart.
This records how often that manufactures a false passage, why the obvious mitigations are
worse than the problem, and what a real fix would look like.

---

## 1. What a key is

```
ngram:  "qu_il_le"        three stemmed tokens joined with _
key:     172795159        mmh3.hash32 of that string, signed 32-bit
```

The text survives in exactly one file, and only for humans:

| file | holds | read by |
|---|---|---|
| `ngrams/{doc}.bin` | keys, CSR offsets, byte positions | the aligner — its inverted index is over keys |
| `ngrams_in_order/{doc}.json` | `(start_byte, key)` | the banality filter |
| `index/most_common_ngrams.txt` | keys, frequency-ordered | the banality filter, top N% as a set |
| `index/index.tab` | `ngram<TAB>key` | only the `--debug` tracer; written only under `debug` |

## 2. How often keys collide

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

**About two spurious passages per million, flat across a 19x range of corpus size.**
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

### Caveats on these numbers

- **No stopword list.** These runs used the default config, which has `stopwords` empty.
  A stopword list removes many n-grams, changing both the distinct-n-gram count (fewer
  collisions) and the match density (fewer near-threshold genuine runs). The direction of
  the net effect on the rate is not obvious and was not measured.
- **One corpus, one language.** frantext is clean French with `modernize` and the Porter
  stemmer. A noisier corpus has a different n-gram population.
- **frantext is not large.** `eebo_ecco_combo` is 56,846 documents and 38.7GB, 5.6x
  frantext; `ap_split` is 113,474 documents. Distinct n-grams there could reach a few
  hundred million, putting P(spurious per match) an order of magnitude higher. Whether
  the per-passage rate follows is unmeasured — on frantext it did not, because passages
  grew too, but that depends on how much genuine reuse a corpus contains.

## 4. Mitigations that do not work

**Raising `minimum_matching_ngrams` to 5.** It does remove the artifacts: at 600
documents the two seeds become identical. But **passages drop from 929,066 to 134,328** —
85% of matches lost to remove two false positives. Not a trade worth making.

**A 64-bit key.** At 64 bits the expected collision count over 70.1M distinct n-grams is
1.3e-04, i.e. none. The hash is not the problem: `mmh3.hash128 >> 64` costs 1.56x
`hash32` (72.6ns against 46.5ns per n-gram), and hashing is 1.9% of per-document time,
so ~0.7% of the stage.

The cost is in the aligner. `aligner/inverted_index.py` sorts `posting_keys` with a
hand-written radix — MSD 12 bits plus two 10-bit LSD passes, **exactly 32 bits**. 64-bit
keys need six LSD passes, so 3 to 7 passes over T postings in the hottest kernel. Plus
`posting_keys` doubles from uint32 to uint64, the `.bin` keys column goes int32 to int64
(+19% on the n-gram indexes, 684MB on frantext), a format version bump, loader changes,
and numba cache invalidation. Roughly 2x on the aligner's sort to remove a 1e-06 effect.

## 5. What a real fix might look like

The idea worth pursuing is post-hoc: **flag, rather than prevent.** A colliding key is
knowable from the index, so a passage built on one can be marked without changing the
matching path at all.

Sketch, in rough order of cost:

1. **Publish the colliding keys.** During index generation, the aggregation already sorts
   all keys; keys carrying more than one distinct n-gram fall out of the same pass at
   negligible cost. Write them to `index/colliding_keys.bin` as a sorted int32 array —
   566,635 keys is 2.3MB on frantext. This much is cheap and useful on its own.
2. **Flag suspect passages.** A post-alignment step, alongside the banality filter, that
   for each passage looks up its keys in `ngrams_in_order` and marks it when a colliding
   key is load-bearing: that is, when removing the colliding matches would drop the
   passage below `minimum_matching_ngrams`. That is the precise condition — a passage
   with 12 matches, one of them colliding, is not at risk.
3. **Confirm against the text.** For a flagged passage only, compare the actual n-gram
   text at the two positions. `index.tab` is lossy here (one text per key), so this needs
   either the full `(ngram, key)` list for the colliding keys only — small, since there
   are few — or a re-normalization of the two passages. This would make the check exact
   rather than a flag, and it runs over a handful of passages per million.

Open questions before building any of it:

- Is a flag useful to a scholar, or does it just add a column nobody reads? Step 3 is what
  makes it actionable, since it turns "possibly spurious" into "certainly spurious".
- Should step 3 simply drop the passage? At ~1e-06 the volume is small enough that
  dropping is defensible, and it would make the output collision-free without touching the
  aligner.
- Does the rate hold at eebo scale? Section 3's method — two seeds, diff the alignments —
  transfers directly; it costs two index builds and two alignments. That should be
  measured before deciding this matters at all.

## 6. Reproducing

`lib/textpair/sequence_alignment/tests/check_hash_collisions.py` runs the whole
experiment: index the corpus under two seeds, align both, confirm the aligner is
deterministic, diff the passages.

```bash
cd /disk1/shared/text-pair/lib
T=textpair/sequence_alignment/tests

# a prefix of a corpus, which is how the rate was measured against size
python $T/check_hash_collisions.py --philo-db /var/www/html/philologic5/frantext \
    --work /disk1/tmp-textpair/hc --docs 600 --workers 32

# the whole thing; needs room for two n-gram indexes and two alignments
python $T/check_hash_collisions.py --philo-db /var/www/html/philologic5/frantext \
    --work /disk1/tmp-textpair/hc --workers 32
```

It prints the discordant rate, or the bound when the two runs agree. `--keep` leaves the
indexes and alignments behind. Note that `--docs N` takes Python's sorted prefix, which is
byte order; `ls | head -N` uses locale collation and picks a slightly different set, so
the two are not interchangeable when comparing runs.

The frantext numbers in section 3 cost roughly: 600 documents 40s, 3,630 documents 25
minutes and ~30GB of alignments.
