# Replacing text-preprocessing with a TextPAIR-native preprocessor

**Status:** implemented
**Date:** 2026-09-17
**Module:** `lib/textpair/preprocessing/`, plus `lib/textpair/sequence_alignment/ngram_index.py`

TextPAIR no longer depends on the external `text_preprocessing` library. Preprocessing is
now a pipeline tailored to what TextPAIR actually does: read a PhiloLogic database,
normalize, optionally tag with spaCy, optionally shingle into n-grams.

The n-gram generation stage is **2.5x faster on a 2.2M-word corpus and 4.8x on a
7.7M-word one**, with `ngrams/*.bin`, `ngrams_in_order/*.json` and `metadata.json`
byte-identical. VSA preprocessing is **15.8x faster** at 15 workers, because the old path
did not parallelise at all. Six latent bugs were found and fixed on the way; one of them
answers a `TODO` that was already in the tree.

---

## 1. Why a rewrite rather than a patch

Profiling the old library showed that **only 7% of serial time was the actual text
normalization.** About 75% was building and taking apart a generic object graph, and with
no `spacy_model` configured -- the common SA case -- it was doing that for a
`spacy.blank("en")` pipeline that performs no linguistic work at all:

| Stage | Time | Share |
|---|---|---|
| `Tokens.__init__` + `__get_tokens` (materialise `PreprocessorToken`s) | 1.090s | 26% |
| spaCy `Language.__call__` -> `PreProcessingPipe.__call__` | 1.050s | 25% |
| `process_philo_text` (lz4 + `orjson.loads` + modernize) | 0.866s | 21% |
| `make_spacy_doc` (build `Doc`, set `_.ext` per token) | 0.552s | 13% |
| `generate_ngrams` | 0.472s | 11% |
| **`__normalize_token` -- the normalization itself** | **0.285s** | **7%** |

(`cProfile`, 10 files / 178,908 words / 4.186s, pool bypassed.) The concrete costs were
807,012 `spacy.tokens.underscore.Underscore.__init__` calls purely for the `Token._.ext`
protocol, 405,337 `PreprocessorToken` objects, a second `Doc` rebuilt from scratch in
`PreProcessingPipe`, and `spacy.lang.lex_attrs.word_shape` computing lexeme attributes
nothing read.

That is not micro-optimisation territory: the speed comes from deleting the object graph.
A library whose public types *are* `Tokens` and `PreprocessorToken` cannot shed them and
remain the same library. Three further reasons settled it:

- **TextPAIR used a narrow slice.** `is_philo_db=True` at every call site; TEI and plain
  text are converted to PhiloLogic by `text_parser.py` first. Never used anywhere in
  `lib/` or `api/`: `convert_entities`, `ents_to_keep`, `word_regex`,
  `sentence_boundaries`, `split_tokens`, `Tokens.load`, `appendleft`, `text_loader`.
- **The class-variable design had to break to be fixed** (§3.1, §3.2), which is a
  breaking change for every consumer regardless.
- **GPU scheduling needs pipeline knowledge.** The library's only available answer was
  `workers = 1`, which left the GPU idle through every CPU phase.

`text_preprocessing` stays pinned at v1.1.2 for topologic, which is its only other
consumer.

---

## 2. What was built

```
lib/textpair/preprocessing/
  __init__.py      facade: config in, iterator of TextObject out; owns parallelism
  config.py        PreprocessConfig, built from a [PREPROCESSING] section
  philo_reader.py  bulk lz4 + msgspec Struct decode, text-object grouping
  metadata.py      toms.db lookup, cached per level
  normalize.py     the normalization chain, memoized per surface form
  modernize.py     fr/en maps from editable TSV, with a packed cache
  ngrams.py        contiguous and skipgram generation
  spacy_stage.py   optional spaCy stage, GPU-gated, batched
  tokens.py        columnar TextObject
  data/*.tsv       modernization maps
  tests/           four property checks, two tools; see tests/README.md
lib/textpair/sequence_alignment/ngram_index.py
                   corpus n-gram index, replacing the shell pipeline
```

The decisions that produce the numbers:

- **Columnar text objects.** A `TextObject` is parallel lists of forms, start bytes, end
  bytes, and optionally surface forms and philo positions. This deletes
  `PreprocessorToken`, `Underscore` and the second `Doc`. It also removes a conversion
  rather than adding one: `text_to_ngram` already built three parallel columns by hand,
  and `save_tokens` four.
- **Normalization memoized per surface form.** Surface forms are Zipfian -- **48,569
  distinct forms across 2,152,214 tokens, a 2.26% type/token ratio** -- so the chain runs
  once per distinct form. On its own that step goes from 1.35s to 0.31s with identical
  output. This is only possible because the chain is a pure `str -> str` function, which
  it was not when it was a spaCy pipe over `Token` objects.
- **`msgspec.Struct` decoding and bulk decompression** for input: 0.83s against 1.10s for
  lz4-readline plus `orjson.loads` into dicts.
- **No `spacy.blank`** when no model is configured. The old code loaded and paid for one
  purely to host its `postprocessor` pipe.
- **Instance-owned configuration** throughout, which is what fixes §3.1, §3.2 and §3.4.
- **Modernization maps as sorted TSV** under `data/`, so they can be edited and reviewed,
  with a msgpack cache built beside them on first use and rebuilt whenever the TSV
  changes. Loading drops from 1389ms (importing a 3.3MB `.pyc`) to 14ms. `install.sh`
  packs them ahead of time.
- **Lemma and stopword maps cached per (path, mtime)**, so every `Normalizer` in a
  process shares one copy and an edited file is still picked up. A real lemma map runs
  to a few hundred thousand entries -- `/home/clovis/french_lemmas` is 217,100 -- which
  is ~45MB as a dict; without this each worker would build its own. Under `fork` the
  parent populates the cache before the pool starts, so the pages are shared
  copy-on-write: adding that file to an 8-worker run costs 114MB in total rather than
  360MB.

### Parallelism, and the macOS special case

The old library stored configuration as class variables on `TextFetcher` specifically so
`fork()` would share it copy-on-write instead of pickling the spaCy model and the language
dictionaries per task. That is the direct cause of §3.1 and §3.2.

What replaces it is `ProcessPoolExecutor(initializer=...)`: each worker builds its own
`Normalizer` once from a small picklable `PreprocessConfig`. Same benefit -- no large
state pickled per task -- without depending on fork semantics, so the start method is free
to be `spawn` on Darwin and `fork` elsewhere (`preprocessing.worker_start_method`,
overridable with `TEXTPAIR_START_METHOD`). The copy-on-write benefit is still there under
fork, since `modernize.py` caches the map at module level and the parent populates it, but
nothing depends on it.

That should make `generate_ngrams`'s macOS branch -- which kept preprocessing serial and
fanned out with threads, because forking after the PhiloLogic parse stage's pool
deadlocks (bpo-33725) -- unnecessary. The branch is removed.
**`tests/check_start_method.py` is the confirmation, and needs a macOS run.** On Linux all
three start methods complete and agree on the n-gram count.

Work is pushed into the workers as far as it can go: with no spaCy stage, a worker reads,
normalizes, shingles and runs the caller's post-processing, so SA returns only a small
metadata dict per text object. With a spaCy stage the workers stop after reading, because
tagging has to happen in the process that owns the model.

---

## 3. Bugs found and fixed

### 3.1 Two `PreProcessor` instances could not coexist

`TextFetcher.__init__` was a `@classmethod` writing `language`, `modernize`,
`strip_tags`, `text_object_type`, `token_regex` and `ngram_config` to **class** variables.
VSA builds two (`vector_space_alignment/__init__.py`), so constructing the target silently
overwrote the source: after a source with `text_object_type="doc", ngrams=3` and a target
with `"sent", ngrams=False`, the source read back as `"sent"` with `ngram_config = None`.

This broke exactly what both shipped configs advertise: `source_text_object_type` and
`target_text_object_type` as independent settings, and `target_language`.

### 3.2 `Modernizer` aliased its dictionary across instances

Same root cause, `Modernizer.language_dict` a class variable set from a `@classmethod`
`__init__`. Constructing an English modernizer disabled the French one:

```
french:  estoit -> était
english: estoit -> estoit
french again: estoit -> estoit     # regressed; fr.language_dict is en.language_dict
```

So a French-source / English-target VSA run lost French modernization on the source
corpus entirely.

### 3.3 `gap > 0` double-counted and mostly had wrong byte ranges

`generate_ngrams` took a *reference* to the leading token's `ext` dict and mutated
`end_byte` on it, so every combination sharing a leading token ended up with whichever
range was written last. The sliding window also re-emitted every combination not involving
the newly arrived token. Measured on 4 documents of `rousseau_complete_works` with
`ngram=3, gap=2`:

```
old emitted 915,950 rows (449,683 distinct n-grams)
new emitted 547,093 rows (449,683 distinct n-grams)
duplicate rows in old output: 371,334
old rows carrying the wrong byte range: 271,897 of 544,616 (49.9%)
```

The new implementation anchors each combination on its leading token, so each is emitted
once, and gives each its own range. Identical rows arising from a repeated form inside one
window -- `ou le violon le son` builds `ou_le_son` two ways over the same bytes -- are
collapsed, since that is one occurrence.

This is the one place where output legitimately changes, so it is checked against its own
properties rather than against the old output.

### 3.4 Metadata fields vanished depending on processing order

`recursive_search` cached, per OHCO level, only the fields that had been assigned when the
entry was built. A sibling with a non-empty value therefore poisoned the parent entry for
siblings whose value was empty. Concretely, for the last `div1` of a document, `next` and
`id` were absent entirely, while every other `div1` had them — because the earlier
siblings' own non-empty values kept the document-level entry from ever recording the field.

The new reader caches whole rows, so it is order-independent and reports the value
`toms.db` actually holds.

### 3.5 The index pipeline ate n-grams' leading whitespace

`uniq -c`'s count was stripped with `sub(/^[[:space:]]*[0-9]+[[:space:]]+/, "", $1)`,
whose greedy trailing class also consumed whitespace belonging to the n-gram. Punctuation
separated by spaces normalizes to a space run -- `- - -` becomes two spaces, which clears
`min_word_length` -- and those do reach the index, so entries like `  _idem_avec` were
silently rewritten to `_idem_avec`.

### 3.6 The index pipeline welded an entry at every file boundary

`text_to_ngram` wrote `"\n".join(...)` with no trailing newline, and the pipeline began
`for i in temp/*; do cat "$i"; done`. So the last n-gram of each document was joined to
the first of the next, and `awk`'s `print $1"\t"$2` then silently dropped the third field.
The result was **exactly one corrupt entry per file boundary** -- 40 for a 41-file corpus,
9 for a 10-file one -- with a key field like `-1101233999abandon_la_victoir`.

That answers a `TODO` that was already in `banality_finder.py`:

```python
try:  # TODO: investigate why we don't always get numbers
    common_ngrams.add(int(next(input_file)))
except ValueError:
    pass
```

and the matching `except ValueError` in `tracing.load_ngram_index`. Both guards are kept
for index directories built before this change, with the reason recorded. The writer now
terminates its files, and the reader no longer concatenates them.

Ground truth for `rousseau_complete_works`, counted independently with a dict:
1,125,378 distinct n-grams. The shell pipeline produced 1,125,344 lines and a different
set; the new builder matches exactly.

### Also fixed

- `source_preproc.strip_tags = False` and `pos_to_keep = []` in
  `vector_space_alignment/__init__.py` set attributes nothing read -- both lived on
  `TextFetcher` as class variables -- so they never took effect. Removed rather than made
  to work, so the rendered passages stay identical.
- VSA passed its config section through as `**kwargs`, so `numbers` and
  `minimum_word_length` -- the names the config files use -- were silently dropped and
  fell back to defaults. `PreprocessConfig` now accepts both spellings. The shipped
  defaults match the old fallbacks, so nothing changes for anyone using the stock config.

---

## 4. The n-gram index

The corpus index was built by

```
cat temp/* | sort | uniq -c | sort -rn | awk | tee index.tab | awk > most_common_ngrams.txt
```

which re-sorted input that workers had already sorted, and carried §3.5 and §3.6. It was
also **61% of the generation stage's wall time** on `rousseau_complete_works`.

`ngram_index.build` replaces it. Three ideas:

- **Frequencies come from `ngrams/*.bin`, not from n-gram text.** Each document's CSR
  already holds its distinct keys and the offsets that give their counts, so the corpus
  total is a numpy aggregation over int32 columns rather than a Python loop over 70M
  distinct n-grams. It is also the frequency the aligner acts on: its inverted index is
  over keys, so a key's corpus frequency is the sum over the n-grams that hash to it,
  where counting `(ngram, key)` rows split that frequency between colliding n-grams.
  `most_common_ngrams.txt` therefore has one line per key -- 69,550,968 rather than
  70,120,689 on frantext, the difference being the 566,635 keys that carry more than one
  n-gram. Checked against an independently counted ground truth: same keys, same order.
- **`index.tab` is written only when asked.** Its only reader is the aligner's `--debug`
  tracer, which is gated on the same flag, so it follows `debug`. That also removes the
  per-document n-gram sort and text write that fed it -- 23% of per-document work.

- **Counting is a merge, not a sort.** `LC_ALL=C sort -m` merges the already-sorted files
  externally and `uniq -c` counts adjacent runs. `LC_ALL=C` matters twice: it is the
  collation the workers sorted in, and it makes the output reproducible where the old
  pipeline's tie order followed the ambient locale.

  `--files0-from` is GNU-only, and the paths of 100k files do not fit in a command line,
  so BSD sort (macOS) merges the list in batches of file arguments, each batch written to
  an intermediate, until one invocation can take what is left. The batch is sized from
  `SC_ARG_MAX` and the descriptor limit rather than a fixed guess, and the soft descriptor
  limit is raised toward the hard one first, since macOS ships 256 and would otherwise
  need several more rounds than the machine requires. Both paths are exercised by the
  tests, at a forced batch of three so the rounds run on a handful of files, and verified
  byte-identical on real corpora and at 100k files.

  Using `sort -m` rather than `cat` also disposes of §3.6 structurally: `sort` treats a
  final incomplete line as a line, so there is nothing to weld.
- **Frequency ordering is a counting sort, not a sort.** Each key is appended to a bucket
  for its count as it is produced, and the buckets are concatenated highest-first. Counts
  up to 255 get an exact bucket -- appended in merge order, so ties keep lexicographic
  order -- and rarer higher counts share power-of-two bands small enough to order in
  memory. Nothing proportional to the corpus is resident.

The whole stage on the full frantext (3,630 documents, 6.87GB, 200.7M n-gram
occurrences, 69.6M distinct keys), 32 workers:

| | total | index | workers |
|---|---|---|---|
| shell pipeline | — | — | — |
| `sort -m` over the text | 149.1s | 83.8s | 65.3s |
| counts from the binary indexes | **69.0s** | **16.6s** | **52.4s** |

The index phase had been 56% of the stage; it is now 24%. The merge path is still there
for `index.tab`, and at 100k text objects it cost 12.9s and 0.18GB against the shell
pipeline's 60.6s and 4.50GB (13.6s and 0.26GB on BSD sort's batched route).

`index.tab` is now written in merge (lexicographic) rather than frequency order. Its only
consumer is the aligner's `--debug` tracer, which builds a key -> n-gram dict, so order is
immaterial there; lexicographic also makes the file bisectable by hand.
`most_common_ngrams.txt` keeps frequency-descending order, which the banality filter
depends on.

---

## 5. Results

### Preprocessing alone

`rousseau_complete_works`: 41 files, 2,152,214 words, 1,675,769 3-grams. 16 cores.

SA path (doc-level 3-grams, stemmer + modernize + lowercase + strip_numbers, no model):

| workers | old | new | |
|---|---|---|---|
| 1 | 30.52s | **2.46s** | 12.4x |
| 4 | 9.69s | **1.37s** | 7.1x |
| 15 | 6.57s | **0.98s** | 6.7x |
| 30 | 6.92s | **1.13s** | 6.1x |

VSA path (sentence objects, `keep_all`):

| workers | old | new | |
|---|---|---|---|
| 1 | 29.47s | **3.89s** | 7.6x |
| 15 | 28.76s | **1.82s** | 15.8x |
| 30 | — | **1.71s** | |

The old VSA path got nothing from `workers`: sentence-level objects meant 70,126 `Tokens`
objects, each a deque of `PreprocessorToken` carrying a per-token `ext` dict, pickled back
to the parent. SA escaped this only because its `post_processing_function` collapsed each
document inside the worker.

### The n-gram generation stage end to end, 15 workers

| corpus | words | old | new | |
|---|---|---|---|---|
| `sieyes` | 352k | 4.94s | 3.81s | 1.3x |
| `rousseau_complete_works` | 2.15M | 12.77s | 5.16s | 2.5x |
| `frantext10` | 7.66M | 53.40s | 11.19s | 4.8x |

`sieyes` gains least because 2.55s of it is `import textpair` pulling in torch, spacy and
sentence-transformers -- now the largest single cost on a small corpus, and untouched here.

### spaCy and the GPU

`/disk1/shared/spacy-historic-french-model` (transformer + tagger + morphologizer +
trainable lemmatizer), `lemmatizer = spacy`, `pos_to_keep = NOUN,VERB,ADJ`, on `sieyes`:

| | throughput |
|---|---|
| old (forced to `workers = 1` whenever the GPU was in use) | 34,418 words/s |
| new (15 CPU readers feeding one GPU consumer) | **50,967 words/s** |

The GPU was already being used before -- this was never a missing feature. The problem was
that one process alternated between CPU-bound decoding and GPU inference, so the GPU idled
through every CPU phase: sampled every 500ms, **14 of 44 samples read 0%**. After the
split, **91.7% of wall time is inside `nlp.pipe`**, with normalization at 1.7% and
segmentation at 2.9%. The stage is now transformer-bound, which is the right place to be;
what remains is inside `spacy-transformers`. Batch size barely matters -- the whole
4k-60k token range lands within 2.5% -- because each document is windowed internally.

GPU use is now gated on `cupy` being importable, which only the `cuda` extra installs, so
a CPU install never touches it. The old code called `thinc.api.prefer_gpu()`
unconditionally. `TEXTPAIR_DISABLE_GPU` forces CPU on a CUDA install.

Long documents are segmented so one text object cannot exhaust VRAM. That is not free: a
transformer lemmatizer's output depends on its context window, so a boundary shifts a few
lemmas near it. Measured on `sieyes`, no segmentation reproduces `text_preprocessing`
byte-for-byte, 10k tokens diverges on 0.16% of positions and 2k on 1.35%. The default is
20,000 tokens, high enough that ordinary documents are never split, and close to the
>100k-character threshold at which the old GPU path began splitting. The
`Doc.from_docs` round trip is gone, and with it its
`[W101] Skipping Doc custom extension 'metadata'` warning.

### Import weight, and what it costs per worker

`import textpair` used to cost 2545ms and pull in torch, transformers,
sentence-transformers, sklearn, faiss, spacy, philologic and psycopg2, because the
package `__init__` eagerly imported the VSA orchestrator, the passage classifier, the
PhiloLogic parser and the web loader. None of that is on the sequence-alignment path.

Four entry points -- `parse_files`, `run_vsa`, `classify_passages`, `create_web_app` --
are now resolved on first use through a module `__getattr__`, `torch` is imported inside
`utils.clear_device_cache`, `transformers` inside the classifier, and the LLM client
inside the banality post-eval. `import textpair` is **77ms**, and none of those stacks
load on an SA run.

That matters more than a one-off 2.5s, because `spawn` re-imports the worker's module in
every child, and macOS defaults to `spawn`:

| 8 workers, rousseau | before | after |
|---|---|---|
| peak private memory, `fork` | 994 MB | **455 MB** |
| peak private memory, `spawn` | 5005 MB | **744 MB** |
| pool startup + work, 12 texts, `fork` | 3.55s | **0.22s** |
| pool startup + work, 12 texts, `spawn` | 7.75s | **0.50s** |

It also shortens `fork`: copying the page tables of a 500MB parent fifteen times is not
free, which is why the whole n-gram stage gained more than the import cost alone.

| stage, 15 workers | before the import work | after |
|---|---|---|
| `sieyes` | 3.81s | **0.52s** |
| `rousseau_complete_works` | 5.16s | **1.92s** |

Against the original: `sieyes` 4.88s -> 0.52s (9.4x), `rousseau_complete_works`
12.88s -> 1.92s (6.7x). spaCy is only imported when a `spacy_model` is configured, and
`cupy` only when that model is to run on the GPU.

### Correctness

- **46 token-stream configurations byte-exact** against `text_preprocessing`: 23
  configurations (stemmer, lemmatizer file, stopwords, ascii, lowercase, min word length,
  numbers, punctuation, n-gram sizes, both stemmer languages) at five text-object levels,
  each with and without `keep_all`, compared on every `(form, start_byte, end_byte)`
  triple.
- **Metadata identical** field by field at every level, apart from §3.4 and the sentence
  `word_count` inconsistency, both recognised by shape so a new difference still fails.
- **SA output byte-identical** on `sieyes`, `rousseau_complete_works` and `frantext10`:
  every `ngrams/*.bin`, every `ngrams_in_order/*.json`, and all of `metadata.json`.
- **VSA chunking identical**: all 17,305 chunks on `rousseau_complete_works`, and
  identical under 1 and 15 workers, so parallelism does not perturb order.
- **VSA transformer path bit-identical**: 3,504 chunks, embeddings with a maximum absolute
  difference of 0.000e+00, and all 3,587,336 matches identical with zero similarity delta.
- **spaCy path byte-exact** at the default segmentation threshold.
- **The token cache halved**: 690,999 tokens to 352,466, 17.1MB to 9.2MB, with identical
  content. The old library interleaved a whitespace pseudo-token between every pair of
  words, because spaCy's `Doc` carries a `whitespace_` attribute; with n-grams enabled its
  own `purge()` removed them again, so they only ever reached a caller on the no-n-gram
  path. They were exactly equivalent to joining surface forms with a space, which is what
  `expansion.py` now does.

---

## 6. What is left

- `text_to_ngram` writes three files per text object. At 100k text objects that is 300k
  files, and the writes are a real share of what remains of the stage.
- `Corpus.get_text_chunks` is now the bottleneck in VSA preprocessing: serial Python in
  the main process, while the readers that feed it are parallel.
- The `gap > 0` fix changes output, so a before/after comparison on a real alignment is
  worth doing before relying on it.
- `check_start_method.py` needs a macOS run to confirm the removed special case.
