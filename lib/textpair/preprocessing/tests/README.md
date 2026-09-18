# Preprocessing checks

Same split as the aligner's checks: the prefix says which kind a script is.

**`test_*.py` assert properties and need nothing but the package.** They build their own
corpora and PhiloLogic databases in temporary directories, so there are no fixtures to
keep in step. `run_tests.py` runs all of them in a few seconds.

**`check_*.py` are tools you point at something.** It does not run by default.

## Running them

```bash
cd /disk1/shared/text-pair/lib
T=textpair/preprocessing/tests

python $T/run_tests.py                     # all five
python $T/run_tests.py --only ngrams       # one of them
```

| check | what it pins down |
|---|---|
| `normalization` | the order of the chain: stopwords are tested before and after lemmatization, `min_word_length` after stemming, `ascii` after that. Reordering any of them changes results, and nothing else would notice. |
| `ngrams` | byte ranges, and the `gap > 0` path: every combination emitted once, each with a range covering its own tokens. |
| `config` | that two preprocessors stay independent, and that the `[PREPROCESSING]` names are honoured. |
| `reader` | text-object grouping per level, punctuation reattachment, sentence metadata, `keep_all` placeholders. |
| `worker_imports` | that the alignment path pulls in none of the heavy optional stacks — torch, transformers, spaCy, faiss and the rest — which `spawn` would re-import in every worker. |

## check_start_method.py

Confirms which multiprocessing start methods survive a pool being created after an
earlier one has torn down. This is the question `generate_ngrams` used to answer with a
macOS special case: `fork()` there deadlocks when the PhiloLogic parse stage's own pool
has just been torn down (bpo-33725), so that stage kept preprocessing serial and fanned
out with threads instead. `preprocessing.worker_start_method` now picks `spawn` on Darwin
and `fork` elsewhere, which should make the special case unnecessary — but only a macOS
run can show that.

```bash
python $T/check_start_method.py                 # all start methods, default workers
python $T/check_start_method.py --workers 8 --texts 24 --timeout 180
```

It builds its own corpus, so it needs no data. Each start method runs in a subprocess
under a timeout, so a deadlock is reported as `HUNG` rather than hanging the run. It
fails if the platform's default does not work, or if two start methods disagree on the
n-gram count.

It also reports which route the n-gram index will take: GNU sort passes the whole file
list through `--files0-from` in one merge, BSD sort merges batches of file arguments in
rounds. Both are checked by `sequence_alignment/tests/test_ngram_index.py`, but the
batched route is the one macOS actually uses.

On Linux (64 cores, Python 3.13) all three complete and agree:

```
method       status       ngrams    inner     wall
fork         ok             4776     0.11     0.22
spawn        ok             4776     0.38     0.50
forkserver   ok             4776     0.37     0.50
```

`spawn` re-imports the worker's module in every child, so these numbers depend on
`import textpair` staying cheap. If they regress into seconds, something has been added
to a module the workers import.

`TEXTPAIR_START_METHOD` overrides the choice if you need to force one.
