# Preprocessing checks

Same split as the aligner's checks: the prefix says which kind a script is.

**`test_*.py` assert properties and need nothing but the package.** They build their own
corpora and PhiloLogic databases in temporary directories, so there are no fixtures to
keep in step. `run_tests.py` runs all of them in a few seconds.

**`check_*.py` are tools you point at something.** Neither runs by default.

## Running them

```bash
cd /disk1/shared/text-pair/lib
T=textpair/preprocessing/tests

python $T/run_tests.py                     # the four property checks
python $T/run_tests.py --only ngrams       # one of them
```

| check | what it pins down |
|---|---|
| `normalization` | the order of the chain: stopwords are tested before and after lemmatization, `min_word_length` after stemming, `ascii` after that. Reordering any of them changes results, and nothing else would notice. |
| `ngrams` | byte ranges, and the `gap > 0` path: every combination emitted once, each with a range covering its own tokens. |
| `config` | that two preprocessors stay independent, and that the `[PREPROCESSING]` names are honoured. |
| `reader` | text-object grouping per level, punctuation reattachment, sentence metadata, `keep_all` placeholders. |

## check_parity.py

Compares this package against the `text_preprocessing` library it replaced, token for
token, over 23 configurations at five text-object levels.

```bash
# needs the old library still importable
PARITY_FIXTURES=/tmp/parity python $T/check_parity.py /path/to/philo_db 4
```

It needs a stopword list at `$PARITY_FIXTURES/stopwords.txt` and a lemmatizer map at
`$PARITY_FIXTURES/lemmas.tsv`; any content will do, they only have to exist. It reports
each configuration, then metadata field by field, then the `gap > 0` properties.

Two metadata differences are expected and recognised by shape rather than by field name,
so a genuinely new one still fails. Both fix order-dependent behaviour in the old
library: a field empty at an object's own level could be dropped entirely depending on
which sibling was processed first, and a sentence's `word_count` was inherited from its
paragraph for the last sentence in each file. `gap > 0` is deliberately not byte-exact —
the old implementation double-counted and gave every combination sharing a leading token
one byte range, wrong on half the rows — so it is checked against its own properties
instead.

This is a migration tool, not a permanent check. Once `text_preprocessing` is gone from
the environment it stops being runnable, and the `test_*.py` files are what remain.

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
