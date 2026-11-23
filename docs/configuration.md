---
title: Configuring the alignment
nav_order: 4
---

# Configuring the alignment
{: .no_toc }

Every run of `textpair` takes a configuration file. Start from the template, `config/sa_config.ini` in the repository and `/var/lib/text-pair/config/sa_config.ini` once installed: it lists every setting with a comment on what it does, and its values are sensible defaults. [Read the template on GitHub](https://github.com/ARTFL-Project/text-pair/blob/master/config/sa_config.ini).

This page covers the choices you will actually make, and how the settings behind each one work together. What it leaves out is in the template's comments.

1. TOC
{:toc}

## What gets compared

In `[TEXT_SOURCES]` and `[TEXT_PARSING]`, with `text_object_type` in `[PREPROCESSING]` and `sort_by` in `[MATCHING]`.

`source_file_path` is required. With `target_file_path` as well, every source text is compared with every target text; leave it empty to compare the source texts with one another. Each path is a directory of TEI or plain-text files, or a PhiloLogic database, which TextPAIR recognizes and takes the text and metadata from.

| Setting | Template value | What it does |
|:--------|:---------------|:-------------|
| `source_file_type`, `target_file_type` | `tei` | `tei` or `plain_text`. Plain text needs `source_metadata` or `target_metadata`: a CSV or TSV file with at least the filename of each text. |
| `parse_source_files`, `parse_target_files` | `yes` | Set to `no` for a PhiloLogic database, which has already parsed its texts. |
| `source_text_object_type`, `target_text_object_type` | `doc` | The unit compared: `doc`, `div1`, `div2`, `div3`, `para`, or `sent`. Each unit is aligned as a text of its own, with its own metadata, so passages do not cross from one unit into the next. |
| `sort_by` | `year` | The metadata field texts are ordered by. When the source is compared with itself, each text is compared only with the texts after it, so a text is never taken as the source of one written before it. |

## How texts are normalized

In `[PREPROCESSING]`, with `words_to_keep` in `[TEXT_PARSING]`.

TextPAIR compares n-grams of normalized words, not the words as spelled. Each normalization lets passages match through differences of spelling or inflection, and each also makes distinct words alike: pushed too far, it brings false matches.

Words left out (stopwords, numbers, short words, parts of speech not kept) are dropped before the n-grams are formed, so an n-gram joins the words on either side of them. That is how "L'homme est né libre, et partout" gives `homme_libre_partout`.

| Setting | Template value | What it does |
|:--------|:---------------|:-------------|
| `language` | empty, read as `french` | The language for stemming and modernization, as a lowercase name such as `french` or `english`. |
| `modernize` | `yes` | Maps older spellings to modern ones before anything else. French and English only. |
| `lemmatizer` | empty | Replaces each word with its lemma: a file of forms and lemmas separated by a tab, `spacy` with `spacy_model` set, or `philologic` for the lemmas a PhiloLogic database stored. |
| `stemmer` | `yes` | Reduces each word to its stem, with the Snowball stemmer for `language`. |
| `stopwords` | empty | A file of words to leave out, one per line. |
| `minimum_word_length`, `numbers` | `2`, `yes` | Leave out words shorter than this, and words containing a digit. |
| `pos_to_keep` | empty | Keep only these parts of speech, as comma-separated [Universal POS tags](https://universaldependencies.org/u/pos/), tagged by `spacy_model`, or by PhiloLogic when no model is set. |
| `spacy_model` | empty | The [spaCy model](https://spacy.io/models) for `lemmatizer = spacy` and `pos_to_keep`. Download it first. |
| `source_words_to_keep`, `target_words_to_keep` | `all` | A file of the only words to keep, one per line. Useful for dirty OCR. |
| `target_language` | empty | Only when the target texts are in another language than the source texts. |

## How strict matching is

In `[MATCHING]`, with `ngram` in `[PREPROCESSING]`.

`ngram` is the number of words in an n-gram, 3 in the template. Lower values match more short, uninteresting runs; higher ones demand longer stretches of the same wording, and miss quotations with small changes.

A passage is a run of n-grams two texts share, in the same order. Distances are counted in n-grams, which is to say in the words left after normalization:

1. Two texts are compared only if they share at least `minimum_matching_ngrams_in_docs` n-grams (4 in the template).
2. A run starts at a shared n-gram, and takes the next one if it lies within `max_gap` n-grams (15) in both texts.
3. Within each stretch of `matching_window_size` n-grams (30), the run needs `minimum_matching_ngrams_in_window` shared n-grams (4) to go on.
4. The run is kept as a passage if it holds at least `minimum_matching_ngrams` distinct shared n-grams (4): an n-gram repeated, as in "ha, ha, ha... ha, ha, ha", counts once.
5. With `flex_gap` (`true`), a run that has reached `minimum_matching_ngrams` may then bridge wider gaps, up to about `matching_window_size`, so an established passage carries on through an insertion or a reworded clause.

To catch shorter reuses, lower `minimum_matching_ngrams`, and expect more commonplaces and formulas. To follow looser reuses, raise `max_gap`; to hold passages to close wording, lower it. `minimum_matching_ngrams_in_window` and `matching_window_size` set how dense the shared n-grams must be: raise the first, or lower the second, to drop sparse, scattered matches.

Passages found close together in both texts are then merged into one, by the `merge_passages_on_byte_distance`, `passage_distance_multiplier` and `merge_passages_on_ngram_distance` settings. Pairs of texts sharing `duplicate_threshold` percent (80) of the smaller text's n-grams are skipped as duplicates, and listed in `duplicate_files.csv`.

## What gets filtered out

In `[MATCHING]`.

Not every shared passage is a reuse worth reading: formulas, titles, and salutations recur across a corpus. Two filters set them aside.

| Setting | Template value | What it does |
|:--------|:---------------|:-------------|
| `banality_auto_detection` | `true` | Flags passages made of n-grams common across the corpus, short passages that are mostly TEI paratext (headings, notes, stage directions), short ones made of numbers that differ between the two sides, and ones made mostly of names. |
| `banality_threshold` | `0.4` | How common a passage's n-grams must be on average, from 0 to 1, for it to be flagged. Lower values flag more passages; past a point, widely quoted passages are flagged along with formulas. |
| `store_banalities` | `false` | With `false`, flagged passages are moved to `banal_alignments.jsonl.lz4`. With `true`, they stay in the database, and the search form has a filter to hide them or show only them. |
| `phrase_filter` | empty | A file of phrases, one per line: a passage containing one, on either side, is removed and written to `filtered_passages.jsonl.lz4`. Phrases match whole words, ignoring case, accents, and punctuation. |

## Large corpora

In `[MATCHING]`.

| Setting | Template value | What it does |
|:--------|:---------------|:-------------|
| `source_batch`, `target_batch` | `1` | Splits each side into this many batches, compared batch against batch, so less is held in memory at once, at the cost of more passes. When the source is compared with itself, use 3 or more: two batches loaded together are the whole collection. TextPAIR raises the count itself, and says so, when a batch would hold more than the aligner can load. |

The `--workers` option of the [`textpair` command](running.html#options) sets how many cores the run uses.

## Everything else

The other normalizations (`ascii`, `lowercase`, `gap`, `word_order`), the context kept around each passage (`context_size`), and the `[GRAPH]` section are explained in the template's comments.
