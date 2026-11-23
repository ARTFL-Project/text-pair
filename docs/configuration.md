---
title: Configuring the alignment
nav_order: 4
---

# Configuring the alignment
{: .no_toc }

Every run of `textpair` takes a configuration file. The template, `config/sa_config.ini` in the repository and `/var/lib/text-pair/config/sa_config.ini` once installed, comments every parameter. Most values are reasonable defaults; these are the settings you will want to check.

1. TOC
{:toc}

## `[TEXT_SOURCES]`

| Setting | Description |
|:--------|:------------|
| `source_file_path`, `target_file_path` | Directory of TEI or plain-text files, or of a PhiloLogic database, which TextPAIR detects by itself and takes the text and metadata from. |
| `source_metadata`, `target_metadata` | For plain-text files: a CSV or TSV file with at least the filename of each text. |

With no target, the source texts are compared to one another, and a text is only compared to texts of the same year or later, so that a text is not taken as the source of one written before it.

## `[TEXT_PARSING]`

| Setting | Description |
|:--------|:------------|
| `parse_source_files`, `parse_target_files` | Whether TextPAIR parses the files itself. `yes` by default; set to `no` when relying on the parsed output of a PhiloLogic database. |
| `source_file_type`, `target_file_type` | `tei` or `plain_text`. Plain text needs a metadata file in `[TEXT_SOURCES]`. |
| `source_words_to_keep`, `target_words_to_keep` | A file listing the words to keep, one per line; other words are discarded. Useful for dirty OCR. `all` by default. |

## `[PREPROCESSING]`

| Setting | Description |
|:--------|:------------|
| `source_text_object_type`, `target_text_object_type` | The unit each text is divided into for comparison: `doc`, `div1`, `div2`, `div3`, `para`, or `sent`. `doc` by default. |
| `ngram` | Number of words per n-gram. The default of 3 works well in most cases; lower values tend to produce more uninteresting short matches. |
| `language` | Language used for stemming and other normalization, as a lowercase name such as `french` or `english`. |
| `spacy_model` | The [spaCy model](https://spacy.io/models) used for lemmatization and part-of-speech filtering. Download it first. |

## `[MATCHING]`

`matching_algorithm` is `sa`, for sequence alignment. Leave it so: the other algorithm, `vsa`, is experimental and not supported yet.

The matching parameters in this section, and the settings for passage merging and banality detection, all have defaults that work for most corpora; their comments in the template explain each.
