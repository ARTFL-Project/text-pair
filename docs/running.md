---
title: Running an alignment
nav_order: 3
---

# Running an alignment
{: .no_toc }

1. TOC
{:toc}

## Before you start

Copy the configuration template to the directory you will run the alignment from, and edit it:

```console
cp /var/lib/text-pair/config/sa_config.ini my_config.ini
```

At minimum, set `source_file_path`. See [Configuring the alignment](configuration.html) for the settings that matter most.

{: .note }
The **source** is the collection reuses are deemed to originate from, and the **target** the collection borrowing from it. In practice, the number of alignments won't vary significantly if you swap the two. With no target, the source texts are compared to one another.

## The `textpair` command

```console
textpair --config=my_config.ini [OPTIONS] database_name
```

A full run parses the texts, generates n-grams, aligns them, loads the results into PostgreSQL, and builds a web application in the `web_app_path` directory (`/var/www/html/text-pair/database_name` by default), served at `/text-pair/database_name` on your server. Intermediate files and results are written to `--output_path` (`output` by default); see [Output files](output.html).

## Common tasks

Align again without regenerating n-grams, reusing the preprocessed files of a previous run in the same output path:

```console
textpair --config=my_config.ini --only_align --workers=10 database_name
```

Align without loading the results into a database or building a web application:

```console
textpair --config=my_config.ini --skip_web_app database_name
```

Load the results of an earlier run into the database and build its web application:

```console
textpair --config=my_config.ini --load_only_web_app --file=output/results/alignments.jsonl.lz4 database_name
```

Delete a database and its web application:

```console
textpair --delete database_name
```

## Options

| Option | Description |
|:-------|:------------|
| `database_name` | Name of the database, also used for its PostgreSQL tables. At most 30 characters. |
| `--config` | Path to the configuration file. Required, except with `--delete`. |
| `--output_path` | Where intermediate files and results are written. Default: `output`. |
| `--workers` | Total number of workers for parsing, n-gram generation, and alignment, which divides them between matching and writing. Default: 4. |
| `--only_align` | Skip parsing and n-gram generation and go straight to the aligner, using the preprocessed files in `--output_path`. |
| `--skip_web_app` | Do not load the results into a database or build a web application. |
| `--load_only_web_app` | Load the results file given with `--file` into the database and build the web application, without parsing or aligning. |
| `--update_db` | Load the results file given with `--file` into the database without rebuilding the web application. Needs `--source_metadata` and `--target_metadata`. |
| `--file` | Alignment results file to load, with `--load_only_web_app` or `--update_db`. |
| `--source_metadata`, `--target_metadata` | Source and target metadata files, with `--update_db`. |
| `--graph_only` | Rebuild the semantic graph of an existing alignment, with the `[GRAPH]` settings of the configuration file, and publish it to the web application. |
| `--delete` | Delete the database's tables and its web application. |
| `--debug` | Turn on debugging. |
