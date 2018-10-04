# Text-Align

A scalable and high-performance sequence aligner for large collections of texts

Built in collaboration with <a href="https://www.lip6.fr/?LANG=en">LIP6</a>

## Installation

Note that Text-Align will only run on 64 bit Linux and MacOS. Windows will NOT be supported.

### Dependencies

-   Python 3.6 and up
-   Node and NPM
-   PostgreSQL: you will need to create a dedicated database and create a user with read/write permissions on that database. You will also need to create the pg_trgm extension on that database using the `CREATE EXTENSION pg_trgm;` run as a superuser in the PostgreSQL shell.
-   A running instance of Apache

### Install script

-   Run `install.sh` script. This should install all needed components
-   Make sure you include `/etc/text-align/apache_wsgi.conf` in your main Apache configuration file to enable searching
-   Edit `/etc/text-align/global_settings.ini` to provide your PostgreSQL user, database, and password.

## Quick start

Before running any alignment, make sure you make a copy of `/var/lib/text-align/config/config.ini` to your working directory, and edit the various fields. In particular, make sure you provide a table name to store your results in PostgreSQL.

The sequence aligner is executed via the `textalign` command.

`textalign` takes the following command-line arguments:

-   `--config`: path to the configuration file where preprocessing, matching, and web application settings are set
-   `--source_files`: path to source files
-   `--source_metadata`: path to source metadata. Only define if not using a PhiloLogic database.
-   `--target_files`: path to target files. Only define if not using a PhiloLogic database.
-   `--target_metadata`: path to target metadata
-   `--is_philo_db`: Define if files are from a PhiloLogic database. If set to `True` metadata will be fetched using the PhiloLogic metadata index. Set to False by default.
-   `--output_path`: path to results
-   `--debug`: turn on debugging
-   `--workers`: Set number of workers/threads to use for parsing, ngram generation, and alignment.
-   `--load_web_app`: Define whether to load results into a database viewable via a web application. Set to True by default.

Example:

```console
textalign --source_files=/path/to/source/files --target_files=/path/to/target/files --config=config.ini --workers=6 --output_path=/path/to/output
```

## Alignments using PhiloLogic databases

This currently uses the dev version of PhiloLogic5 to read PhiloLogic4 databases. So you'll need a version of PhiloLogic5 installed.
A future version will have that functionnality baked in.

To leverage a PhiloLogic database, use the `--is_philo_db` flag, and point to the `data/words_and_philo_ids` directory of the PhiloLogic DB used.
For instance, if the source DB is in `/var/www/html/philologic/source_db` and the target DB is in `/var/www/html/philologic/target_db`,
run the following:

```console
textalign --is_philo_db --source_files=/var/www/html/philologic/source_db/data/words_and_philo_ids/ --target_files=/var/www/html/philologic/target_db/data/words_and_philo_ids/ --workers=8 --config=config.ini
```

Note that the `--is_philo_db` flag assumes both source and target DBs are PhiloLogic databases.

## Run comparison between preprocessed files manually

It's possible run a comparison between documents without having to regenerate ngrams. In this case you need to use the
`--only_align` argument with the `textalign` command. Source files (and target files if doing a cross DB alignment) need to point
to the location of generated ngrams. You will also need to point to the `metadata.json` file which should be found in the `metadata`
directory found in the parent directory of your ngrams.

-   `--source_files`: path to source ngrams generated by `textalign`
-   `--target_files`: path to target ngrams generated by `textalign`. If this option is not defined, the comparison will be done between source files.
-   `--source_metadata`: path to source metadata, a required parameter
-   `--target_metadata`: path to target metadata, a required parameter if target files are defined.

Example: assuming source files are in `./source` and target files in `./target`:

```console
textalign --only_align --source_files=source/ngrams --source_metadata=source/metadata/metadata.json --target_files=target/ngrams --target_metadata=target/metadata/metadata.json --workers=10 --output_path=results/
```

## Configuring your web application

Your web application can be configured from the `appConfig.json` file located in the directory of your web application. By default, this is in `/var/www/html/text-align/YOUR_DB_NAME`. Once you have configured the file, you will need to run `npm run build` to regenerate the web application.
