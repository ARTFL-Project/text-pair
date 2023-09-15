"Nous ne faisons que nous entregloser" Montaigne wrote famously in his <i>Essais</i>... Since all we do is glose over what's already been written, we may as well build a tool to detect these intertextual relationships...

# TextPAIR (Pairwise Alignment for Intertextual Relations)

TextPAIR is a scalable and high-performance sequence aligner for humanities text analysis designed to identify "similar passages" in large collections of texts. These may include direct quotations, plagiarism and other forms of borrowings, commonplace expressions and the like. It is a complete rewrite and rethink of the <a href="https://code.google.com/archive/p/text-pair/">original implementation</a> released in 2009.

While TextPAIR was developed in response to the fairly specific phenomenon of similar passages across literary works, the sequence analysis techniques employed in TextPAIR were developed in widely disparate fields, such as bioinformatics and computer science, with applications ranging from genome sequencing to plagiarism detection. TextPAIR generates a set of overlapping word sequence shingles for every text in a corpus, then stores and indexes that information to be analyzed against shingles from other texts. For example, the opening declaration from Rousseau's Du Contrat Social,

`"L'homme est né libre, est partout il est dans les fers. Tel se croit le maître des autres, qui ne laisse pas d'être plus esclave qu'eux,"`

would be rendered in trigram shingles (with lemmatization, accents flattened and function words removed) as:

`homme_libre_partout, libre_partout_fer, partout_fer_croire, fer_croire_maitre, croire_maitre_laisser, maitre_laisser_esclave`

Common shingles across texts indicate many different types of textual borrowings, from direct citations to more ambiguous and unattributed usages of a passage. Using a simple search form, the user can quickly identify similar passages shared between different texts in one database, or even across databases, such as in the example below.

![alt text](example.png)

## Installation

The recommended install is to build your own Docker image and run TextPAIR inside a container.

### Docker container method

-   Go to the docker folder and build a docker image: `docker build -t textpair .`
-   Start a new container: `docker run -td -p 80:80 --name textpair artfl/textpair init_textpair_db`
    Note that you may want to customize the `run` command according to your needs (e.g. to mount a volume for your data)
    You will need to copy your texts to the container, and then follow the normal procedure described below once inside the container.

If you do run into the issue where the web server does not respond, restart the web server with the following command:
`/var/lib/text-pair/api_server/web_server.sh &`

### Manual installation

If you wish to install TextPAIR on a host machine, note that TextPair will only run on 64 bit Linux, see below.

#### Dependencies

-   Python 3.10 and up
-   Node and NPM
-   PostgreSQL: you will need to create a dedicated database and create a user with read/write permissions on that database. You will also need to create the pg_trgm extension on that database by running the following command in the PostgreSQL shell: `CREATE EXTENSION pg_trgm;` run as a superuser.

#### Install script

See <a href="docs/ubuntu_installation.md">Ubuntu install instructions</a>

-   Run `install.sh` script. This should install all needed components
-   Make sure you include `/etc/text-pair/apache_wsgi.conf` in your main Apache configuration file to enable searching
-   Edit `/etc/text-pair/global_settings.ini` to provide your PostgreSQL user, database, and password.

## Quick start

Before running any alignment, make sure you edit your copy of `config.ini`. See [below](#configuring-the-alignment) for details

#### NOTE: source designates the source database from which reuses are deemed to originate, and target is the collection borrowing from source. In practice, the number of alignments won't vary significantly if you swap source and target

The sequence aligner is executed via the `textpair` command. The basic command is:
`textpair --config=/path/to/config [OPTIONS] [database_name]`

`textpair` takes the following command-line arguments:

-   `--config`: This argument is required. It defines the path to the configuration file where preprocessing, matching, and web application settings are set
-   `--is_philo_db`: Define if files are from a PhiloLogic database. If set to `True` metadata will be fetched using the PhiloLogic metadata index. Set to False by default.
-   `--output_path`: path to results
-   `--debug`: turn on debugging
-   `--workers`: Set number of workers/threads to use for parsing, ngram generation, and alignment.
-   `--update_db`: update database without rebuilding web_app. Should be used in conjunction with the --file argument
-   `--file`: alignment results file to load into database. Only used with the --update_db argument.
-   `--source_metadata`: source metadata needed for loading database. Used only with the --update_db and --file argument.
-   `--target_metadata`: target metadata needed for loading database. Used only with the --update_db and --file argument.
-   `--only_align`: Run alignment based on preprocessed text data from a previous alignment.
-   `--load_only_web_app`: Define whether to load results into a database viewable via a web application. Set to True by default.
-   `--skip_web_app`: define whether to load results into a database and build a corresponding web app

## Configuring the alignment

When running an alignment, you need to provide a configuration file to the `textpair` command.
You can find a generic copy of the file in `/var/lib/text-pair/config/config.ini`.
You should copy this file to the directory from which you are starting the alignment.
Then you can start editing this file. Note that all parameters have comments explaining their role.

While most values are reasonable defaults and don't require any edits, here are the most important settings you will want to checkout:

#### In the TEXT_SOURCES section

This is where you should define the paths for your source and target files. Note that if you define no target, files from source will be compared to one another. In this case, files will be compared only when the source file is older or of the same year as the target file. This is to avoid considering a source a document which was written after the target.
To leverage a PhiloLogic database to extract text and relevant metadata, point to the directory of the PhiloLogic DB used. You should then use the `--is_philo_db` flag.
To link your TextPAIR web app to PhiloLogic databases (for source and target), set source_url and target_url.

#### In the TEXT_PARSING section

-   `parse_source_files`, and `parse_target_files`: both of these setting determine whether you want textPAIR to parse your TEI files or not.
    Set to `yes` by default. If you are relying on parsed output from PhiloLogic, you will want to set this to `no` or `false`.
-   `source_file_type` and `target_file_type`: defines the type of text file: either TEI or plain text. If using plain text, you will need to supply a metadata file in the TEXT_SOURCES section
-   `source_words_to_keep` and `target_words_to_keep`: defines files containing lists of words (separated by a newline) which the parser should keep.
    Other words are discarded.

#### In the Preprocessing section

-   `source_text_object_level` and `target_text_object_level`: Define the individual text object from which to compare other texts with.
    Possible values are `doc`, `div1`, `div2`, `div3`, `para`, `sent`. This is only used when relying on a PhiloLogic database.
-   `ngram`: Size of your ngram. The default is 3, which seems to work well in most cases. A lower number tends to produce more uninteresting short matches.
-   `language`: This determines the language used by the Porter Stemmer as well as by Spacy (if using more advanced POS filtering features, lemmatization, or NER).
    Note that you should use language codes from the <a href="https://spacy.io/models/">Spacy
    documentation</a>.
    Note that there is a section on Vector Space Alignment preprocessing. These options are for the `vsa` matcher (see next section) only. It is not recommended that you use these at this time.

#### In the Matching section

Note that there are two different types of matching algorithms, with different parameters. The current recommended one is `sa` (for sequence alignment). The `vsa` algorith is HIGHLY experimental, still under heavy development, and is not guaranteed to work.

## Run comparison between preprocessed files manually

It is possible to run a comparison between documents without having to regenerate ngrams. In this case you need to use the
`--only_align` argument with the `textpair` command.

Example:

```console
textpair --config=config.ini--only_align --workers=10 my_database_name
```

## Configuring the Web Application

The `textpair` script automatically generates a Web Application, and does so by relying on the defaults configured in the `appConfig.json` file which is copied to the directory where the Web Application lives, typically `/var/www/html/text-pair/database_name`.

#### Note on metadata naming: metadata fields extracted for the text files are prepended by `source_` for source texts and `target_` for target texts.

In this file, there are a number of fields that can be configured:

-   `webServer`: should not be changed as only Apache is supported for the foreseeable future.
-   `appPath`: this should match the WSGI configuration in `/etc/text-pair/apache_wsgi.conf`. Should not be changed without knowing how to work with `mod_wsgi`.
-   `databaseName`: Defines the name of the PostgreSQL database where the data lives.
-   `matchingAlgorithm`: DO NOT EDIT: tells the web app which matching method you used, and therefore impacts functionality within the Web UI.
-   `databaseLabel`: Title of the database used in the Web Application
-   `branding`: Defines links in the header
-   `sourcePhiloDBLink` and `targetPhiloDBLink`: Provide URL to PhiloLogic database to contextualize shared passages.
-   `sourceLabel` and `targetLabel` are the names of source DB and target DB. This field supports HTML tags.
-   `sourceCitation` and `targetCitation` define the bibliography citation in results. `field` defines the metadata field to use, and `style` is for CSS styling (using key/value for CSS rules)
-   `metadataFields` defines the fields available for searching in the search form for `source` and `target`.
    `label` is the name used in the form and `value` is the actual name of the metadata field as stored in the SQL database.
-   `facetFields` works the same way as `metadataFields` but for defining which fields are available in the faceted browser section.
-   `timeSeriesIntervals` defines the time intervals available for the time series functionnality.
-   `banalitiesStored` DO NOT EDIT: defines whether banalities (formulaic passages) have been stored.

Once you've edited these fields to your liking, you can regenerate your database by running the `npm run build` command from the directory where the `appConfig.json` file is located.

Built with support from the Mellon Foundation and the Fondation de la Maison des Sciences de l'Homme.

## Post processing alignment results

TextPAIR produces two (or three if passage filtering is enabled) different files (found in the `output/results/` directory) as a result of each alignment task:

-   The `alignments.jsonl` file: this contains all alignments which were found by TextPAIR. Each line is formatted as an individual JSON string.
-   The `duplicate_files.csv` file: this contains a list of potential duplicate files TextPAIR identified between the source and target databases.
-   The `filtered_passages` file: shows source_passages which were filtered out based on phrase matching. Only generated if a file containing passages to filter has been provided.

These files are designed to be used for further inspection of the alignments, and potential post processing tasks such as alignment filtering or clustering.
