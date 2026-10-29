#!/usr/bin/env python3
"""Web loading module"""

import ast
import html
import io
import json
import glob
import os
import re
import shutil
import sys
from collections import OrderedDict
from typing import Any

import lz4.frame
import orjson
import psycopg2
from pgvector.psycopg2 import register_vector
from tqdm import tqdm

from .parse_config import read_global_config
from .web_settings import SETTINGS_FILE, SETTINGS_KEYS, show_view, write_app_settings

DEFAULT_FIELDS = {
    "rowid",
    "group_id",
    "source_doc_id",
    "target_doc_id",
    "source_passage",
    "source_start_byte",
    "source_end_byte",
    "source_context_before",
    "source_context_after",
    "source_start_position",
    "source_end_position",
    "source_passage_length",
    "target_passage",
    "target_start_byte",
    "target_end_byte",
    "target_context_before",
    "target_context_after",
    "target_start_position",
    "target_end_position",
    "target_passage_length",
    "source_year",
    "target_year",
}

DEFAULT_FIELD_TYPES = {
    "source_year": "INTEGER",
    "source_pub_date": "INTEGER",
    "target_year": "INTEGER",
    "target_pub_date": "INTEGER",
    "source_start_byte": "INTEGER",
    "target_start_byte": "INTEGER",
    "source_end_byte": "INTEGER",
    "target_end_byte": "INTEGER",
    "source_start_position": "INTEGER",
    "target_start_position": "INTEGER",
    "source_end_position": "INTEGER",
    "target_end_position": "INTEGER",
    "source_passage_length": "INTEGER",
    "target_passage_length": "INTEGER",
    "similarity": "FLOAT",
    "llm_stance": "TEXT",
    "llm_reasoning": "TEXT",
    "group_id": "INTEGER[]",  # Define directly as INTEGER[]
    "count": "INTEGER",
    "banality": "BOOLEAN",
    "embedding": "VECTOR",  # rendered before validation; never part of the generated DDL
}

FILTERED_FIELDS = {
    "source_philo_seq",
    "source_parent",
    "source_prev",
    "source_next",
    "source_philo_name",
    "source_philo_type",
    "source_word_count",
    "target_philo_seq",
    "target_parent",
    "target_prev",
    "target_next",
    "target_philo_name",
    "target_philo_type",
}

YEAR_FINDER = re.compile(r"^.*?(\d{1,}).*")
TOKENIZER = re.compile(r"\w+")
CONTROL_CHARS = dict.fromkeys(range(32))
CONTROL_CHAR_FINDER = re.compile(r"[\x00-\x1f]").search


# Language names of the [PREPROCESSING] language option, as BCP 47 codes for the web app's lang attributes
LANGUAGE_CODES = {
    "arabic": "ar",
    "basque": "eu",
    "catalan": "ca",
    "chinese": "zh",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "finnish": "fi",
    "french": "fr",
    "german": "de",
    "greek": "el",
    "hebrew": "he",
    "hindi": "hi",
    "hungarian": "hu",
    "indonesian": "id",
    "irish": "ga",
    "italian": "it",
    "japanese": "ja",
    "latin": "la",
    "lithuanian": "lt",
    "nepali": "ne",
    "norwegian": "no",
    "polish": "pl",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "serbian": "sr",
    "spanish": "es",
    "swedish": "sv",
    "tamil": "ta",
    "turkish": "tr",
    "yiddish": "yi",
}


def language_code(language: str) -> str:
    """BCP 47 code of a language name such as "french"; empty when unknown."""
    language = language.strip().lower()
    if re.fullmatch(r"[a-z]{2,3}(-[a-z0-9]+)*", language):
        return language
    return LANGUAGE_CODES.get(language, "")


def philo_db_name(db_path: str) -> str:
    """Title of a PhiloLogic database: dbname in its web_config.cfg, else its directory name."""
    try:
        with open(os.path.join(db_path, "data/web_config.cfg"), encoding="utf8") as config_file:
            tree = ast.parse(config_file.read())
    except (OSError, SyntaxError, ValueError):
        tree = ast.Module(body=[], type_ignores=[])
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "dbname" for t in node.targets):
            try:
                name = ast.literal_eval(node.value)
            except ValueError:
                continue
            if isinstance(name, str) and name.strip():
                return name.strip()
    return os.path.basename(os.path.normpath(db_path))


def philo_database_label(source_path: str, target_path: str, source_against_source: bool) -> str:
    """Header title for a database built from PhiloLogic: its name, or "source vs target" for two databases."""
    source_name = philo_db_name(source_path)
    target_name = source_name if source_against_source else philo_db_name(target_path)
    label = source_name if target_name == source_name else f"{source_name} vs {target_name}"
    return html.escape(label, quote=False)


class WebAppConfig:
    """Web app config class"""

    def __init__(
        self,
        db_name: str,
        api_server: str,
        source_philo_db_path: str,
        target_philo_db_path: str,
        algorithm: str,
        store_banalities: bool,
        source_against_source: bool,
        textpair_params=None,
    ):
        with open("/var/lib/text-pair/config/appConfig.json", encoding="utf8") as app_config:
            self.options: OrderedDict = json.load(app_config, object_pairs_hook=OrderedDict)
        self.options["apiServer"] = api_server
        self.options["appPath"] = os.path.join("/text-pair", db_name)
        self.options["databaseName"] = db_name.lower()
        self.options["matchingAlgorithm"] = algorithm
        self.options["sourcePhiloDBPath"] = source_philo_db_path
        if source_against_source is True:
            self.options["targetPhiloDBPath"] = source_philo_db_path
        else:
            self.options["targetPhiloDBPath"] = target_philo_db_path
        self.options["sourceLinkToDocMetadata"] = "source_title"
        self.options["targetLinkToDocMetadata"] = "target_title"
        self.options["banalitiesStored"] = store_banalities
        self.options["showSemanticGraph"] = False  # until a graph is published
        if textpair_params is not None:
            if textpair_params.is_philo_db:
                self.options["databaseLabel"] = philo_database_label(
                    source_philo_db_path, target_philo_db_path, source_against_source
                )
            for direction in ("source", "target"):
                language = textpair_params.preprocessing_params[direction].get("language", "")
                self.options[f"{direction}Language"] = language_code(language)
            self.options["themeLanguage"] = language_code(textpair_params.graph_params.get("label_language", ""))

        # Populate passage classification if available
        if textpair_params and hasattr(textpair_params, "passage_classification"):
            classes = textpair_params.passage_classification.get("classes", {})
            if classes:
                self.options["passageClassification"] = [
                    {"label": label, "description": desc} for label, desc in classes.items()
                ]
            else:
                self.options["passageClassification"] = []
        else:
            self.options["passageClassification"] = []

    def __call__(self) -> dict[str, Any]:
        return self.options

    def __getattr__(self, attr: str) -> Any:
        return self.options[attr]

    def searchable_fields(self) -> list[str]:
        """Return list of all searchable fields"""
        fields = []
        for field in self.options["metadataFields"]["source"]:
            fields.append(field["value"])
        for field in self.options["metadataFields"]["target"]:
            fields.append(field["value"])
        return fields

    def update(self, available_fields: list):
        """Only store fields that are actually in the table in config"""
        source_fields = []
        target_fields = []
        for field in self.options["metadataFields"]["source"]:
            if field["value"] in available_fields:
                source_fields.append(field)
        for field in self.options["metadataFields"]["target"]:
            if field["value"] in available_fields:
                target_fields.append(field)
        self.options["metadataFields"]["source"] = source_fields
        self.options["metadataFields"]["target"] = target_fields

        source_fields = []
        target_fields = []
        for field in self.options["facetsFields"]["source"]:
            if field["value"] in available_fields:
                source_fields.append(field)
        for field in self.options["facetsFields"]["target"]:
            if field["value"] in available_fields:
                target_fields.append(field)
        self.options["facetsFields"]["source"] = source_fields
        self.options["facetsFields"]["target"] = target_fields

        source_fields = []
        target_fields = []
        for field in self.options["sourceCitation"]:
            if field["field"] in available_fields:
                source_fields.append(field)
        for field in self.options["targetCitation"]:
            if field["field"] in available_fields:
                target_fields.append(field)
        self.options["sourceCitation"] = source_fields
        self.options["targetCitation"] = target_fields


def copy_data(params, direction):
    """Copy files and database file to web app directory"""
    print(f"Copying {direction} text files to web app directory...", end="", flush=True)
    if os.path.exists(f"{params.web_app_config['web_application_directory']}/{params.dbname}/{direction}_data"):
        os.system(f"rm -rf {params.web_app_config['web_application_directory']}/{params.dbname}/{direction}_data")
    os.makedirs(
        f"{params.web_app_config['web_application_directory']}/{params.dbname}/{direction}_data/data/TEXT",
        exist_ok=True,
    )
    os.system(
        f"cp {params.output_path}/{direction}/db.locals.py {params.web_app_config['web_application_directory']}/{params.dbname}/{direction}_data/data/"
    )
    with open(params.paths[direction]["metadata_path"], encoding="utf8") as metadata_file:
        metadata = json.load(metadata_file)
    # Deduplicated: metadata has one entry per text object, not per file, so
    # below the doc object type copying per entry recopies whole files repeatedly.
    text_dir = os.path.join(
        params.web_app_config["web_application_directory"], params.dbname, f"{direction}_data", "data", "TEXT"
    )
    for filename in dict.fromkeys(file["filename"] for file in metadata.values()):
        shutil.copy(filename, text_dir)
    os.system(
        f"cp {params.output_path}/{direction}/toms.db {params.web_app_config['web_application_directory']}/{params.dbname}/{direction}_data/data/"
    )
    print("done")


def parse_file(file):
    """Parse tab delimited file and insert into table"""
    with lz4.frame.open(file) as input_file:
        for line in input_file:
            yield orjson.loads(line)


def clean_text(text):
    """Clean passages for HTML viewing before storing"""
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


# COPY ... FORMAT text delimits on tabs and newlines, and reads \N as NULL.
COPY_ESCAPES = str.maketrans({"\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t"})
NEEDS_ESCAPE = re.compile(r"[\\\t\n\r]").search


def render_vector(value) -> str:
    """Render an embedding as a pgvector literal.

    9 significant digits is the shortest decimal that round-trips float32,
    which is what pgvector stores whatever we send it.
    """
    if value is None:
        return "\\N"
    return "[" + ",".join(f"{v:.9g}" for v in value) + "]"


class CopyStream(io.RawIOBase):
    """Present an iterator of encoded rows as the file object copy_expert reads."""

    def __init__(self, rows):
        self.rows = rows
        self.buffer = bytearray()

    def readable(self) -> bool:
        return True

    def readinto(self, target) -> int:
        wanted = len(target)
        buffer = self.buffer
        for row in self.rows:  # generator resumes where the last call left it
            buffer += row
            if len(buffer) >= wanted:
                break
        read = min(wanted, len(buffer))
        target[:read] = buffer[:read]
        del buffer[:read]
        return read


ALIGNMENTS_PER_CHUNK = 2000
# Below a couple of chunks the pool costs about what it saves.
PARALLEL_LOAD_THRESHOLD = 5_000
# Past this the parent's decompress-and-feed loop is the limit, not the cores.
MAX_LOAD_WORKERS = 8

_COPY_WORKER: dict[str, Any] = {}


def load_workers(textpair_params) -> int:
    """How many cores to load with, from --workers, capped where scaling stops."""
    requested = int(getattr(textpair_params, "workers", 1) or 1)
    return max(1, min(requested, MAX_LOAD_WORKERS))


def prepare_alignment(alignment_fields, rowid, embeddings):
    """Fill in the derived columns a stored alignment needs."""
    alignment_fields["rowid"] = rowid
    alignment_fields["passage_id"] = rowid
    alignment_fields["source_passage_length"] = len(TOKENIZER.findall(alignment_fields["source_passage"]))
    alignment_fields["target_passage_length"] = len(TOKENIZER.findall(alignment_fields["target_passage"]))
    categories = alignment_fields.get("passage_categories") or []
    alignment_fields["target_first_class"] = categories[0] if len(categories) > 0 else ""
    alignment_fields["target_second_class"] = categories[1] if len(categories) > 1 else ""
    alignment_fields["target_third_class"] = categories[2] if len(categories) > 2 else ""
    if embeddings is not None:
        alignment_fields["embedding"] = render_vector(embeddings[rowid - 1])  # rowid is 1-indexed
    return alignment_fields


def chunk_alignment_file(file, chunk_size):
    """Yield (rowid of first line, raw lines) so a worker can number its own rows."""
    rowid = 1
    with lz4.frame.open(file) as input_file:
        batch = []
        for line in input_file:
            batch.append(line)
            if len(batch) == chunk_size:
                yield rowid, b"".join(batch)
                rowid += chunk_size
                batch = []
        if batch:
            yield rowid, b"".join(batch)


def _init_copy_worker(table_name, field_order, database_config, embeddings_spec):
    """Give a worker its own validator, embedding view and database connection."""
    import atexit

    _COPY_WORKER["validate"] = RowValidator(field_order, DEFAULT_FIELD_TYPES)
    _COPY_WORKER["statement"] = copy_statement(table_name, field_order)
    _COPY_WORKER["embeddings"] = None
    if embeddings_spec is not None:
        import numpy as np

        path, rows, dimensions = embeddings_spec
        _COPY_WORKER["embeddings"] = np.memmap(path, dtype="float32", mode="r", shape=(rows, dimensions))
    connection = psycopg2.connect(**database_config)
    tune_load_session(connection.cursor())
    _COPY_WORKER["connection"] = connection
    atexit.register(connection.close)


def _copy_alignment_chunk(job) -> int:
    """Decode, validate and COPY one chunk. Runs in a worker process."""
    first_rowid, blob = job
    validate = _COPY_WORKER["validate"]
    embeddings = _COPY_WORKER["embeddings"]
    connection = _COPY_WORKER["connection"]

    lines = []
    rowid = first_rowid
    for raw_line in blob.splitlines():
        record = prepare_alignment(orjson.loads(raw_line), rowid, embeddings)
        lines.append(render_copy_line(validate(record)))
        rowid += 1

    cursor = connection.cursor()
    cursor.copy_expert(_COPY_WORKER["statement"], io.BytesIO(b"".join(lines)), size=1 << 20)
    connection.commit()
    return len(lines)


def parallel_copy_alignments(file, table_name, field_order, count, workers, database_config, embeddings_spec):
    """Decode and COPY the alignment file across processes.

    Each worker COPYs on its own connection, so no rendered row ever travels
    back to the parent and the COPY itself parallelizes too. Chunks may land
    out of order; that is harmless because a row's identity is its rowid, which
    comes from its chunk's offset in the file, not from insertion order.

    Submission is windowed rather than handed to the pool all at once: the
    executor pickles a task as soon as it is submitted, so an unbounded loop
    would pull the whole decompressed file into the call queue.
    """
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from .preprocessing import worker_start_method

    jobs = chunk_alignment_file(file, ALIGNMENTS_PER_CHUNK)
    window = workers * 3
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context(worker_start_method()),
        initializer=_init_copy_worker,
        initargs=(table_name, field_order, database_config, embeddings_spec),
    ) as executor:
        pending = []
        for job in jobs:
            pending.append(executor.submit(_copy_alignment_chunk, job))
            if len(pending) >= window:
                break
        with tqdm(total=count, leave=False) as progress:
            while pending:
                done = pending.pop(0)
                next_job = next(jobs, None)
                if next_job is not None:
                    pending.append(executor.submit(_copy_alignment_chunk, next_job))
                progress.update(done.result())  # re-raises whatever the worker hit


MAX_CONCURRENT_INDEX_BUILDS = 8
# Each concurrent build gets its own, so the peak is the cap times this.
CONCURRENT_INDEX_WORK_MEM = "512MB"


def create_indexes(statements, database_config, workers):
    """Run CREATE INDEX statements concurrently, each on its own connection.

    Builds on one table take a SHARE lock, which is compatible with itself, so
    they do not block each other. Worth doing because the trigram GIN indexes
    over the passage columns take longer than everything else put together, and
    Postgres cannot parallelize a single GIN build before version 18.

    Those same GIN builds are started first: the phase cannot finish before the
    slowest one does, so anything shorter should be filling in behind it.
    """
    statements = sorted(statements, key=lambda statement: "USING GIN" not in statement)
    concurrency = min(len(statements), workers, MAX_CONCURRENT_INDEX_BUILDS)

    def build(statement, work_mem):
        connection = psycopg2.connect(**database_config)
        try:
            cursor = connection.cursor()
            tune_load_session(cursor, work_mem)
            cursor.execute(statement)
            connection.commit()
        finally:
            connection.close()

    if concurrency < 2:
        for statement in statements:
            build(statement, "1GB")
        return

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(build, statement, CONCURRENT_INDEX_WORK_MEM) for statement in statements]
        for future in futures:
            future.result()  # re-raises whatever a build hit


def tune_load_session(cursor, maintenance_work_mem="1GB"):
    """Widen the session limits that bulk loading and index building run into.

    All session-scoped, so nothing here outlives the load. Losing the tail of a
    load on a crash is fine: these tables are dropped and rebuilt from the
    result files anyway.
    """
    for setting in (
        f"SET maintenance_work_mem = '{maintenance_work_mem}'",  # index builds, default 64MB
        "SET max_parallel_maintenance_workers = 4",
        "SET synchronous_commit = off",
    ):
        try:
            cursor.execute(setting)
        except psycopg2.Error:  # insufficient privileges or unknown on this server
            cursor.connection.rollback()


def render_copy_line(row) -> bytes:
    """Render one validated row as a COPY text line.

    Values are rendered inline rather than through a per-field call: at ~90
    columns this runs several million times per load. Strings are only copied
    when they actually contain a delimiter, which is the bulk of the saving.
    """
    fields = []
    for value in row:
        if type(value) is str:
            fields.append(value.translate(COPY_ESCAPES) if NEEDS_ESCAPE(value) else value)
        elif value is None:
            fields.append("\\N")
        elif value is True:
            fields.append("true")
        elif value is False:
            fields.append("false")
        elif type(value) is list:  # group_id, the only array column
            fields.append("{" + ",".join(map(str, value)) + "}")
        else:
            fields.append(str(value))
    return ("\t".join(fields) + "\n").encode("utf8")


def copy_statement(table_name, field_order) -> str:
    return f"COPY {table_name} ({', '.join(field_order)}) FROM STDIN WITH (FORMAT text)"


def copy_into(cursor, table_name, field_order, rows):
    """Stream validated rows into a table with COPY."""
    cursor.copy_expert(
        copy_statement(table_name, field_order),
        CopyStream(render_copy_line(row) for row in rows),
        size=1 << 20,
    )


class RowValidator:
    """Coerces one result record into the column order a table expects.

    The per-column decisions (which type branch applies, what a missing value
    becomes) depend only on the schema, so they are made once here rather than
    ~90 times per row. Kept byte-for-byte equivalent to the per-row version it
    replaced, including the quirk that the TEXT branch matches the declared type
    case-sensitively while the others do not.
    """

    GROUP_ID, RAW, BOOLEAN, YEAR, TEXT, PASSTHROUGH = range(6)

    def __init__(self, field_names, field_types, groups_file=False):
        self.plan = []
        for field in field_names:
            if field in FILTERED_FIELDS:
                continue
            declared = field_types.get(field, "TEXT")
            declared_upper = declared.upper()
            if field == "group_id":
                # Group files already carry a plain integer; alignment files carry
                # a list, or something that has to be coerced into one. Either way
                # the value is taken as-is, with no missing-value default.
                kind = self.GROUP_ID if groups_file is False else self.RAW
            elif declared_upper == "BOOLEAN":
                kind = self.BOOLEAN
            elif declared_upper == "INTEGER" and not field.endswith("passage_length") and field != "rowid":
                kind = self.YEAR
            elif declared == "TEXT":
                kind = self.TEXT
            else:
                kind = self.PASSTHROUGH
            missing = None if declared_upper in ("INTEGER", "FLOAT", "BOOLEAN", "VECTOR") else ""
            self.plan.append((field, kind, missing))

    def __call__(self, fields) -> list:
        values = []
        for field, kind, missing in self.plan:
            value = fields.get(field)

            if kind == self.GROUP_ID:
                if value is None:
                    value = []
                elif not isinstance(value, list):
                    try:
                        value = [int(value)]
                    except (ValueError, TypeError):
                        value = []
                values.append(value)
                continue
            if kind == self.RAW:
                values.append(value)
                continue

            if value is None:
                value = missing

            if kind == self.TEXT:
                if isinstance(value, str):
                    # Both rewrites allocate, so only pay for them when the
                    # string actually holds something that needs rewriting.
                    if CONTROL_CHAR_FINDER(value):
                        value = value.translate(CONTROL_CHARS)
                    if "<" in value or ">" in value:
                        value = clean_text(value)
            elif kind == self.YEAR:
                if value is not None:
                    value = self.parse_year(value)
            elif kind == self.BOOLEAN:
                if isinstance(value, str):  # older result files wrote the flag as text
                    value = value.strip().lower() in ("true", "t", "1", "yes")
                elif value is not None:
                    value = bool(value)

            values.append(value)
        return values

    @staticmethod
    def parse_year(value):
        """First run of digits in the value, negated when the value opens with it."""
        text = value if isinstance(value, str) else str(value)
        match = YEAR_FINDER.search(text)
        if match is None:
            return None
        digits = match.group(1)
        try:
            number = int(digits)
        except ValueError:  # ints beyond the str-conversion limit
            return None
        return -number if text.startswith("-" + digits) else number


def get_metadata_fields(metadata_file, direction):
    """Get all metadata fields from metadata file"""
    fields: set[str] = set()
    with open(metadata_file, encoding="utf8") as input_file:
        metadata: dict[str, dict[str, str]] = json.load(input_file)
    for values in metadata.values():
        fields.update(f"{direction}_{field}" for field in values.keys())
    for field in FILTERED_FIELDS:
        fields.discard(field)
    return fields


def load_db(
    file,
    source_metadata,
    target_metadata,
    table_name,
    searchable_fields,
    count,
    algorithm,
    banalities_stored,
    textpair_params=None,
):
    """Load SQL table"""
    import numpy as np

    config = read_global_config()
    database_config = {
        "user": config["DATABASE"]["database_user"],
        "password": config["DATABASE"]["database_password"],
        "database": config["DATABASE"]["database_name"],
    }
    workers = load_workers(textpair_params)
    database = psycopg2.connect(**database_config)
    cursor = database.cursor()
    cursor2 = database.cursor()
    tune_load_session(cursor)

    # Register the vector type with psycopg2
    # Note: The vector extension must be created by a superuser before running this
    register_vector(database)

    # Try to load embeddings if they exist
    embeddings_memmap = None
    embeddings_spec = None  # what a worker needs to open the same view
    sbert_dim = None
    alignments_dir = os.path.dirname(file)

    # The cache directory is named after whichever model produced it, which may
    # be [GRAPH] embedding_model rather than [PREPROCESSING]. Try the configured
    # names, then any *_embeddings directory present.
    candidates = []
    if textpair_params is not None:
        graph_params = getattr(textpair_params, "graph_params", None) or {}
        if graph_params.get("embedding_model"):
            candidates.append(graph_params["embedding_model"])
        if hasattr(textpair_params, "preprocessing_params"):
            configured = textpair_params.preprocessing_params.get("source", {}).get("embedding_model")
            if configured:
                candidates.append(configured)
    candidates.append("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    embeddings_cache_path = embeddings_meta_path = None
    for model_name in candidates:
        directory = os.path.join(alignments_dir, f"{model_name.replace('/', '_')}_embeddings")
        if os.path.isdir(directory):
            embeddings_cache_path = os.path.join(directory, "passage_embeddings.dat")
            embeddings_meta_path = os.path.join(directory, "metadata.json")
            break
    if embeddings_cache_path is None:
        found = sorted(glob.glob(os.path.join(alignments_dir, "*_embeddings")))
        directory = found[0] if found else os.path.join(alignments_dir, "_embeddings")
        embeddings_cache_path = os.path.join(directory, "passage_embeddings.dat")
        embeddings_meta_path = os.path.join(directory, "metadata.json")

    if os.path.exists(embeddings_cache_path) and os.path.exists(embeddings_meta_path):
        with open(embeddings_meta_path, "rb") as f:
            metadata = orjson.loads(f.read())
        sbert_dim = metadata["sbert_dim"]
        alignment_counts = metadata["alignment_counts"]

        # The cache is keyed only by model name, so an earlier run's cache sits
        # where this one looks. Row n is alignment n, so a count mismatch means
        # every embedding may be the wrong passage, not just the overflow.
        with lz4.frame.open(file) as input_file:
            file_counts = sum(1 for _ in input_file)
        if file_counts != alignment_counts:
            print(
                f"WARNING: embeddings cache in {os.path.basename(os.path.dirname(embeddings_cache_path))} "
                f"describes {alignment_counts:,} alignments but this file has {file_counts:,}. "
                "It belongs to a different run; loading without embeddings. Re-run the graph "
                "build to regenerate it.",
                file=sys.stderr,
            )
        else:
            embeddings_memmap = np.memmap(
                embeddings_cache_path,
                dtype="float32",
                mode="r",
                shape=(alignment_counts, sbert_dim),
            )
            embeddings_spec = (embeddings_cache_path, alignment_counts, sbert_dim)

    fields_in_table = ["rowid INTEGER PRIMARY KEY"]
    field_names = DEFAULT_FIELDS
    if banalities_stored is True:
        field_names.add("banality")
    if algorithm == "vsa":
        field_names.update(("similarity", "llm_stance", "llm_reasoning", "source_passage_with_matches", "target_passage_with_matches"))
    field_names.update(get_metadata_fields(source_metadata, "source"))
    if target_metadata:
        field_names.update(get_metadata_fields(target_metadata, "target"))
    else:
        field_names.update(get_metadata_fields(source_metadata, "target"))  # add the fields in source as target fields
    field_names.update({"target_first_class", "target_second_class", "target_third_class"})

    # Add embedding column if embeddings are available
    if embeddings_memmap is not None:
        fields_in_table.append(f"embedding vector({sbert_dim})")
        field_names.add("embedding")

    fields_and_types = [
        f"{f} {DEFAULT_FIELD_TYPES.get(f, 'TEXT')}" for f in field_names if f not in ("rowid", "embedding")
    ]
    fields_in_table.extend(fields_and_types)
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    cursor.execute(f"CREATE TABLE {table_name} ({', '.join(fields_in_table)})")

    # RowValidator drops FILTERED_FIELDS from each row, so the column list
    # has to drop them too or the values land in the wrong columns.
    field_order = [f for f in field_names if f not in FILTERED_FIELDS]
    validate = RowValidator(field_order, DEFAULT_FIELD_TYPES)

    def alignment_rows():
        rowid = 0
        for alignment_fields in tqdm(parse_file(file), total=count, leave=False):
            rowid += 1
            yield validate(prepare_alignment(alignment_fields, rowid, embeddings_memmap))

    print("Populating main table...")
    if workers > 1 and count and count >= PARALLEL_LOAD_THRESHOLD:
        database.commit()  # the workers connect separately and must see the table
        parallel_copy_alignments(
            file, table_name, field_order, count, workers, database_config, embeddings_spec
        )
    else:
        copy_into(cursor, table_name, field_order, alignment_rows())

    print("Creating indexes...")
    statements = []
    for field in searchable_fields:
        if field not in field_names:
            continue
        try:
            field_type = DEFAULT_FIELD_TYPES[field].upper()
        except KeyError:
            if field == "source_passage_length" or field == "target_passage_length":
                field_type = "INTEGER"
            else:
                field_type = "TEXT"
        if field_type == "TEXT":
            statements.append(
                f"CREATE INDEX {field}_{table_name}_trigrams_idx ON {table_name} USING GIN({field} gin_trgm_ops)"
            )
            if not field.endswith("passage"):
                statements.append(f"CREATE INDEX {field}_{table_name}_idx ON {table_name} USING HASH({field})")
        elif not field.endswith("year") and field_type in ("INTEGER", "BOOLEAN"):  # year is used for results ordering
            statements.append(f"CREATE INDEX {field}_{table_name}_idx ON {table_name} USING BTREE({field})")
    statements.append(
        f"CREATE INDEX year_{table_name}_idx ON {table_name} USING BTREE(source_year, target_year, source_start_byte)"
    )
    statements.append(f"CREATE INDEX source_start_byte_{table_name}_idx ON {table_name} USING BTREE(source_start_byte)")
    statements.append(f"CREATE INDEX source_end_byte_{table_name}_idx ON {table_name} USING BTREE(source_end_byte)")
    statements.append(f"CREATE INDEX target_start_byte_{table_name}_idx ON {table_name} USING BTREE(target_start_byte)")
    statements.append(f"CREATE INDEX target_end_byte_{table_name}_idx ON {table_name} USING BTREE(target_end_byte)")
    statements.append(f"CREATE INDEX source_doc_id_{table_name}_idx ON {table_name} USING HASH(source_doc_id)")
    statements.append(f"CREATE INDEX target_doc_id_{table_name}_idx ON {table_name} USING HASH(target_doc_id)")
    statements.append(f"CREATE INDEX group_id_{table_name}_idx ON {table_name} USING GIN(group_id)")

    # Result paging is a keyset scan over this order, so the index has to match
    # the ORDER BY in api/text_pair.py exactly, COALESCE included. The sentinel
    # reproduces NULLS LAST while keeping the sort key non-null, which is what
    # lets the cursor be a plain row comparison.
    statements.append(
        f"""CREATE INDEX {table_name}_paging_idx ON {table_name} USING BTREE(
                COALESCE(source_year, 2147483647), COALESCE(target_year, 2147483647),
                source_start_byte, target_start_byte, rowid)"""
    )
    if embeddings_memmap is not None:
        statements.append(
            f"CREATE INDEX {table_name}_embedding_idx ON {table_name} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
        )

    database.commit()  # index builds run on their own connections and must see the rows
    create_indexes(statements, database_config, workers)
    cursor2.execute(f"DROP TABLE IF EXISTS {table_name}_ordered")  # superseded by the paging index
    database.commit()

    # The planner is otherwise left guessing on a table that was empty at CREATE time.
    print("Analyzing table...")
    database.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cursor.execute(f"ANALYZE {table_name}")
    database.close()
    return field_names


def groups_year_index(groups_table: str) -> str:
    """The statement that indexes the groups in the order of the sorted results' "earliest first" sort."""
    return (
        f"CREATE INDEX IF NOT EXISTS year_count_group_id_{groups_table}_idx "
        f'ON {groups_table} USING BTREE(source_year, "count" DESC, group_id)'
    )


def load_groups_file(groups_file: str, alignments_table: str, searchable_fields: list[str], workers: int = 1):
    """Load the groups file into the database."""
    config = read_global_config()
    table_name = f"{alignments_table}_groups"

    with open(groups_file, encoding="utf8") as input_file:
        line = input_file.readline()
        field_names = orjson.loads(line).keys()
        field_names = [f for f in field_names if f not in FILTERED_FIELDS]
        row_count = 1 + sum(1 for _ in input_file)
    fields_in_table = [f"{f} {DEFAULT_FIELD_TYPES.get(f, 'TEXT')}" for f in field_names if f != "group_id"]
    fields_in_table.append("group_id INTEGER PRIMARY KEY")
    searchable_fields = [f for f in searchable_fields if f in field_names and f != "group_id"]

    database_config = {
        "user": config["DATABASE"]["database_user"],
        "password": config["DATABASE"]["database_password"],
        "database": config["DATABASE"]["database_name"],
    }
    database = psycopg2.connect(**database_config)
    cursor = database.cursor()
    tune_load_session(cursor)
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    cursor.execute(f"CREATE TABLE {table_name} ({', '.join(fields_in_table)})")

    print("Populating groups table...")
    validate = RowValidator(field_names, DEFAULT_FIELD_TYPES, groups_file=True)
    with open(groups_file, "rb") as input_file:

        def group_rows():
            for line in tqdm(input_file, total=row_count, desc="Storing alignment groups...", leave=False):
                yield validate(orjson.loads(line))

        copy_into(cursor, table_name, field_names, group_rows())

    statements = []
    for field in searchable_fields:
        try:
            field_type = DEFAULT_FIELD_TYPES[field].upper()
        except KeyError:
            if field == "source_passage_length" or field == "target_passage_length":
                field_type = "INTEGER"
            else:
                field_type = "TEXT"
        if field_type == "TEXT":
            statements.append(
                f"CREATE INDEX {field}_{table_name}_trigrams_idx ON {table_name} USING GIN({field} gin_trgm_ops)"
            )
            if not field.endswith("passage"):
                statements.append(f"CREATE INDEX {field}_{table_name}_idx ON {table_name} USING HASH({field})")
        elif field_type in ("INTEGER", "BOOLEAN"):
            statements.append(f"CREATE INDEX {field}_{table_name}_idx ON {table_name} USING BTREE({field})")
    # The sorted results page through families in this order (see extras/add_groups_index.py)
    statements.append(
        f'CREATE INDEX count_group_id_{table_name}_idx ON {table_name} USING BTREE("count" DESC, group_id)'
    )
    statements.append(f"CREATE INDEX group_id_{table_name}_idx ON {table_name} USING HASH(group_id)")
    if "source_year" in field_names:
        statements.append(groups_year_index(table_name))

    database.commit()  # index builds run on their own connections and must see the rows
    create_indexes(statements, database_config, workers)
    database.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cursor.execute(f"ANALYZE {table_name}")
    database.close()


def network_fields(searchable_fields: list[str], fields_in_table: list[str]) -> list[str]:
    """Metadata fields the network view can aggregate by: present on both sides, not passage or year."""
    fields = []
    for field in searchable_fields:
        if not field.startswith("source_"):
            continue
        name = field[len("source_") :]
        if name in ("passage", "passage_length", "year") or name in fields:
            continue
        if field in fields_in_table and f"target_{name}" in fields_in_table:
            fields.append(name)
    return fields


def load_members_table(alignments_table: str, fields: list[str], workers: int = 1):
    """One row per passage family and document reusing it, for the network view."""
    config = read_global_config()
    table_name = f"{alignments_table}_members"
    database_config = {
        "user": config["DATABASE"]["database_user"],
        "password": config["DATABASE"]["database_password"],
        "database": config["DATABASE"]["database_name"],
    }
    database = psycopg2.connect(**database_config)
    cursor = database.cursor()
    tune_load_session(cursor)
    cursor.execute("SET work_mem = '256MB'")
    print("Storing passage members for the network view...", flush=True)

    def side(direction):
        columns = ", ".join(f"{direction}_{field} AS {field}" for field in fields)
        return (
            f"SELECT x AS group_id, {direction}_doc_id AS doc_id, {direction}_year AS year, {columns} "
            f"FROM {alignments_table}, unnest(group_id) x"
        )

    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    cursor.execute(f"CREATE TABLE {table_name} AS {side('source')} UNION {side('target')}")
    statements = [f"CREATE INDEX group_id_{table_name}_idx ON {table_name} USING BTREE(group_id)"]
    statements.extend(f"CREATE INDEX {field}_{table_name}_idx ON {table_name} USING HASH({field})" for field in fields)
    database.commit()
    create_indexes(statements, database_config, workers)
    database.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cursor.execute(f"ANALYZE {table_name}")
    database.close()


# Alignment-level columns the sorted results can be filtered on without the alignments table
FAMILY_FILTER_ALIGNMENT_FIELDS = ("source_passage_length", "target_passage_length", "banality")


def family_filter_fields(searchable_fields: list[str], fields_in_table: set[str], direction: str) -> list[str]:
    """Searchable fields of one side that belong to the document, so can be matched once per document."""
    return [
        field
        for field in searchable_fields
        if field.startswith(f"{direction}_")
        and field in fields_in_table
        and not field.endswith(("passage", "passage_length"))
    ]


def family_rank_indexes(alignments_table: str, chrono: bool = True) -> list[str]:
    """Statements that index each side's documents, and the alignments, by family, as they are otherwise indexed by document.

    The sorted results read the documents and alignments of given families: of the few that meet an origin
    filter's earlier-side conditions, of the 50 of a page, to show where each was found again, and of those
    a facet counts. With `chrono`, the families are also indexed in the order of their anchor's year (see
    add_chrono_rank).
    """
    statements = [
        f"CREATE INDEX IF NOT EXISTS r_{alignments_table}_{side}_families_idx "
        f"ON {alignments_table}_{side}_families(r) INCLUDE ({side[0]})"
        for side in ("source", "target")
    ]
    statements.append(
        f"CREATE INDEX IF NOT EXISTS r_{alignments_table}_family_alignments_idx ON {alignments_table}_family_alignments(r)"
    )
    if chrono:
        statements.append(
            f"CREATE INDEX IF NOT EXISTS chrono_rank_{alignments_table}_family_rank_idx "
            f"ON {alignments_table}_family_rank(chrono_rank) INCLUDE (rank)"
        )
    return statements


def add_family_anchors(cursor, alignments_table: str) -> bool:
    """Store each family's anchor document on `_family_rank`, as its ids in `_source_meta` and `_target_meta`.

    `anchor_s` is the id the anchor's document has among the documents on the source side of the alignments,
    `anchor_t` among those on the target side; either is NULL where it has none. Safe to repeat. The sorted
    results' origin filters tell the anchor's document from the others by these ids, which is only
    possible if each document has one id, so nothing is stored (False) where one has several.
    The columns appear all at once, but `_family_rank` is locked while they are filled.
    """
    for side in ("source", "target"):
        cursor.execute(f"SELECT COUNT(*) > COUNT(DISTINCT {side}_doc_id) FROM {alignments_table}_{side}_meta")
        if cursor.fetchone()[0]:
            return False
    rank = f"{alignments_table}_family_rank"
    cursor.execute("BEGIN")
    try:
        cursor.execute(f"ALTER TABLE {rank} ADD COLUMN IF NOT EXISTS anchor_s int, ADD COLUMN IF NOT EXISTS anchor_t int")
        cursor.execute(
            f"""UPDATE {rank} f SET anchor_s = s.id, anchor_t = t.id
            FROM {alignments_table}_groups g
            LEFT JOIN {alignments_table}_source_meta s ON s.source_doc_id = g.source_doc_id
            LEFT JOIN {alignments_table}_target_meta t ON t.target_doc_id = g.source_doc_id
            WHERE g.group_id = f.group_id"""
        )
        cursor.execute("COMMIT")
    except psycopg2.Error:
        cursor.execute("ROLLBACK")
        raise
    cursor.execute(f"VACUUM (ANALYZE) {rank}")
    return True


def add_chrono_rank(cursor, alignments_table: str) -> bool:
    """Store each family's place in the order of its anchor's year on `_family_rank`, as `chrono_rank`.

    The order is the anchor's year, undated last, then `"count" DESC, group_id`: that of the groups table's
    year index (groups_year_index). False where the groups have no year. Safe to repeat.
    """
    groups = f"{alignments_table}_groups"
    cursor.execute(
        "SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = current_schema() "
        "AND table_name = %s AND column_name = 'source_year'",
        (groups,),
    )
    if not cursor.fetchone()[0]:
        return False
    rank = f"{alignments_table}_family_rank"
    cursor.execute("BEGIN")
    try:
        cursor.execute(f"ALTER TABLE {rank} ADD COLUMN IF NOT EXISTS chrono_rank int")
        cursor.execute(
            f"""UPDATE {rank} f SET chrono_rank = c.place
            FROM (SELECT group_id, (ROW_NUMBER() OVER (ORDER BY source_year, "count" DESC, group_id))::int AS place
                  FROM {groups}) c
            WHERE c.group_id = f.group_id"""
        )
        cursor.execute("COMMIT")
    except psycopg2.Error:
        cursor.execute("ROLLBACK")
        raise
    cursor.execute(f"VACUUM (ANALYZE) {rank}")
    return True


# The passage length ranges of the facets: the top of each, and the label of the open one last
LENGTH_RANGES = (
    (25, "1-25"),
    (100, "26-100"),
    (250, "101-250"),
    (500, "251-500"),
    (1000, "501-1000"),
    (3000, "1001-3000"),
)
OPEN_LENGTH_RANGE = "3001-"

# The facet tables, in the order they are made: the last marks them complete
FACET_TABLES = ("found_again", "facet_years", "facet_values")


def length_range_number(column: str) -> str:
    """SQL that numbers the range of LENGTH_RANGES a passage length is in, from 0"""
    cases = " ".join(f"WHEN {column} <= {top} THEN {number}" for number, (top, _) in enumerate(LENGTH_RANGES))
    return f"CASE {cases} ELSE {len(LENGTH_RANGES)} END"


def relation_columns(cursor, table: str) -> list[str]:
    """The columns of a table, in order, or none for a table that does not exist"""
    cursor.execute(
        "SELECT attname FROM pg_attribute WHERE attrelid = to_regclass(%s) AND attnum > 0 AND NOT attisdropped "
        "ORDER BY attnum",
        (table,),
    )
    return [name for (name,) in cursor]


def load_facet_tables(cursor, alignments_table: str) -> bool:
    """Tables that count passage families by where they were found again, for the sorted view's facets.

    `_found_again` has a row per family and document it was found again in: one of its documents other
    than the anchor's, at either end of its alignments. `u` is the document's id among the source
    documents, or its negated id among the target ones where no source document has the same id and fields,
    so a search on the later side matches it once. A row holds the family's anchor year (`source_year`, as
    the groups table has it), the ids of the document's values (`v_<field>`, numbered in the order of the
    values) and of the anchor's (`anchor_<field>`), and masks of the length ranges of the alignments it
    ends (`source_lengths`, `target_lengths`, bit n for range n of LENGTH_RANGES). `_facet_years` counts
    each facet's values by the anchor year and `_facet_values` over all the families, with a document
    (`doc`, as `u`) to read each value from: the anchor's for a source_ field, those found again for a
    target_ one, the alignments' ranges for a passage length. Document 0, with no values, stands for an
    anchor in neither documents table. False, and nothing made, where `_family_rank` has no anchors
    (add_family_anchors) or there is nothing to count. Safe to repeat.
    """
    table = alignments_table
    rank, groups, found = f"{table}_family_rank", f"{table}_groups", f"{table}_found_again"
    for name in reversed(FACET_TABLES):
        cursor.execute(f"DROP TABLE IF EXISTS {table}_{name}")
    if not {"anchor_s", "anchor_t"} <= set(relation_columns(cursor, rank)):
        return False
    target_columns = set(relation_columns(cursor, f"{table}_target_meta"))
    names = [
        column[len("source_") :]
        for column in relation_columns(cursor, f"{table}_source_meta")
        if column.startswith("source_") and f"target_{column[len('source_'):]}" in target_columns
    ]
    fields = [name for name in names if name != "doc_id"]
    alignment_columns = relation_columns(cursor, f"{table}_family_alignments")
    sides = [side for side in ("source", "target") if f"{side}_passage_length" in alignment_columns]
    by_year = "source_year" in relation_columns(cursor, groups)
    if not fields and not sides:
        return False

    # A target document with the same id and fields as a source one is that one, so a search matches both or neither
    same = "".join(f" AND s.source_{name} IS NOT DISTINCT FROM t.target_{name}" for name in fields)
    cursor.execute("DROP TABLE IF EXISTS pg_temp.facet_targets, pg_temp.facet_documents, pg_temp.facet_anchors")
    cursor.execute(
        f"""CREATE TEMP TABLE facet_targets AS SELECT t.id AS t, COALESCE(s.id, -t.id) AS u
        FROM {table}_target_meta t LEFT JOIN {table}_source_meta s ON s.source_doc_id = t.target_doc_id{same}"""
    )
    cursor.execute("ALTER TABLE pg_temp.facet_targets ADD PRIMARY KEY (t)")
    numbered = "".join(f", (DENSE_RANK() OVER (ORDER BY {name}))::int AS v_{name}" for name in fields)
    cursor.execute(
        f"""CREATE TEMP TABLE facet_documents AS SELECT u{numbered} FROM (
            SELECT id AS u{"".join(f", source_{name} AS {name}" for name in fields)} FROM {table}_source_meta
            UNION ALL SELECT ft.u{"".join(f", target_{name}" for name in fields)} FROM {table}_target_meta t
            JOIN pg_temp.facet_targets ft ON ft.t = t.id WHERE ft.u < 0
            UNION ALL SELECT 0{", NULL" * len(fields)}) documents"""
    )
    cursor.execute("ALTER TABLE pg_temp.facet_documents ADD PRIMARY KEY (u)")

    # The anchor of each family, as `u`
    cursor.execute(
        f"""CREATE TEMP TABLE facet_anchors AS SELECT f.rank AS r{", g.source_year" if by_year else ""},
        COALESCE(f.anchor_s, ft.u, 0) AS u FROM {rank} f LEFT JOIN pg_temp.facet_targets ft ON ft.t = f.anchor_t
        {f"JOIN {groups} g USING (group_id)" if by_year else ""}"""
    )
    cursor.execute("ALTER TABLE pg_temp.facet_anchors ADD PRIMARY KEY (r)")

    masks = "".join(
        f", (CASE WHEN x.{side}_passage_length IS NULL THEN 0 "
        f"ELSE 1 << {length_range_number(f'x.{side}_passage_length')} END)::smallint AS {side}_lengths"
        for side in sides
    )
    ends = (
        f"SELECT x.r, x.s AS u{masks} FROM {table}_family_alignments x JOIN {rank} f ON f.rank = x.r "
        f"WHERE x.s IS DISTINCT FROM f.anchor_s UNION ALL "
        f"SELECT x.r, ft.u{masks} FROM {table}_family_alignments x JOIN {rank} f ON f.rank = x.r "
        f"JOIN pg_temp.facet_targets ft ON ft.t = x.t WHERE x.t IS DISTINCT FROM f.anchor_t"
    )
    columns = [
        "e.r",
        "e.u",
        *(["a.source_year"] if by_year else []),
        *(f"d.v_{name}" for name in fields),
        *(f"ad.v_{name} AS anchor_{name}" for name in fields),
        *(f"e.{side}_lengths" for side in sides),
    ]
    cursor.execute(
        f"""CREATE TABLE {found} AS SELECT {", ".join(columns)}
        FROM (SELECT r, u{"".join(f", bit_or({side}_lengths) AS {side}_lengths" for side in sides)} FROM ({ends}) ends
              GROUP BY r, u) e
        JOIN pg_temp.facet_documents d USING (u) JOIN pg_temp.facet_anchors a USING (r)
        JOIN pg_temp.facet_documents ad ON ad.u = a.u ORDER BY e.r, e.u"""
    )
    cursor.execute(f"ALTER TABLE {found} ADD PRIMARY KEY (r, u)")
    # ANALYZE samples a tenth of the families and documents, so plans would expect too many rows for a few
    cursor.execute(f"SELECT (SELECT COUNT(DISTINCT r) FROM {found}), (SELECT COUNT(DISTINCT u) FROM {found})")
    distinct_families, distinct_documents = cursor.fetchone()
    cursor.execute(
        f"ALTER TABLE {found} ALTER COLUMN r SET (n_distinct = {max(distinct_families, 1)}), "
        f"ALTER COLUMN u SET (n_distinct = {max(distinct_documents, 1)})"
    )
    # Read by the documents a search on the later side matches, and by the anchor years one on the earlier side does
    kept = ", ".join(column.split()[-1].split(".")[-1] for column in columns if column not in ("e.u", "a.source_year"))
    cursor.execute(f"CREATE INDEX u_{found}_idx ON {found}(u{', source_year' if by_year else ''}) INCLUDE ({kept})")

    # The families anchored in a document with each value, found again in one, or with an alignment in each range
    def counts(year: str) -> str:
        by = f"{year}, " if year else ""
        statements = []
        for name in fields:
            statements.append(
                f"SELECT 'source_{name}'::text AS facet, {by}d.v_{name} AS v, COUNT(*)::int AS n "
                f"FROM pg_temp.facet_anchors a JOIN pg_temp.facet_documents d USING (u) GROUP BY {by}d.v_{name}"
            )
            statements.append(
                f"SELECT 'target_{name}', {by}v_{name}, COUNT(*)::int "
                f"FROM (SELECT DISTINCT r, {by}v_{name} FROM {found}) p GROUP BY {by}v_{name}"
            )
        for side in sides:
            length = f"{side}_passage_length"
            joined = " JOIN pg_temp.facet_anchors a USING (r)" if year else ""
            statements.append(
                f"SELECT '{length}', {by}l.v, COUNT(*)::int FROM (SELECT DISTINCT r, "
                f"{length_range_number(length)} AS v FROM {table}_family_alignments WHERE {length} IS NOT NULL) l"
                f"{joined} GROUP BY {by}l.v"
            )
        return " UNION ALL ".join(statements)

    if by_year:
        years = f"{table}_facet_years"
        cursor.execute(f"CREATE TABLE {years} AS {counts('source_year')}")
        cursor.execute(f"CREATE INDEX facet_{years}_idx ON {years}(facet, source_year) INCLUDE (v, n)")
        cursor.execute(f"VACUUM (ANALYZE) {years}")
        totals = f"SELECT facet, v, SUM(n)::int AS n FROM {years} GROUP BY facet, v"
    else:
        totals = counts("")
    documents = " UNION ALL ".join(
        f"SELECT '{side}_{name}' AS facet, v_{name} AS v, MIN(u) AS doc FROM pg_temp.facet_documents GROUP BY v_{name}"
        for name in fields
        for side in ("source", "target")
    )
    read = f"LEFT JOIN ({documents}) d USING (facet, v)" if fields else "CROSS JOIN (SELECT NULL::int AS doc) d"
    cursor.execute(f"CREATE TABLE {table}_facet_values AS SELECT c.facet, c.v, c.n, d.doc FROM ({totals}) c {read}")
    cursor.execute(f"ALTER TABLE {table}_facet_values ADD PRIMARY KEY (facet, v)")
    cursor.execute(f"VACUUM (ANALYZE) {found}")
    cursor.execute(f"VACUUM (ANALYZE) {table}_facet_values")
    cursor.execute("DROP TABLE pg_temp.facet_targets, pg_temp.facet_documents, pg_temp.facet_anchors")
    return True


def load_family_filter_tables(alignments_table: str, searchable_fields: list[str], workers: int = 1) -> bool:
    """Tables that pick the passage families a search leaves in the running, in the order of the sorted results.

    A family is in the running when any alignment carrying it matches the filters. Alignments are not
    searched for that. Each side's distinct documents are (`_source_meta`, `_target_meta`); `_family_rank`
    puts the families in `"count" DESC, group_id` order, and the other tables hold family ranks:
    `_source_families` and `_target_families` one per family and document, `_family_alignments` one per
    family and alignment, for filters on both sides. `_family_rank` also holds each family's anchor document
    (see add_family_anchors) and its place by the anchor's year (add_chrono_rank). The facet tables count the
    families by where they were found again (load_facet_tables). `_family_filters` is made last, to mark them
    complete.
    """
    config = read_global_config()
    database_config = {
        "user": config["DATABASE"]["database_user"],
        "password": config["DATABASE"]["database_password"],
        "database": config["DATABASE"]["database_name"],
    }
    database = psycopg2.connect(**database_config)
    database.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)  # each step is visible when done
    cursor = database.cursor()
    tune_load_session(cursor)
    cursor.execute("SET work_mem = '1GB'")
    cursor.execute(
        """SELECT column_name, data_type FROM information_schema.columns
        WHERE table_schema = current_schema() AND table_name = %s""",
        (alignments_table,),
    )
    columns = dict(cursor.fetchall())
    if "group_id" not in columns or any(f"{side}_doc_id" not in columns for side in ("source", "target")):
        database.close()
        return False
    print("Storing family filters for the sorted results...", flush=True)

    built = ("family_rank", "source_meta", "target_meta", "family_alignments", "source_families", "target_families")
    for name in ("family_filters", *reversed(FACET_TABLES), *built):
        cursor.execute(f"DROP TABLE IF EXISTS {alignments_table}_{name}")

    fields = {side: family_filter_fields(searchable_fields, set(columns), side) for side in ("source", "target")}
    for side in fields:
        select = ", ".join([f"{side}_doc_id", *fields[side]])
        cursor.execute(
            f"""CREATE TABLE {alignments_table}_{side}_meta AS
            SELECT (ROW_NUMBER() OVER (ORDER BY {side}_doc_id))::int AS id, * FROM
            (SELECT DISTINCT {select} FROM {alignments_table}) documents"""
        )
        cursor.execute(f"ANALYZE {alignments_table}_{side}_meta")
    cursor.execute(
        f"""CREATE TABLE {alignments_table}_family_rank AS
        SELECT (ROW_NUMBER() OVER (ORDER BY "count" DESC, group_id))::int AS rank, group_id
        FROM {alignments_table}_groups"""
    )
    cursor.execute(f"ALTER TABLE {alignments_table}_family_rank ADD PRIMARY KEY (rank)")
    cursor.execute(
        f"CREATE UNIQUE INDEX group_id_{alignments_table}_family_rank_idx ON {alignments_table}_family_rank(group_id)"
    )
    cursor.execute(f"ANALYZE {alignments_table}_family_rank")
    add_family_anchors(cursor, alignments_table)
    has_chrono = add_chrono_rank(cursor, alignments_table)

    # Documents are told apart by their id and fields together, so this holds even if a field varies within one
    joins = ""
    for side in fields:
        same = " AND ".join(f"{side[0]}.{field} IS NOT DISTINCT FROM a.{field}" for field in fields[side])
        joins += f" JOIN {alignments_table}_{side}_meta {side[0]} ON {side[0]}.{side}_doc_id = a.{side}_doc_id"
        joins += f" AND {same}" if same else ""
    if columns["group_id"] == "ARRAY":
        joins += " CROSS JOIN LATERAL unnest(a.group_id) AS g(group_id)"
        family = "g.group_id"
    else:
        family = "a.group_id"
    alignment_fields = [field for field in FAMILY_FILTER_ALIGNMENT_FIELDS if field in columns]
    cursor.execute(
        f"""CREATE TABLE {alignments_table}_family_alignments AS
        SELECT f.rank AS r, s.id AS s, t.id AS t{"".join(f", a.{field}" for field in alignment_fields)}
        FROM {alignments_table} a {joins}
        JOIN {alignments_table}_family_rank f ON f.group_id = {family}
        ORDER BY f.rank"""
    )
    statements = []
    for side in fields:
        column = side[0]
        families = f"{alignments_table}_{side}_families"
        meta = f"{alignments_table}_{side}_meta"
        cursor.execute(
            f"""CREATE TABLE {families} AS
            SELECT r, {column}, (COUNT(*) OVER (PARTITION BY r))::int AS n
            FROM (SELECT DISTINCT r, {column} FROM {alignments_table}_family_alignments) pairs
            ORDER BY r, {column}"""
        )
        cursor.execute(f"ALTER TABLE {meta} ADD COLUMN n int")
        cursor.execute(
            f"""UPDATE {meta} m SET n = COALESCE(d.n, 0)
            FROM (SELECT m2.id, COUNT(f.r) AS n FROM {meta} m2
                  LEFT JOIN {families} f ON f.{column} = m2.id GROUP BY m2.id) d
            WHERE d.id = m.id"""
        )
        statements.append(f"ALTER TABLE {meta} ADD PRIMARY KEY (id)")
        for field in fields[side]:
            if columns[field].upper() == "TEXT":
                statements.append(f"CREATE INDEX {field}_{meta}_trigrams_idx ON {meta} USING GIN({field} gin_trgm_ops)")
            else:
                statements.append(f"CREATE INDEX {field}_{meta}_idx ON {meta} USING BTREE({field})")
        statements.append(f"CREATE UNIQUE INDEX {column}_{families}_idx ON {families}({column}, r) INCLUDE (n)")
        statements.append(
            f"CREATE INDEX {column}_{alignments_table}_family_alignments_idx "
            f"ON {alignments_table}_family_alignments({column}, r)"
        )
    statements.extend(family_rank_indexes(alignments_table, has_chrono))
    for field in alignment_fields:
        if field != "banality":
            statements.append(
                f"CREATE INDEX {field}_{alignments_table}_family_alignments_idx "
                f"ON {alignments_table}_family_alignments({field}, r)"
            )
    create_indexes(statements, database_config, workers)

    # Index-only scans need the visibility map, which only a vacuum fills in
    for name in built:
        cursor.execute(f"VACUUM (ANALYZE) {alignments_table}_{name}")
    load_facet_tables(cursor, alignments_table)
    cursor.execute(
        f"CREATE TABLE {alignments_table}_family_filters AS "
        f"SELECT COUNT(*)::int AS families FROM {alignments_table}_family_rank"
    )
    database.close()
    return True


def set_up_app(web_config, db_path):
    """Copy and build web application with correct configuration"""
    os.system(f"rm -rf {db_path}")
    os.mkdir(db_path)
    print("Building web application...", flush=True)
    os.system(f"cp -R /var/lib/text-pair/web-app/. {db_path}")
    with open(os.path.join(db_path, "appConfig.json"), "w", encoding="utf8") as config_file:
        json.dump({key: value for key, value in web_config().items() if key not in SETTINGS_KEYS}, config_file, indent=4)
    write_app_settings(web_config(), os.path.join(db_path, SETTINGS_FILE))
    os.system(f"""cd {db_path}; npm install --silent; npm run build > "/dev/null" 2>&1;""")


def publish_graph_data(source_graph_data: str, db_dir: str) -> bool:
    """Put a graph_data directory where the API reads it.

    A full replace rather than an overlay: theme count and author mapping
    change between runs, so files from a previous build (an anchor_positions
    array sized for a different number of themes, say) would otherwise linger
    and be read alongside the new ones.
    """
    if not os.path.exists(source_graph_data):
        print(
            f"Note: No graph data found at {source_graph_data}. Clustering visualization will not be available."
        )
        return False
    dest_graph_data = os.path.join(db_dir, "graph_data")
    print(f"Copying graph data to web app directory: {dest_graph_data}")
    if os.path.exists(dest_graph_data):
        shutil.rmtree(dest_graph_data)
    shutil.copytree(source_graph_data, dest_graph_data)
    show_view(db_dir, "semantic_graph")
    return True


def create_web_app(
    file,
    source_metadata,
    target_metadata,
    count,
    table,
    web_app_dir,
    api_server,
    source_philo_db_path,
    target_philo_db_path,
    algorithm,
    textpair_params,
    load_only_db=False,
    groups_file=None,
    store_banalities=False,
):
    """Main routine"""
    web_config = WebAppConfig(
        table,
        api_server,
        source_philo_db_path,
        target_philo_db_path,
        algorithm,
        store_banalities,
        textpair_params.source_against_source,
        textpair_params,
    )

    print("\n### Storing results in database ###", flush=True)
    fields_in_table = load_db(
        file,
        source_metadata,
        target_metadata,
        table,
        web_config.searchable_fields(),
        count,
        algorithm,
        web_config.banalitiesStored,
        textpair_params,
    )
    db_dir = os.path.join(web_app_dir, table)

    if groups_file is not None:
        load_groups_file(
            groups_file,
            table,
            web_config.searchable_fields(),
            load_workers(textpair_params),
        )
        load_members_table(
            table,
            network_fields(web_config.searchable_fields(), fields_in_table),
            load_workers(textpair_params),
        )
        load_family_filter_tables(table, web_config.searchable_fields(), load_workers(textpair_params))

    if load_only_db is False:
        print("\n### Setting up Web Application ###", flush=True)
        web_config.update(fields_in_table)
        set_up_app(web_config, db_dir)

    # Copy the config file used for this run into the web app directory
    config_file = textpair_params.config
    if config_file and os.path.exists(config_file):
        shutil.copy2(config_file, os.path.join(db_dir, f"{table}_config.ini"))

    publish_graph_data(os.path.join(os.path.dirname(file), "graph_data"), db_dir)

    if textpair_params.is_philo_db:
        # Stage db.locals.py and toms.db from the existing PhiloLogic DB into output_path
        # so copy_data can work uniformly regardless of input type.
        for direction, philo_path in (
            ("source", textpair_params.web_app_config["source_philo_db_path"]),
            ("target", textpair_params.web_app_config["target_philo_db_path"]),
        ):
            if textpair_params.source_against_source and direction == "target":
                continue
            dest_dir = os.path.join(textpair_params.output_path, direction)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(os.path.join(philo_path, "data/db.locals.py"), dest_dir)
            shutil.copy2(os.path.join(philo_path, "data/toms.db"), dest_dir)

    if textpair_params.source_against_source is True:
        copy_data(textpair_params, "source")
        os.system(
            f"cd {textpair_params.web_app_config['web_application_directory']}/{textpair_params.dbname}/; rm -rf target_data; ln -s source_data target_data"
        )
    else:
        copy_data(textpair_params, "source")
        copy_data(textpair_params, "target")

    # Remove intermediate source/target directories — data is now in the web app directory
    for direction in ("source", "target"):
        intermediate = os.path.join(textpair_params.output_path, direction)
        if os.path.exists(intermediate):
            shutil.rmtree(intermediate)

    db_url = os.path.join(web_config.apiServer.replace("-api", ""), table)
    print("\n### Finished ###", flush=True)
    print(f"The database is viewable at this URL: {db_url}")
    print(f"To configure the web application, edit {db_dir}/{SETTINGS_FILE}: changes apply when the page is reloaded")


if __name__ == "__main__":
    load_groups_file(
        "/shared/alignments/eccotcp/output/results/passage_group_source.jsonl",
        "ecootcp",
        ["source_author", "source_title", "source_year", "source_passage"],
    )
