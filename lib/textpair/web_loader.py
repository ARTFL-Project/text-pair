#!/usr/bin/env python3
"""Web loading module"""

import configparser
import json
import os
import re
from collections import OrderedDict
from typing import Any

import lz4.frame
import orjson
from psycopg2.extras import execute_values
from tqdm import tqdm

try:
    import psycopg2
except ImportError:
    print(
        "The textpair lib was not installed with the web components. Please \
    run pip3 install .[web] from the lib/ directory to install missing dependencies \
    or run the textpair command with --disable_web_app"
    )

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
    "group_id": "INTEGER",
    "count": "INTEGER",
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


class WebAppConfig:
    """Web app config class"""

    def __init__(self, db_name, api_server, source_database_link, target_database_link, algorithm, store_banalities):
        with open("/var/lib/text-pair/config/appConfig.json", encoding="utf8") as app_config:
            self.options = json.load(app_config, object_pairs_hook=OrderedDict)
        self.options["apiServer"] = api_server
        self.options["appPath"] = os.path.join("/text-pair", db_name)
        self.options["databaseName"] = db_name.lower()
        self.options["matchingAlgorithm"] = algorithm
        self.options["sourcePhiloDBLink"] = source_database_link
        self.options["targetPhiloDBLink"] = target_database_link
        self.options["banalitiesStored"] = store_banalities

    def __call__(self):
        return self.options

    def __getattr__(self, attr):
        return self.options[attr]

    def searchable_fields(self):
        """Return list of all searchable fields"""
        fields = []
        for field in self.options["metadataFields"]["source"]:
            fields.append(field["value"])
        for field in self.options["metadataFields"]["target"]:
            fields.append(field["value"])
        return fields

    def update(self, available_fields):
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


def validate_field_type(fields, field_types, field_names):
    """Check field type and modify value type if needed"""
    values = []
    for field in field_names:
        if field in FILTERED_FIELDS:
            continue
        value = fields.get(field, "")
        field_type = field_types.get(field, "TEXT")
        if field_type.upper() == "INTEGER" and not field.endswith("passage_length") and field != "rowid":
            if isinstance(value, int):
                value = str(value)
            year_match = YEAR_FINDER.search(value)
            if year_match:
                matching_year = year_match.groups()[0]
                neg_match = re.search(rf"^(\-{matching_year})", value)  # account for negative years
                if neg_match:
                    value = int(neg_match.groups()[0])
                else:
                    value = int(year_match.groups()[0])
            else:
                value = None
        if field_type == "TEXT" and isinstance(value, str):
            value = value.translate(CONTROL_CHARS)
            value = clean_text(value)
        values.append(value)
    return values


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


def load_db(file, source_metadata, target_metadata, table_name, searchable_fields, count, algorithm, banalities_stored):
    """Load SQL table"""
    config = configparser.ConfigParser()
    config.read("/etc/text-pair/global_settings.ini")
    database = psycopg2.connect(
        user=config["DATABASE"]["database_user"],
        password=config["DATABASE"]["database_password"],
        database=config["DATABASE"]["database_name"],
    )
    cursor = database.cursor()
    cursor2 = database.cursor()

    fields_in_table = ["rowid INTEGER PRIMARY KEY"]
    field_names = DEFAULT_FIELDS
    if banalities_stored is True:
        field_names.add("banality")
    if algorithm == "vsa":
        field_names.update(("similarity", "source_passage_with_matches", "target_passage_with_matches"))
    field_names.update(get_metadata_fields(source_metadata, "source"))
    if target_metadata:
        field_names.update(get_metadata_fields(target_metadata, "target"))
    else:
        field_names.update(get_metadata_fields(source_metadata, "target"))  # add the fields in source as target fields
    fields_and_types = [f"{f} {DEFAULT_FIELD_TYPES.get(f, 'TEXT')}" for f in field_names if f != "rowid"]
    fields_in_table.extend(fields_and_types)
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    cursor.execute(f"CREATE TABLE {table_name} ({', '.join(fields_in_table)})")
    lines = 0
    rows = []
    rowid = 0
    alignments = parse_file(file)
    print("Populating main table...")
    for alignment_fields in tqdm(alignments, total=count, leave=False):
        rowid += 1
        alignment_fields["rowid"] = rowid
        alignment_fields["passage_id"] = rowid
        alignment_fields["source_passage_length"] = len(TOKENIZER.findall(alignment_fields["source_passage"]))
        alignment_fields["target_passage_length"] = len(TOKENIZER.findall(alignment_fields["target_passage"]))
        row = validate_field_type(alignment_fields, DEFAULT_FIELD_TYPES, field_names)
        rows.append(row)
        lines += 1
        if lines == 100:
            insert = f"INSERT INTO {table_name} ({', '.join(field_names)}) VALUES %s"
            execute_values(cursor, insert, rows)
            rows = []
            lines = 0
    if lines:
        insert = f"INSERT INTO {table_name} ({', '.join(field_names)}) VALUES %s"
        execute_values(cursor, insert, rows)
        rows = []
        lines = 0

    print("Creating indexes for all searchable fields...")
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
            cursor.execute(
                f"CREATE INDEX {field}_{table_name}_trigrams_idx ON {table_name} USING GIN({field} gin_trgm_ops)"
            )
            if not field.endswith("passage"):
                cursor.execute(f"CREATE INDEX {field}_{table_name}_idx ON {table_name} USING HASH({field})")
        elif not field.endswith("year") and field_type == "INTEGER":  # year is a special case used for results ordering
            cursor.execute(f"CREATE INDEX {field}_{table_name}_idx ON {table_name} USING BTREE({field})")
    cursor.execute(
        f"CREATE INDEX year_{table_name}_idx ON {table_name} USING BTREE(source_year, target_year, source_start_byte)"
    )
    cursor.execute(f"CREATE INDEX source_doc_id_{table_name}_idx ON {table_name} USING HASH(source_doc_id)")
    cursor.execute(f"CREATE INDEX target_doc_id_{table_name}_idx ON {table_name} USING HASH(target_doc_id)")
    cursor.execute(f"CREATE INDEX group_id_{table_name}_idx ON {table_name} USING HASH(group_id)")
    database.commit()

    print("Populating index table...")
    ordered_table = table_name + "_ordered"
    cursor2.execute(f"DROP TABLE if exists {ordered_table}")
    cursor2.execute(
        f"CREATE TABLE {ordered_table} (rowid_ordered INTEGER PRIMARY KEY, source_year_target_year INTEGER)"
    )
    cursor.execute(
        f"SELECT rowid FROM {table_name} ORDER BY source_year, target_year, source_start_byte, target_start_byte ASC"
    )
    lines = 0
    rows = []
    rowid = 0
    for row in tqdm(cursor, total=count, leave=False):
        lines += 1
        rowid += 1
        rows.append((rowid, row[0]))
        if lines == 100:
            insert = f"INSERT INTO {ordered_table} (rowid_ordered, source_year_target_year) VALUES %s"
            execute_values(cursor2, insert, rows)
            rows = []
            lines = 0
    if lines:
        insert = f"INSERT INTO {ordered_table} (rowid_ordered, source_year_target_year) VALUES %s"
        execute_values(cursor2, insert, rows)
        rows = []
        lines = 0
    print("Creating indexes...")
    cursor2.execute(
        f"CREATE INDEX {ordered_table}_source_year_target_year_rowid_idx ON {ordered_table} USING BTREE(rowid_ordered)"
    )
    database.commit()
    database.close()
    return field_names


def load_groups_file(groups_file: str, alignments_table: str, searchable_fields: list[str]):
    """Load the groups file into the database."""
    config = configparser.ConfigParser()
    config.read("/etc/text-pair/global_settings.ini")
    table_name = f"{alignments_table}_groups"

    with open(groups_file, encoding="utf8") as input_file:
        line = input_file.readline()
        field_names = orjson.loads(line).keys()
        field_names = [f for f in field_names if f not in FILTERED_FIELDS]
        row_count = 1 + sum(1 for _ in input_file)
    fields_in_table = [f"{f} {DEFAULT_FIELD_TYPES.get(f, 'TEXT')}" for f in field_names if f != "group_id"]
    fields_in_table.append("group_id INTEGER PRIMARY KEY")
    searchable_fields = [f for f in searchable_fields if f in field_names and f != "group_id"]

    database = psycopg2.connect(
        user=config["DATABASE"]["database_user"],
        password=config["DATABASE"]["database_password"],
        database=config["DATABASE"]["database_name"],
    )
    cursor = database.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    cursor.execute(f"CREATE TABLE {table_name} ({', '.join(fields_in_table)})")

    print("Populating groups table...")
    with open(groups_file, encoding="utf8") as input_file:
        for line in tqdm(input_file, total=row_count, desc="Storing alignment groups...", leave=False):
            group = orjson.loads(line)
            row = validate_field_type(group, DEFAULT_FIELD_TYPES, field_names)
            insert = (
                f"INSERT INTO {table_name} ({', '.join(field_names)}) VALUES (" + ", ".join("%s" for _ in row) + ")"
            )
            cursor.execute(insert, row)

    for field in searchable_fields:
        try:
            field_type = DEFAULT_FIELD_TYPES[field].upper()
        except KeyError:
            if field == "source_passage_length" or field == "target_passage_length":
                field_type = "INTEGER"
            else:
                field_type = "TEXT"
        if field_type == "TEXT":
            cursor.execute(
                f"CREATE INDEX {field}_{table_name}_trigrams_idx ON {table_name} USING GIN({field} gin_trgm_ops)"
            )
            if not field.endswith("passage"):
                cursor.execute(f"CREATE INDEX {field}_{table_name}_idx ON {table_name} USING HASH({field})")
        elif field_type == "INTEGER":
            cursor.execute(f"CREATE INDEX {field}_{table_name}_idx ON {table_name} USING BTREE({field})")
    cursor.execute(f"CREATE INDEX count_{table_name}_idx ON {table_name} USING BTREE(count)")
    cursor.execute(f"CREATE INDEX group_id_{table_name}_idx ON {table_name} USING HASH(group_id)")
    database.commit()
    database.close()

def generate_database_stats(table_name, algorithm):
    """Generate statistics for the database"""
    print("Generating database statistics (this could take a while)...")
    config = configparser.ConfigParser()
    config.read("/etc/text-pair/global_settings.ini")
    database = psycopg2.connect(
        user=config["DATABASE"]["database_user"],
        password=config["DATABASE"]["database_password"],
        database=config["DATABASE"]["database_name"],
    )
    cursor = database.cursor()
    stats = {}
    if algorithm == "sa":
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}_groups")
        stats["group_count"] = cursor.fetchone()[0]
        cursor.execute(f"SELECT COUNT(DISTINCT source_author) FROM {table_name}_groups")
        stats["author_group_count"] = cursor.fetchone()[0]
        cursor.execute(f"SELECT COUNT(DISTINCT source_title) FROM {table_name}_groups")
        stats["title_group_count"] = cursor.fetchone()[0]
    else:
        # TODO: Add stats for other algorithms
        pass
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    stats["pairs_count"] = cursor.fetchone()[0]
    return stats

def set_up_app(web_config, db_path):
    """Copy and build web application with correct configuration"""
    os.system(f"rm -rf {db_path}")
    os.mkdir(db_path)
    os.system(f"cp -R /var/lib/text-pair/web-app/. {db_path}")
    with open(os.path.join(db_path, "appConfig.json"), "w", encoding="utf8") as config_file:
        json.dump(web_config(), config_file, indent=4)
    os.system(f"""cd {db_path}; npm install --silent > "/dev/null" 2>&1; npm run build > "/dev/null" 2>&1;""")


def create_web_app(
    file,
    source_metadata,
    target_metadata,
    count,
    table,
    web_app_dir,
    api_server,
    source_database_link,
    target_database_link,
    algorithm,
    load_only_db=False,
    groups_file=None,
    store_banalities=False,
):
    """Main routine"""
    web_config = WebAppConfig(
        table, api_server, source_database_link, target_database_link, algorithm, store_banalities
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
    )
    db_dir = os.path.join(web_app_dir, table)
    if groups_file is not None:
        load_groups_file(
            groups_file,
            table,
            web_config.searchable_fields(),
        )
    if load_only_db is False:
        print("\n### Setting up Web Application ###", flush=True)
        web_config.update(fields_in_table)
        print("Building web application...", flush=True)
        set_up_app(web_config, db_dir)
        db_url = os.path.join(web_config.apiServer.replace("-api", ""), table)
        print("\n### Finished ###", flush=True)
        print(f"The database is viewable at this URL: {db_url}")
        print(
            f"To configure the web application, edit {db_dir}/appConfig.json and run 'npm run build' from the {db_dir} directory"
        )
    stats = generate_database_stats(table, algorithm)
    with open(os.path.join(db_dir, "stats.json"), "w", encoding="utf8") as stats_file:
        json.dump(stats, stats_file)

if __name__ == "__main__":
    load_groups_file(
        "/shared/alignments/eccotcp/output/results/passage_group_source.jsonl",
        "ecootcp",
        ["source_author", "source_title", "source_year", "source_passage"],
    )
