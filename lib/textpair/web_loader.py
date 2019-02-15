#!/usr/bin/env python3
"""Web loading module"""

import argparse
import configparser
import json
import os
import re
import sys
from collections import OrderedDict

from psycopg2.extras import execute_values
from textpair import parse_config
from tqdm import tqdm

try:
    import psycopg2
except ImportError:
    print(
        "The textpair lib was not installed with the web components. Please \
    run pip3 install .[web] from the lib/ directory to install missing dependencies \
    or run the textpair command with --disable_web_app"
    )

DEFAULT_FIELD_TYPES = {
    "source_year": "INTEGER",
    "source_pub_date": "INTEGER",
    "target_year": "INTEGER",
    "target_pub_date": "INTEGER",
    "source_start_byte": "INTEGER",
    "target_start_byte": "INTEGER",
    "source_end_byte": "INTEGER",
    "target_end_byte": "INTEGER",
    "source_passage_length": "INTEGER",
    "target_passage_length": "INTEGER",
    "similarity": "FLOAT",
}

FILTERED_FIELDS = {
    "source_doc_id",
    "source_philo_seq",
    "source_parent",
    "source_prev",
    "source_next",
    "source_parent",
    "source_philo_name",
    "source_philo_type",
    "source_word_count",
    "target_doc_id",
    "target_philo_seq",
    "target_parent",
    "target_prev",
    "target_next",
    "target_parent",
    "target_philo_name",
    "target_philo_type",
}

YEAR_FINDER = re.compile(r"^.*?(\d{1,}).*")
TOKENIZER = re.compile(r"\w+")


class WebAppConfig:
    """ Web app config class"""

    def __init__(self, field_types, db_name, api_server, source_database_link, target_database_link, algorithm):
        with open("/var/lib/text-pair/config/appConfig.json") as app_config:
            self.options = json.load(app_config, object_pairs_hook=OrderedDict)
        self.options["apiServer"] = api_server
        self.options["appPath"] = os.path.join("text-pair", db_name)
        self.options["databaseName"] = db_name
        self.options["matchingAlgorithm"] = algorithm
        self.options["sourcePhiloDBLink"] = source_database_link
        self.options["targetPhiloDBLink"] = target_database_link

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


def count_lines(filename):
    """Count lines in file"""
    return sum(1 for _ in open(filename, "rbU"))


def parse_file(file):
    """Parse tab delimited file and insert into table"""
    with open(file, encoding="utf8", errors="ignore") as input_file:
        for line in input_file:
            fields = json.loads(line.rstrip("\n"))
            yield fields


def clean_text(text):
    """Clean passages for HTML viewing before storing"""
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


def validate_field_type(fields, field_types, field_names):
    """Check field type and modify value type if needed"""
    values = []
    for field in field_names:
        value = fields.get(field, "")
        field_type = field_types.get(field, "TEXT")
        if field_type.upper() == "INTEGER" and not field.endswith("passage_length") and field != "rowid":
            if isinstance(value, int):
                value = str(value)
            year_match = YEAR_FINDER.search(value)
            if year_match:
                value = int(year_match.groups()[0])
            else:
                value = None
        if field_type == "TEXT" and isinstance(value, str):
            value = clean_text(value)
        values.append(value)
    return values


def load_db(file, table_name, field_types, searchable_fields):
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
    line_count = count_lines(file) - 1  # skip first line with field names
    alignments = parse_file(file)
    fields_in_table = ["rowid INTEGER PRIMARY KEY"]
    field_names = ["rowid"]
    with open(file, errors="ignore") as input_file:
        extra_fields = json.loads(
            input_file.readline().rstrip("\n")
        ).keys()  # TODO: we need to add fields on the fly as they occur, since not all are in the first line
        field_names.extend(extra_fields)
        field_names.extend(["source_passage_length", "target_passage_length"])
        if "source_year" not in field_names:
            field_names.append("source_year")
        if "target_year" not in field_names:
            field_names.append("target_year")
        fields_and_types = ["{} {}".format(f, field_types.get(f, "TEXT")) for f in field_names if f != "rowid" and f not in FILTERED_FIELDS]
        fields_in_table.extend(fields_and_types)
    cursor.execute("DROP TABLE IF EXISTS {}".format(table_name))
    cursor.execute("CREATE TABLE {} ({})".format(table_name, ", ".join(fields_in_table)))
    lines = 0
    rows = []
    rowid = 0
    print("Populating main table...")
    for alignment_fields in tqdm(alignments, total=line_count, leave=False):
        rowid += 1
        alignment_fields["rowid"] = rowid
        alignment_fields["source_passage_length"] = len(TOKENIZER.findall(alignment_fields["source_passage"]))
        alignment_fields["target_passage_length"] = len(TOKENIZER.findall(alignment_fields["target_passage"]))
        row = validate_field_type(alignment_fields, field_types, field_names)
        rows.append(row)
        lines += 1
        if lines == 100:
            insert = "INSERT INTO {} ({}) VALUES %s".format(table_name, ", ".join(field_names))
            execute_values(cursor, insert, rows)
            rows = []
            lines = 0
    if lines:
        insert = "INSERT INTO {} ({}) VALUES %s".format(table_name, ", ".join(field_names))
        execute_values(cursor, insert, rows)
        rows = []
        lines = 0

    print("Creating indexes for all searchable fields...")
    for field in searchable_fields:
        if field not in field_names:
            continue
        try:
            field_type = field_types[field].upper()
        except KeyError:
            if field == "source_passage_length" or field == "target_passage_length":
                field_type = "INTEGER"
            else:
                field_type = "TEXT"
        if field_type == "TEXT":
            cursor.execute("CREATE INDEX {}_{}_trigrams_idx ON {} USING GIN({} gin_trgm_ops)".format(field, table_name, table_name, field))
            if not field.endswith("passage"):
                cursor.execute("CREATE INDEX {}_{}_idx ON {} USING HASH({})".format(field, table_name, table_name, field))
        elif not field.endswith("year") and field_type == "INTEGER":  # year is a special case used for results ordering
            cursor.execute("CREATE INDEX {}_{}_idx ON {} USING BTREE({})".format(field, table_name, table_name, field))
    cursor.execute("CREATE INDEX year_{}_idx ON {} USING BTREE(source_year, target_year, source_start_byte)".format(table_name, table_name))
    database.commit()

    print("Populating index table...")
    ordered_table = table_name + "_ordered"
    cursor2.execute("DROP TABLE if exists {}".format(ordered_table))
    cursor2.execute("CREATE TABLE {} ({})".format(ordered_table, "rowid_ordered INTEGER PRIMARY KEY, source_year_target_year INTEGER"))
    cursor.execute("SELECT rowid FROM {} ORDER BY source_year, target_year, source_start_byte, target_start_byte ASC".format(table_name))
    lines = 0
    rows = []
    rowid = 0
    for row in tqdm(cursor, total=line_count, leave=False):
        lines += 1
        rowid += 1
        rows.append((rowid, row[0]))
        if lines == 100:
            insert = "INSERT INTO {} ({}) VALUES %s".format(ordered_table, "rowid_ordered, source_year_target_year")
            execute_values(cursor2, insert, rows)
            rows = []
            lines = 0
    if lines:
        insert = "INSERT INTO {} ({}) VALUES %s".format(ordered_table, "rowid_ordered, source_year_target_year")
        execute_values(cursor2, insert, rows)
        rows = []
        lines = 0
    print("Creating indexes...")
    cursor2.execute(
        "CREATE INDEX {}_source_year_target_year_rowid_idx ON {} USING BTREE(rowid_ordered)".format(ordered_table, ordered_table)
    )
    database.commit()
    database.close()
    return field_names


def set_up_app(web_config, db_path):
    """Copy and build web application with correct configuration"""
    os.system("rm -rf {}".format(db_path))
    os.mkdir(db_path)
    os.system("cp -R /var/lib/text-pair/web/web_app/. {}".format(db_path))
    with open(os.path.join(db_path, "appConfig.json"), "w") as config_file:
        json.dump(web_config(), config_file, indent=4)
    os.system(f"""cd {db_path}; npm install --silent > "/dev/null" 2>&1; npm run build > "/dev/null" 2>&1;""")
    if web_config.webServer.lower() == "apache":
        os.system(f"""cp /var/lib/text-pair/web/apache_htaccess.conf {os.path.join(db_path, ".htaccess")}""")


def create_web_app(file, table, field_types, web_app_dir, api_server, source_database_link, target_database_link, algorithm):
    """Main routine"""
    web_config = WebAppConfig(field_types, table, api_server, source_database_link, target_database_link, algorithm)
    print("\n### Storing results in database ###", flush=True)
    fields_in_table = load_db(file, table, field_types, web_config.searchable_fields())
    print("\n### Setting up Web Application ###", flush=True)
    web_config.update(fields_in_table)
    db_dir = os.path.join(web_app_dir, table)
    print("Building web application...", flush=True)
    set_up_app(web_config, db_dir)
    db_url = os.path.join(web_config.apiServer.replace("-api", ""), table)
    print("\n### Finished ###", flush=True)
    print(f"The database is viewable at this URL: {db_url}")
    print(f"To configure the web application, edit {db_dir}/appConfig.json and run 'npm run build' from the {db_dir} directory")

