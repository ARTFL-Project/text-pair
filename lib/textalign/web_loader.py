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
from textalign import parse_config
from tqdm import tqdm

try:
    import psycopg2
except ImportError:
    print("The textalign lib was not installed with the web components. Please \
    run pip3 install .[web] from the lib/ directory to install missing dependencies \
    or run the texalign command with --disable_web_app")

DEFAULT_FIELD_TYPES = {
    "source_year": "INTEGER", "source_pub_date": "INTEGER", "target_year": "INTEGER", "target_pub_date": "INTEGER",
    "source_start_byte": "INTEGER", "target_start_byte": "INTEGER", "source_end_byte": "INTEGER", "target_end_byte": "INTEGER",
    "source_passage_length": "INTEGER", "target_passage_length": "INTEGER"
}

YEAR_FINDER = re.compile(r'^.*?(\d{1,}).*')
TOKENIZER = re.compile(r"\w+")


class WebAppConfig:
    """ Web app config class"""

    def __init__(self, field_types, db_name, api_server, source_database,
                 source_database_link, target_database, target_database_link):
        with open("/var/lib/text-align/config/appConfig.json") as app_config:
            self.options = json.load(app_config, object_pairs_hook=OrderedDict)
        for field, field_type in field_types.items():
            self.options["metadataTypes"][field] = field_type
        self.options["apiServer"] = api_server
        self.options["appPath"] = os.path.join("text-align", db_name)
        self.options["databaseName"] = db_name
        self.options["sourceDB"] = OrderedDict([("philoDB", source_database), ("link", source_database_link)])
        self.options["targetDB"] = OrderedDict([("philoDB", target_database), ("link", target_database_link)])

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


def parse_command_line(args):
    """Command line parsing function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="configuration file used to override defaults",
                        type=str, default="")
    parser.add_argument("--file", help="alignment file to load", type=str, default=None)
    args = vars(parser.parse_args(args=args))
    if args["file"] is None:
        print("Please supply a file argument\nExiting....")
        exit()
    field_types = DEFAULT_FIELD_TYPES
    if args["config"]:
        if os.path.exists(args["config"]):
            _, _, _, web_app_config = parse_config(args["config"])
            field_types.update(web_app_config["field_types"])
    return args["file"], web_app_config["table_name"], field_types, web_app_config["web_application_directory"], \
           web_app_config["api_server"], web_app_config["source_database"], web_app_config["source_database_link"], \
           web_app_config["target_database"], web_app_config["target_database_link"]

def count_lines(filename):
    """Count lines in file"""
    return sum(1 for _ in open(filename, 'rbU'))

def parse_file(file):
    """Parse tab delimited file and insert into table"""
    with open(file, encoding="utf8", errors="ignore") as input_file:
        for line in input_file:
            fields = json.loads(line.rstrip("\n"))
            yield fields

def clean_text(text):
    """Clean passages for HTML viewing before storing"""
    text = text.replace("<", "&lt;")
    text = text.replace(">", "gt;")
    return text

def validate_field_type(fields, field_types, field_names):
    """Check field type and modify value type if needed"""
    values = []
    for field in field_names:
        value = fields.get(field, "")
        field_type = field_types.get(field, "TEXT")
        if field_type.upper() == "INTEGER" and not field.endswith("passage_length") and field != "rowid":
            year_match = YEAR_FINDER.search(value)
            if year_match:
                value = int(year_match.groups()[0])
            else:
                value = None
        if field_type == "TEXT":
            value = clean_text(value)
        values.append(value)
    return values

def load_db(file, table_name, field_types, searchable_fields):
    """Load SQL table"""
    config = configparser.ConfigParser()
    config.read("/etc/text-align/global_settings.ini")
    database = psycopg2.connect(user=config["DATABASE"]["database_user"],
                                password=config["DATABASE"]["database_password"],
                                database=config["DATABASE"]["database_name"])
    cursor = database.cursor()
    cursor2 = database.cursor()
    line_count = count_lines(file) - 1 # skip first line with field names
    alignments = parse_file(file)
    fields_in_table = ["rowid INTEGER PRIMARY KEY"]
    field_names = ["rowid"]
    with open(file, errors="ignore") as input_file:
        extra_fields = json.loads(input_file.readline().rstrip("\n")).keys() # TODO: we need to add fields on the fly as they occur, since not all are in the first line
        field_names.extend(extra_fields)
        field_names.extend(["source_passage_length", "target_passage_length"])
        if "source_year" not in field_names:
            field_names.append("source_year")
        if "target_year" not in field_names:
            field_names.append("target_year")
        fields_and_types = ["{} {}".format(f, field_types.get(f, "TEXT")) for f in field_names if f != "rowid"]
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
        elif not field.endswith("year") and field_type == "INTEGER": # year is a special case used for results ordering
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
        rows.append((rowid, row[0],))
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
    cursor2.execute("CREATE INDEX {}_source_year_target_year_rowid_idx ON {} USING BTREE(rowid_ordered)".format(ordered_table, ordered_table))
    database.commit()
    database.close()
    return field_names

def set_up_app(web_config, db_path):
    """Copy and build web application with correct configuration"""
    print("Copying and building web application...")
    os.system("rm -rf {}".format(db_path))
    os.mkdir(db_path)
    os.system("cp -R /var/lib/text-align/web/web_app/. {}".format(db_path))
    with open(os.path.join(db_path, "appConfig.json"), "w") as config_file:
        json.dump(web_config(), config_file, indent=4)
    os.system("cd {}; npm install --silent; npm run build;".format(db_path))
    if web_config.webServer == "Apache":
        os.system("cp /var/lib/text-align/web/apache_htaccess.conf {}".format(os.path.join(db_path, ".htaccess")))

def create_web_app(file, table, field_types, web_app_dir, api_server, source_database,
                   source_database_link, target_database, target_database_link):
    """Main routine"""
    print("\n### Building Web Application ###", flush=True)
    web_config = WebAppConfig(field_types, table, api_server, source_database,
                              source_database_link, target_database, target_database_link)
    fields_in_table = load_db(file, table, field_types, web_config.searchable_fields())
    web_config.update(fields_in_table)
    set_up_app(web_config, os.path.join("{}/{}/".format(web_app_dir, table)))
    print("DB viewable at {}".format(os.path.join(web_config.apiServer.replace("-api", ""), table)))

def load_from_cli():
    """Called from textalign script"""
    main(sys.argv[2:])

def main(args):
    """Main function"""
    file, table, field_types, web_app_dir, api_server, source_database, \
    source_database_link, target_database, target_database_link = parse_command_line(args)
    create_web_app(file, table, field_types, web_app_dir, api_server,
                   source_database, source_database_link, target_database,
                   target_database_link)

if __name__ == '__main__':
    main(sys.argv[1:])
