#!/usr/bin/env python3
"""Web loading module"""

import argparse
import configparser
import os
import json
from collections import OrderedDict

import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm


DEFAULT_FIELD_TYPES = {"source_year": "INTEGER", "source_pub_date": "INTEGER", "target_year": "INTEGER", "target_pub_date": "INTEGER"}


class WebAppConfig:
    """ Web app config class"""

    def __init__(self, field_types, db_name, api_server):
        with open("/var/lib/text-align/config/appConfig.json") as app_config:
            self.options = json.load(app_config, object_pairs_hook=OrderedDict)
        for field, field_type in field_types.items():
            self.options["metadataTypes"][field] = field_type
        self.options["apiServer"] = api_server
        self.options["appPath"] = os.path.join("text-align", db_name)
        self.options["databaseName"] = db_name

    def __call__(self):
        return self.options

    def __getattr__(self, attr):
        return self.options[attr]


def parse_command_line():
    """Command line parsing function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="configuration file used to override defaults",
                        type=str, default="")
    parser.add_argument("--file", help="alignment file to load", type=str, default=None)
    args = vars(parser.parse_args())
    if args["file"] is None:
        print("Please supply a file argument\nExiting....")
        exit()
    if args["table"] is None:
        print("Please supply a table argument\nExiting....")
        exit()
    field_types = DEFAULT_FIELD_TYPES
    api_server = ""
    table = ""
    web_application_directory = ""
    if args["config"]:
        if os.path.exists(args["config"]):
            config = configparser.ConfigParser()
            config.read(args["config"])
            for key, value in dict(config["WEB_APPLICATION"]).items():
                if key == "api_server":
                    api_server = value
                elif key == "table_name":
                    table = value
                elif "web_application_directory":
                    web_application_directory = value
                else:
                    field_types[key] = value
    return args["file"], table, field_types, web_application_directory, api_server

def count_lines(filename):
    """Count lines in file"""
    return sum(1 for _ in open(filename, 'rbU'))

def parse_file(file):
    """Parse tab delimited file and insert into table"""
    with open(file, errors="ignore") as input_file:
        for pos, line in enumerate(input_file):
            if pos < 2:
                continue
            fields = line.rstrip("\n")
            yield fields

def validate_field_type(row, field_types):
    """Check field type and modify value type if needed"""
    values = []
    for field, value in row:
        field_type = field_types.get(field, "TEXT")
        if field_type == "INTEGER":
            try:
                value = int(value)
            except ValueError:
                value = 0
        values.append(value)
    return values

def load_db(file, table_name, field_types):
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
        field_names.extend(input_file.readline().rstrip("\n").split("\t"))
        fields_and_types = ["{} {}".format(f, field_types.get(f, "TEXT")) for f in field_names if f != "rowid"]
        fields_in_table.extend(fields_and_types)
    cursor.execute("DROP TABLE IF EXISTS {}".format(table_name))
    cursor.execute("CREATE TABLE {} ({})".format(table_name, ", ".join(fields_in_table)))
    lines = 0
    rows = []
    rowid = 0
    skipped = 0
    field_num = len(fields_in_table)-1
    print("Populating main table...")
    for alignment_fields in tqdm(alignments, total=line_count):
        row = zip(field_names[1:], alignment_fields.split("\t"))
        row = validate_field_type(row, field_types)
        if len(row) != field_num:
            skipped += 1
            continue
        lines += 1
        rowid += 1
        rows.append([rowid] + row)
        if lines == 100:
            insert = "INSERT INTO {} ({}) VALUES %s".format(table_name, ", ".join(field_names))
            execute_values(cursor, insert, rows)
            rows = []
            lines = 0
    print("Creating indexes...")
    cursor.execute("CREATE INDEX {}_source_author_index ON {} USING BTREE(source_author)".format(table_name, table_name))
    cursor.execute("CREATE INDEX {}_target_author_index ON {} USING BTREE(target_author)".format(table_name, table_name))
    cursor.execute("CREATE INDEX {}_source_title_index ON {} USING BTREE(source_title)".format(table_name, table_name))
    cursor.execute("CREATE INDEX {}_target_title_index ON {} USING BTREE(target_title)".format(table_name, table_name))
    cursor.execute("CREATE INDEX {}_year_index ON {} USING BTREE(source_year, target_year)".format(table_name, table_name))
    database.commit()

    if skipped != 0:
        print("{} rows were skipped due to mismatch".format(skipped))

    print("Populating index table...")
    ordered_table = table_name + "_ordered"
    cursor2.execute("DROP TABLE if exists {}".format(ordered_table))
    cursor2.execute("CREATE TABLE {} ({})".format(ordered_table, "rowid_ordered INTEGER PRIMARY KEY, source_year_target_year INTEGER"))
    cursor.execute("SELECT rowid FROM {} ORDER BY source_year, target_year ASC".format(table_name))
    lines = 0
    rows = []
    rowid = 0
    for row in tqdm(cursor, total=line_count):
        lines += 1
        rowid += 1
        rows.append((rowid, row[0],))
        if lines == 100:
            insert = "INSERT INTO {} ({}) VALUES %s".format(ordered_table, "rowid_ordered, source_year_target_year")
            execute_values(cursor2, insert, rows)
            rows = []
            lines = 0
    print("Creating indexes...")
    cursor2.execute("CREATE INDEX {}_source_year_target_year_rowid_index ON {} USING BTREE(rowid_ordered)".format(ordered_table, ordered_table))
    database.commit()
    database.close()

def set_up_app(web_config, db_path):
    """Copy and build web application with correct configuration"""
    print("Copying and building web application...")
    os.system("rm -rf {}".format(db_path))
    os.mkdir(db_path)
    os.system("cp -R /var/lib/text-align/web/web_app/. {}".format(db_path))
    with open(os.path.join(db_path, "appConfig.json"), "w") as config_file:
        json.dump(web_config(), config_file)
    os.system("cd {}; npm run build;".format(db_path))
    if web_config.webServer == "Apache":
        os.system("cp /var/lib/text-align/web/apache_htaccess.conf {}".format(os.path.join(db_path, ".htaccess")))

def create_web_app(file, table, field_types, web_app_dir, api_server):
    """Main routine"""
    web_config = WebAppConfig(field_types, table, api_server)
    load_db(file, table, field_types)
    set_up_app(web_config, os.path.join("{}/{}/".format(web_app_dir, table)))
    print("DB viewable at {}/{}".format(web_config.apiServer.replace("-api", ""), table))

def main():
    """Main function"""
    file, table, field_types, web_app_dir, api_server = parse_command_line()
    create_web_app(file, table, field_types, web_app_dir, api_server)

if __name__ == '__main__':
    main()
