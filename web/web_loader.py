#!/usr/bin/env python3

import sys

import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm

DATABASE = psycopg2.connect(user="alignments", password="martini", database="alignments")
CURSOR = DATABASE.cursor()
CURSOR2 = DATABASE.cursor()


def count_lines(filename):
    """Count lines in file"""
    return sum(1 for _ in open(filename, 'rbU'))

def parse_file(file):
    """Parse tab delimited file and insert into table"""
    with open(file, errors="ignore") as input_file:
        for pos, line in enumerate(input_file):
            if pos < 2:
                continue
            fields = line
            yield fields

def main():
    line_count = count_lines(sys.argv[1]) - 1 # skip first line with field names
    alignments = parse_file(sys.argv[1])
    table_name = sys.argv[2]
    fields_in_table = ["rowid INTEGER PRIMARY KEY"]
    field_names = ["rowid"]
    with open(sys.argv[1], errors="ignore") as input_file:
        field_names.extend(input_file.readline().rstrip().split("\t"))
        fields_in_table.extend([i + " TEXT" for i in field_names if i != "rowid"])
    CURSOR.execute("DROP TABLE if exists {}".format(table_name))
    CURSOR.execute("CREATE TABLE {} ({})".format(table_name, ", ".join(fields_in_table)))
    lines = 0
    rows = []
    rowid = 0
    print("Populating main table...")
    for alignment_fields in tqdm(alignments, total=line_count):
        lines += 1
        rowid += 1
        rows.append([rowid] + alignment_fields.split('\t'))
        if lines == 100:
            insert = "INSERT INTO {} ({}) VALUES %s".format(table_name, ", ".join(field_names))
            execute_values(CURSOR, insert, rows)
            rows = []
            lines = 0
    print("Creating indexes...")
    CURSOR.execute("CREATE INDEX {}_source_author_index ON {} USING BTREE(source_author)".format(table_name, table_name))
    CURSOR.execute("CREATE INDEX {}_target_author_index ON {} USING BTREE(target_author)".format(table_name, table_name))
    CURSOR.execute("CREATE INDEX {}_source_title_index ON {} USING BTREE(source_title)".format(table_name, table_name))
    CURSOR.execute("CREATE INDEX {}_target_title_index ON {} USING BTREE(target_title)".format(table_name, table_name))
    CURSOR.execute("CREATE INDEX {}_year_index ON {} USING BTREE(source_year, target_year)".format(table_name, table_name))
    DATABASE.commit()

    print("Populating index table...")
    ordered_table = table_name + "_ordered"
    CURSOR2.execute("DROP TABLE if exists {}".format(ordered_table))
    CURSOR2.execute("CREATE TABLE {} ({})".format(ordered_table, "rowid_ordered INTEGER PRIMARY KEY, source_year_target_year INTEGER"))
    CURSOR.execute("SELECT rowid FROM {} ORDER BY source_year, target_year ASC".format(table_name))
    lines = 0
    rows = []
    rowid = 0
    for row in tqdm(CURSOR, total=line_count):
        lines += 1
        rowid += 1
        rows.append((rowid, row[0],))
        if lines == 100:
            insert = "INSERT INTO {} ({}) VALUES %s".format(ordered_table, "rowid_ordered, source_year_target_year")
            execute_values(CURSOR2, insert, rows)
            rows = []
            lines = 0
    print("Creating indexes...")
    CURSOR2.execute("CREATE INDEX {}_source_year_target_year_rowid_index ON {} USING BTREE(rowid_ordered)".format(ordered_table, ordered_table))
    DATABASE.commit()
    DATABASE.close()

    print("DB viewable at http://root_url_for_alignment_dbs/{}".format(table_name))
    print("Configure database at http://root_url_for_alignment_dbs/{}_config.json".format(table_name))

if __name__ == '__main__':
    main()
