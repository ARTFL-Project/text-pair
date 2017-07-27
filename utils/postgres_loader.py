#!/usr/bin/env python3

import sys

import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm

DATABASE = psycopg2.connect(user="alignments", password="martini", database="alignments")
CURSOR = DATABASE.cursor()


def count_lines(filename):
    """Count lines in file"""
    return sum(1 for _ in open(filename, 'rbU'))

def parse_file(file):
    """Parse tab delimited file and insert into table"""
    with open(file, errors="ignore") as input_file:
        for pos, line in enumerate(input_file):
            if pos < 2:
                continue
            fields = line.rstrip()
            yield fields

if __name__ == '__main__':
    line_count = count_lines(sys.argv[1]) - 1 # skip first line with field names
    alignments = parse_file(sys.argv[1])
    table_name = sys.argv[2]
    with open(sys.argv[1]) as input_file:
        field_names = input_file.readline().rstrip().split("\t")
    CURSOR.execute("DROP TABLE if exists {}".format(table_name))
    CURSOR.execute("CREATE TABLE {} ({})".format(table_name, ", ".join([i + " text" for i in field_names])))
    lines = 0
    rows = []
    for alignment_fields in tqdm(alignments, total=line_count):
        lines += 1

        rows.append(alignment_fields.split('\t'))
        if lines == 100:
            insert = "INSERT INTO {} ({}) VALUES %s".format(table_name, ", ".join(field_names))
            execute_values(CURSOR, insert, rows)
            rows = []
            lines = 0
    CURSOR.execute("CREATE INDEX {}_source_author_index ON {} USING BTREE(source_author)".format(table_name, table_name))
    CURSOR.execute("CREATE INDEX {}_target_author_index ON {} USING BTREE(target_author)".format(table_name, table_name))
    CURSOR.execute("CREATE INDEX {}_source_title_index ON {} USING BTREE(source_title)".format(table_name, table_name))
    CURSOR.execute("CREATE INDEX {}_target_title_index ON {} USING BTREE(target_title)".format(table_name, table_name))
    DATABASE.commit()
    DATABASE.close()
