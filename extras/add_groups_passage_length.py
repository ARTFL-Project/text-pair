"""Adds `source_passage_length`, the word count of each family's own passage, to the `_groups` tables of TextPAIR
databases loaded before load_groups_file stored it.

The sorted results filter and count the earlier text's passage length by it, the passage the family cards show;
without it they fall back to the lengths of the family's alignments. Counted as the loader counts it (TOKENIZER),
on the passage as stored, less the loader's escaping of < and >. Safe to re-run, and safe while the API is
serving: the rows are updated in one transaction, which reads do not wait for.
"""

import io
import sys
from argparse import ArgumentParser
from configparser import ConfigParser

import psycopg2

from textpair.web_loader import TOKENIZER, groups_length_index

GLOBAL_CONFIG = ConfigParser()
GLOBAL_CONFIG.read("/etc/text-pair/global_settings.ini")
COLUMN = "source_passage_length"


def groups_tables(cursor):
    """Every `_groups` table with a source passage and no length for it."""
    cursor.execute(
        """SELECT table_name FROM information_schema.columns
           WHERE table_schema='public' AND table_name LIKE '%%\\_groups' AND column_name IN ('source_passage', %s)
           GROUP BY table_name HAVING bool_and(column_name = 'source_passage') ORDER BY table_name""",
        (COLUMN,),
    )
    return [table for (table,) in cursor]


def lengths(connection, table):
    """group_id and word count of every family, as COPY text."""
    out = io.StringIO()
    with connection.cursor(name="passages") as cursor:
        cursor.itersize = 20_000
        cursor.execute(f"SELECT group_id, source_passage FROM {table}")
        for group_id, passage in cursor:
            passage = (passage or "").replace("&lt;", "<").replace("&gt;", ">")
            out.write(f"{group_id}\t{len(TOKENIZER.findall(passage))}\n")
    out.seek(0)
    return out


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--tables", nargs="*", help="only these groups tables (default: every one without the column)")
    parser.add_argument("--dry-run", action="store_true", help="report what would change and exit")
    args = parser.parse_args()

    connection = psycopg2.connect(
        database=GLOBAL_CONFIG.get("DATABASE", "database_name"),
        user=GLOBAL_CONFIG.get("DATABASE", "database_user"),
        password=GLOBAL_CONFIG.get("DATABASE", "database_password"),
    )
    cursor = connection.cursor()
    tables = args.tables or groups_tables(cursor)
    print(f"{len(tables)} groups table(s) to consider\n")

    for table in tables:
        if args.dry_run:
            print(f"  {table}: would add {COLUMN}")
            continue
        print(f"  {table}: counting...", end="", flush=True)
        try:
            counted = lengths(connection, table)
            cursor.execute("CREATE TEMP TABLE passage_lengths (group_id integer PRIMARY KEY, n integer)")
            cursor.copy_expert("COPY passage_lengths FROM STDIN", counted)
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {COLUMN} INTEGER")
            cursor.execute(
                f"UPDATE {table} g SET {COLUMN} = l.n FROM passage_lengths l "
                f"WHERE l.group_id = g.group_id AND g.{COLUMN} IS DISTINCT FROM l.n"
            )
            updated = cursor.rowcount
            cursor.execute("DROP TABLE passage_lengths")
            connection.commit()
            print(f" {updated:,} rows;", end="", flush=True)
            connection.autocommit = True
            # The loader's name for the index of a searchable integer field
            index = f"{COLUMN}_{table}_idx"[:63]
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {index} ON {table} USING BTREE({COLUMN})")
            cursor.execute(groups_length_index(table))
            cursor.execute(f"VACUUM ANALYZE {table}")
            connection.autocommit = False
            print(" indexed")
        except psycopg2.Error as error:
            connection.rollback()
            connection.autocommit = False
            print(f" FAILED: {str(error).strip()}", file=sys.stderr)

    connection.close()


if __name__ == "__main__":
    main()
