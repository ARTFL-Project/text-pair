"""Adds the landing page's counts of documents and authors to TextPAIR databases loaded before they were stored.

Where alignments form passage families, the landing page shows how many documents the alignments span,
counted by file as a family's `count` is, and how many authors wrote them. Counting them scans the
alignments table, so the loader stores them in `_collection_counts` (load_collection_counts in
lib/textpair/web_loader.py); this stores them for databases loaded earlier, whose landing page shows the
other figures until then.

Safe to re-run, and safe while the API is serving: the table is made by one statement, and the API reads
it only once it is there. A database is skipped if it already has it (--force counts again, and its page
lacks the two figures while it does), or has no passage families. Run it with the textpair environment's
Python; each database takes one scan of its alignments table.
"""

import sys
import time
from argparse import ArgumentParser
from configparser import ConfigParser

import psycopg2
from textpair.web_loader import load_collection_counts

GLOBAL_CONFIG = ConfigParser()
GLOBAL_CONFIG.read("/etc/text-pair/global_settings.ini")


def family_tables(cursor):
    """Every alignments table whose alignments form passage families: a group_id array and a groups table."""
    cursor.execute(
        """SELECT c.table_name FROM information_schema.columns c
           WHERE c.table_schema = current_schema() AND c.column_name = 'group_id' AND c.data_type = 'ARRAY'
           AND to_regclass(quote_ident(c.table_name || '_groups')) IS NOT NULL
           ORDER BY c.table_name"""
    )
    return [name for (name,) in cursor]


def has_columns(cursor, table, columns):
    cursor.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = %s",
        (table,),
    )
    found = {name for (name,) in cursor}
    return all(column in found for column in columns)


def table_exists(cursor, table):
    cursor.execute("SELECT to_regclass(%s) IS NOT NULL", (table,))
    return cursor.fetchone()[0]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--tables", nargs="*", help="only these alignments tables (default: all with passage families)")
    parser.add_argument("--dry-run", action="store_true", help="report what would change and exit")
    parser.add_argument("--force", action="store_true", help="count again for databases that have the counts")
    args = parser.parse_args()

    connection = psycopg2.connect(
        database=GLOBAL_CONFIG.get("DATABASE", "database_name"),
        user=GLOBAL_CONFIG.get("DATABASE", "database_user"),
        password=GLOBAL_CONFIG.get("DATABASE", "database_password"),
    )
    connection.autocommit = True
    cursor = connection.cursor()

    families = family_tables(cursor)
    tables = args.tables or families
    print(f"{len(tables)} table(s) to consider\n")

    for table in tables:
        if table not in families or not has_columns(cursor, table, ("source_filename", "target_filename")):
            print(f"  {table}: no passage families or file names, skipped")
            continue
        if table_exists(cursor, f"{table}_collection_counts") and not args.force:
            print(f"  {table}: counts already present")
            continue
        if args.dry_run:
            print(f"  {table}: would count its documents and authors")
            continue
        started = time.time()
        try:
            load_collection_counts(table)
        except psycopg2.Error as error:
            print(f"  {table}: FAILED: {str(error).strip()}", file=sys.stderr)
            continue
        print(f"  {table}: done in {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
