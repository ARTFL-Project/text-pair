"""Adds the family filter tables to TextPAIR databases loaded before they existed.

The sorted results view lists passage families, the most reused first, and when the search form
filters it, only those with an alignment that matches. The loader now builds tables that answer that
from indexes (see load_family_filter_tables in lib/textpair/web_loader.py); without them the API
searches the alignments table on every page. This builds them for databases loaded earlier.

Safe to re-run, and safe while the API is serving: nothing existing is touched, the API ignores the
tables until `_family_filters`, made last, is there, and a database without them is served as before.
A database is skipped if it already has them (--force rebuilds), or has no groups table, no web app
directory to read its searchable fields from, or no document ids. Run it with the textpair
environment's Python, and expect minutes per million alignments plus the disk they take (a few
percent of the alignments table).
"""

import os
import sys
import time
from argparse import ArgumentParser
from configparser import ConfigParser

import psycopg2
from textpair.web_loader import load_family_filter_tables
from textpair.web_settings import load_app_config

GLOBAL_CONFIG = ConfigParser()
GLOBAL_CONFIG.read("/etc/text-pair/global_settings.ini")


def groups_tables(cursor):
    """The alignments table of every `_groups` table carrying the columns the filter tables are built from."""
    cursor.execute(
        """SELECT table_name FROM information_schema.columns
           WHERE table_schema = current_schema() AND table_name LIKE '%\\_groups'
           AND column_name IN ('count', 'group_id')
           GROUP BY table_name HAVING count(*) = 2
           ORDER BY table_name"""
    )
    return [name[: -len("_groups")] for (name,) in cursor]


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


def searchable_fields(table):
    """The fields of the table's search form, or None without a web app directory to read them from."""
    db_dir = os.path.join(GLOBAL_CONFIG.get("WEB_APP", "web_app_path"), table)
    if not os.path.exists(os.path.join(db_dir, "appConfig.json")):
        return None
    fields = load_app_config(db_dir)["metadataFields"]
    return [field["value"] for direction in ("source", "target") for field in fields[direction]]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--tables", nargs="*", help="only these alignments tables (default: all with a groups table)")
    parser.add_argument("--dry-run", action="store_true", help="report what would change and exit")
    parser.add_argument("--force", action="store_true", help="rebuild the tables of databases that have them")
    parser.add_argument("--workers", type=int, default=4, help="index builds to run at once (default 4)")
    args = parser.parse_args()

    connection = psycopg2.connect(
        database=GLOBAL_CONFIG.get("DATABASE", "database_name"),
        user=GLOBAL_CONFIG.get("DATABASE", "database_user"),
        password=GLOBAL_CONFIG.get("DATABASE", "database_password"),
    )
    connection.autocommit = True
    cursor = connection.cursor()

    tables = args.tables or groups_tables(cursor)
    print(f"{len(tables)} table(s) to consider\n")

    for table in tables:
        if not table_exists(cursor, f"{table}_groups") or not has_columns(
            cursor, table, ("group_id", "source_doc_id", "target_doc_id")
        ):
            print(f"  {table}: no groups table or document ids, skipped")
            continue
        if table_exists(cursor, f"{table}_family_filters") and not args.force:
            print(f"  {table}: tables already present")
            continue
        fields = searchable_fields(table)
        if fields is None:
            print(f"  {table}: no web app directory, skipped")
            continue
        if args.dry_run:
            print(f"  {table}: would build the family filter tables")
            continue
        print(f"  {table}: building...", flush=True)
        started = time.time()
        try:
            load_family_filter_tables(table, fields, args.workers)
            print(f"  {table}: done in {time.time() - started:.0f} s")
        except psycopg2.Error as error:
            print(f"  {table}: FAILED: {str(error).strip()}", file=sys.stderr)


if __name__ == "__main__":
    main()
