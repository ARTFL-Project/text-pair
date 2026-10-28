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

Databases that have the tables from before the sorted results' "first found" tabs get only what those
need: each family's anchor document, as two id columns of `_family_rank` (add_family_anchors in
lib/textpair/web_loader.py), and an index of each documents table by family (family_rank_indexes).
The anchors are one UPDATE of the family table, seconds per million families, during which the
sorted view of that database waits for it.
"""

import os
import sys
import time
from argparse import ArgumentParser
from configparser import ConfigParser

import psycopg2
from textpair.web_loader import add_family_anchors, family_rank_indexes, load_family_filter_tables
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


def has_anchors(cursor, table):
    return has_columns(cursor, f"{table}_family_rank", ("anchor_s", "anchor_t"))


def missing_rank_indexes(cursor, table):
    """The statements of family_rank_indexes for indexes the database does not have yet."""
    missing = []
    for statement in family_rank_indexes(table):
        name = statement.split()[5]
        cursor.execute("SELECT to_regclass(%s) IS NULL", (name,))
        if cursor.fetchone()[0]:
            missing.append(statement)
    return missing


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
            anchors = has_anchors(cursor, table)
            indexes = missing_rank_indexes(cursor, table)
            if anchors and not indexes:
                print(f"  {table}: tables already present")
            elif args.dry_run:
                print(f"  {table}: would add the anchors and indexes to the family filter tables")
            else:
                print(f"  {table}: adding the anchors and indexes...", flush=True)
                started = time.time()
                try:
                    done = anchors or add_family_anchors(cursor, table)
                    for statement in indexes if done else []:
                        cursor.execute(statement)
                    for name in ("source_families", "target_families") if indexes else ():
                        cursor.execute(f"VACUUM (ANALYZE) {table}_{name}")
                except psycopg2.Error as error:
                    print(f"  {table}: FAILED: {str(error).strip()}", file=sys.stderr)
                    continue
                if done:
                    print(f"  {table}: done in {time.time() - started:.1f} s")
                else:
                    print(f"  {table}: a document has several ids, anchors not stored")
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
