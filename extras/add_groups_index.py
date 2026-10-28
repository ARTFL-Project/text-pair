"""Adds the sorted-results paging index to the `_groups` tables of TextPAIR databases loaded before it existed.

The sorted results view pages through passage families by `ORDER BY "count" DESC, group_id`. With an
index in that order Postgres reads each page straight off the index; without it, deep pages are much
slower. The loader now builds the index with the table, so this is only for existing databases. Safe
to re-run, and safe while the API is serving: the index is built CONCURRENTLY, so reads and writes
carry on.

The single-column `count_{table}_idx` that older loads made is covered by the new index and can be
dropped once it is in place.
"""

import sys
from argparse import ArgumentParser
from configparser import ConfigParser

import psycopg2

GLOBAL_CONFIG = ConfigParser()
GLOBAL_CONFIG.read("/etc/text-pair/global_settings.ini")

# Must stay identical to the statement in load_groups_file (lib/textpair/web_loader.py).
SORT_INDEX = 'CREATE INDEX CONCURRENTLY IF NOT EXISTS {index} ON {table} USING BTREE("count" DESC, group_id)'


def groups_tables(cursor):
    """Every `_groups` table carrying the columns the index is on."""
    cursor.execute(
        """SELECT table_name FROM information_schema.columns
           WHERE table_schema='public' AND table_name LIKE '%\\_groups' AND column_name IN ('count', 'group_id')
           GROUP BY table_name HAVING count(*) = 2
           ORDER BY table_name"""
    )
    return [table for (table,) in cursor]


def index_state(cursor, index):
    """None if the index does not exist, else whether it is valid (a failed build leaves an invalid one)."""
    cursor.execute(
        "SELECT i.indisvalid FROM pg_class c JOIN pg_index i ON i.indexrelid = c.oid WHERE c.relname = %s", (index,)
    )
    row = cursor.fetchone()
    return None if row is None else row[0]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--tables", nargs="*", help="only these tables (default: every groups table)")
    parser.add_argument("--dry-run", action="store_true", help="report what would change and exit")
    args = parser.parse_args()

    connection = psycopg2.connect(
        database=GLOBAL_CONFIG.get("DATABASE", "database_name"),
        user=GLOBAL_CONFIG.get("DATABASE", "database_user"),
        password=GLOBAL_CONFIG.get("DATABASE", "database_password"),
    )
    connection.autocommit = True  # CREATE INDEX CONCURRENTLY cannot run in a transaction
    cursor = connection.cursor()

    tables = args.tables or groups_tables(cursor)
    print(f"{len(tables)} groups table(s) to consider\n")

    for table in tables:
        index = f"count_group_id_{table}_idx"[:63]  # Postgres truncates longer names
        state = index_state(cursor, index)
        if state:
            print(f"  {table}: index already present")
            continue
        if args.dry_run:
            print(f"  {table}: would create {index}")
            continue
        print(f"  {table}: creating {index}...", end="", flush=True)
        try:
            if state is False:  # left half-built by an interrupted run
                cursor.execute(f"DROP INDEX CONCURRENTLY {index}")
            cursor.execute("SET maintenance_work_mem = '1GB'")
            cursor.execute(SORT_INDEX.format(index=index, table=table))
            print(" done")
        except psycopg2.Error as error:
            print(f" FAILED: {str(error).strip()}", file=sys.stderr)

    connection.close()


if __name__ == "__main__":
    main()
