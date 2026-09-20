"""Adds the keyset paging index to TextPAIR tables loaded before it existed.

Result paging used to join against a `{table}_ordered` rank table. It now runs
as a keyset scan over the main table, which needs an index matching the ORDER BY
in api/text_pair.py. Without it paging still returns correct results, but sorts
the whole table on every page.

Databases loaded from now on get the index during the load; this is only for
existing ones. Safe to re-run, and `--drop-ordered` reclaims the `_ordered`
tables once the new API is serving.
"""

import sys
from argparse import ArgumentParser
from configparser import ConfigParser

import psycopg2

GLOBAL_CONFIG = ConfigParser()
GLOBAL_CONFIG.read("/etc/text-pair/global_settings.ini")

# Must stay identical to PAGE_SORT_EXPRESSIONS in api/text_pair.py.
PAGING_INDEX = """CREATE INDEX {index} ON {table} USING BTREE(
    COALESCE(source_year, 2147483647), COALESCE(target_year, 2147483647),
    source_start_byte, target_start_byte, rowid)"""

SORT_COLUMNS = ("source_year", "target_year", "source_start_byte", "target_start_byte", "rowid")


def alignment_tables(cursor):
    """Every table carrying the full sort key, minus the auxiliary ones."""
    cursor.execute(
        """SELECT table_name FROM information_schema.columns
           WHERE table_schema='public' AND column_name = ANY(%s)
           GROUP BY table_name HAVING count(DISTINCT column_name) = %s
           ORDER BY table_name""",
        (list(SORT_COLUMNS), len(SORT_COLUMNS)),
    )
    return [t for (t,) in cursor if not t.endswith(("_ordered", "_groups"))]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--tables", nargs="*", help="only these tables (default: all alignment tables)")
    parser.add_argument("--drop-ordered", action="store_true", help="also drop the superseded _ordered tables")
    parser.add_argument("--dry-run", action="store_true", help="report what would change and exit")
    args = parser.parse_args()

    connection = psycopg2.connect(
        database=GLOBAL_CONFIG.get("DATABASE", "database_name"),
        user=GLOBAL_CONFIG.get("DATABASE", "database_user"),
        password=GLOBAL_CONFIG.get("DATABASE", "database_password"),
    )
    connection.autocommit = True
    cursor = connection.cursor()

    tables = args.tables or alignment_tables(cursor)
    print(f"{len(tables)} alignment table(s) to consider\n")

    for table in tables:
        index = f"{table}_paging_idx"
        cursor.execute("SELECT 1 FROM pg_indexes WHERE tablename=%s AND indexname=%s", (table, index))
        if cursor.fetchone():
            print(f"  {table}: index already present")
        elif args.dry_run:
            print(f"  {table}: would create {index}")
        else:
            print(f"  {table}: creating {index}...", end="", flush=True)
            cursor.execute("SET maintenance_work_mem = '1GB'")
            cursor.execute("SET max_parallel_maintenance_workers = 4")
            try:
                cursor.execute(PAGING_INDEX.format(index=index, table=table))
                print(" done")
            except psycopg2.Error as error:
                print(f" FAILED: {str(error).strip()}", file=sys.stderr)
                continue

        if args.drop_ordered:
            cursor.execute("SELECT 1 FROM information_schema.tables WHERE table_name=%s", (f"{table}_ordered",))
            if cursor.fetchone():
                if args.dry_run:
                    print(f"  {table}: would drop {table}_ordered")
                else:
                    cursor.execute(f"DROP TABLE {table}_ordered")
                    print(f"  {table}: dropped {table}_ordered")

    connection.close()


if __name__ == "__main__":
    main()
