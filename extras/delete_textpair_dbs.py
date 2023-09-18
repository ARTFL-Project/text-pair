"""Delete TextPAIR database by name."""

import os
import sys
import configparser

import psycopg2


GLOBAL_CONFIG = configparser.ConfigParser()
GLOBAL_CONFIG.read("/etc/text-pair/global_settings.ini")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 delete_textpair_dbs.py <dbname>')
        sys.exit(1)

    dbname = sys.argv[1]

    conn = psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    )
    conn.autocommit = True
    cursor = conn.cursor()

    print(f"Dropping table {dbname}...", end="")
    cursor.execute(f"DROP TABLE IF EXISTS {dbname}")
    print("done")
    print(f"Dropping table {dbname}__ordered...", end="")
    cursor.execute(f"DROP TABLE IF EXISTS {dbname}__ordered")
    print("done")
    print(f"Dropping table {dbname}__groups...", end="")
    cursor.execute(f"DROP TABLE IF EXISTS {dbname}___groups")
    print("done")
    print(f"Deleting {dbname} web app directory...", end="")
    os.system(f"rm -rf {GLOBAL_CONFIG['WEB_APP']['web_app_path']}/{dbname}")
    print("done")

    print(f"Deleted database {dbname}")
    conn.close()