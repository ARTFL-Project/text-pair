"""Backs up existing TextPAIR database (a table in PostGreSQL), along with web config and web files to a tarball file."""

import os
import shutil
from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path

import psycopg2

GLOBAL_CONFIG = ConfigParser()
GLOBAL_CONFIG.read("/etc/text-pair/global_settings.ini")


import psycopg2


def table_exists(user, password, table_name):
    conn = psycopg2.connect(database=GLOBAL_CONFIG.get("DATABASE", "database_name"), user=user, password=password)
    with conn.cursor() as cursor:
        cursor.execute("SELECT 1 FROM information_schema.tables WHERE table_name=%s", (table_name,))
        result = cursor.fetchone()
    conn.close()
    return result if result else None


def extract_textpair_database(table, web_app_path, output_path):
    db_name = GLOBAL_CONFIG.get("DATABASE", "database_name")
    db_user = GLOBAL_CONFIG.get("DATABASE", "database_user")
    db_password = GLOBAL_CONFIG.get("DATABASE", "database_password")

    # Create a temporary directory for organizing files
    temp_dir = Path(output_path) / f"textpair_{table}_temp"
    temp_dir.mkdir(exist_ok=True)

    # Create the backup directory that will be the root in the tarball
    backup_dir = temp_dir / f"{table}_textpair_backup"
    backup_dir.mkdir(exist_ok=True)

    # Copy web app contents to backup directory using original directory name
    web_app_name = Path(web_app_path).name
    web_app_dest = backup_dir / web_app_name
    shutil.copytree(web_app_path, web_app_dest, dirs_exist_ok=True)

    # Dump database tables to backup directory
    existing_tables = [
        table_name
        for table_name in [table, f"{table}_groups", f"{table}_ordered"]
        if table_exists(db_user, db_password, table_name)
    ]

    sql_files = []
    for table_name in existing_tables:
        sql_file = f"textpair_{table_name}.sql"
        sql_path = backup_dir / sql_file
        os.system(f"PGPASSWORD={db_password} pg_dump -U {db_user} {db_name} -t {table_name} > {sql_path}")
        sql_files.append(sql_file)

    # Create tarball from temp directory
    tar_path = Path(output_path) / f"textpair_{table}.tar.gz"
    current_dir = os.getcwd()
    try:
        os.chdir(temp_dir)
        os.system(f"tar -czf {tar_path} {table}_textpair_backup")
    finally:
        os.chdir(current_dir)

    # Clean up
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--db_name", type=str, required=True, help="Name of the database to backup.")
    parser.add_argument("--web_app_path", type=str, default="", help="Path to the output tarball file.")
    parser.add_argument("--output_path", type=str, default="", help="Path to the output tarball file.")
    args = parser.parse_args()

    output_path = args.output_path or os.getcwd()
    web_app_path = args.web_app_path or os.path.join(GLOBAL_CONFIG.get("WEB_APP", "web_app_path"), args.db_name)
    web_app_path = web_app_path.rstrip("/")
    extract_textpair_database(args.db_name, web_app_path, output_path)
