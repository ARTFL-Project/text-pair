"""Backs up existing TextPAIR database (a table in PostGreSQL), along with web config and web files to a tarball file."""

import json
import os
import shutil
from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path

import lz4.frame
import psycopg2

GLOBAL_CONFIG = ConfigParser()
GLOBAL_CONFIG.read("/etc/text-pair/global_settings.ini")


def table_exists(user, password, table_name):
    conn = psycopg2.connect(database=GLOBAL_CONFIG.get("DATABASE", "database_name"), user=user, password=password)
    with conn.cursor() as cursor:
        cursor.execute("SELECT 1 FROM information_schema.tables WHERE table_name=%s", (table_name,))
        result = cursor.fetchone()
    conn.close()
    return result if result else None


def back_up_philo_db_data(philo_db_path, output_path):
    """Backs up the source data for a PhiloLogic database."""
    print(f"  - Backing up PhiloLogic database from {philo_db_path}")

    # Copy text files
    text_path = output_path / "TEXT"
    text_path.mkdir()
    print("    - Copying TEXT files...")
    for file in os.scandir(philo_db_path / 'data/TEXT/'):
        shutil.copy(file.path, text_path)

    # Copy db related files:
    print("    - Copying database files...")
    shutil.copy(philo_db_path / 'data/toms.db', output_path)
    shutil.copy(philo_db_path / 'data/db.locals.py', output_path)
    print("    ✓ PhiloLogic database backup complete")


def extract_textpair_database(table, web_app_path, output_path):
    print(f"\nStarting TextPAIR backup for database: {table}")
    print(f"Web app path: {web_app_path}")
    print(f"Output path: {output_path}\n")

    db_name = GLOBAL_CONFIG.get("DATABASE", "database_name")
    db_user = GLOBAL_CONFIG.get("DATABASE", "database_user")
    db_password = GLOBAL_CONFIG.get("DATABASE", "database_password")

    # Create a temporary directory for organizing files
    print("Creating temporary directory structure...")
    temp_dir = Path(output_path) / f"textpair_{table}_temp"
    temp_dir.mkdir(exist_ok=True)

    # Create the backup directory that will be the root in the tarball
    backup_dir = temp_dir / f"{table}_textpair_backup"
    backup_dir.mkdir(exist_ok=True)

    # Copy web app contents to backup directory using original directory name
    web_app_name = Path(web_app_path).name
    web_app_dest = backup_dir / web_app_name
    print(f"Copying web application files to {web_app_name}...")
    shutil.copytree(web_app_path, web_app_dest, dirs_exist_ok=True)
    print("✓ Web application files copied")

    # Check if we have a source_data directory in our temp directory
    # if not, call the back_up_philo_db_data function to create one
    source_data_dir = web_app_dest / "source_data/data"
    if not source_data_dir.exists():
        print("\nBacking up PhiloLogic databases...")
        app_config = json.load(open(web_app_dest / "appConfig.json"))
        source_data_dir.mkdir(parents=True)
        back_up_philo_db_data(Path(app_config["sourcePhiloDBPath"]), source_data_dir)
        if app_config["targetPhiloDBPath"] and app_config["sourcePhiloDBPath"] != app_config["targetPhiloDBPath"]:
            print("\n  - Found separate target database, backing up...")
            target_data_dir = web_app_dest / "target_data/data"
            target_data_dir.mkdir(parents=True)
            back_up_philo_db_data(Path(app_config["targetPhiloDBPath"]), target_data_dir)
        print("✓ PhiloLogic databases backup complete\n")

    # Dump database tables to backup directory
    print("Checking for TextPAIR database tables...")
    existing_tables = [
        table_name
        for table_name in [table, f"{table}_groups", f"{table}_ordered"]
        if table_exists(db_user, db_password, table_name)
    ]

    print(f"Found {len(existing_tables)} tables to backup")
    sql_files = []
    for table_name in existing_tables:
        sql_file = f"textpair_{table_name}.sql"
        sql_path = backup_dir / sql_file
        print(f"  - Dumping table {table_name}...")
        os.system(f"PGPASSWORD={db_password} pg_dump -U {db_user} {db_name} -t {table_name} > {sql_path}")
        sql_files.append(sql_file)
    print("✓ Database tables backup complete\n")

    # Create tarball from temp directory
    print("Creating final backup archive...")
    tar_path = Path(output_path) / f"textpair_{table}.tar.lz4"
    current_dir = os.getcwd()
    try:
        os.chdir(temp_dir)
        print("  - Creating tar archive...")
        # Create temporary tar file
        temp_tar = "temp.tar"
        os.system(f"tar cf {temp_tar} {table}_textpair_backup")

        print("  - Compressing with LZ4...")
        # Read the tar file and compress with lz4
        with open(temp_tar, 'rb') as f:
            tar_data = f.read()
        compressed_data = lz4.frame.compress(tar_data, compression_level=3)

        # Write the compressed data
        with open(tar_path, 'wb') as f:
            f.write(compressed_data)

        # Clean up temporary tar file
        os.remove(temp_tar)

    finally:
        os.chdir(current_dir)

    # Clean up
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)

    print(f"\n✓ Backup completed successfully!")
    print(f"Backup archive created at: {tar_path}")


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