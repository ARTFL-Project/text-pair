---
title: Linux
parent: Installation
nav_order: 2
---

# Linux

These steps install TextPAIR on the host machine. The commands are for Ubuntu; adapt the package names to your distribution.

## 1. Install dependencies

```console
sudo apt install postgresql postgresql-contrib nodejs npm curl
```

You also need [pgvector](https://github.com/pgvector/pgvector?tab=readme-ov-file#installation) for your PostgreSQL version, such as the `postgresql-16-pgvector` package for PostgreSQL 16. The install script stops if it is missing.

## 2. Configure PostgreSQL

Open a PostgreSQL shell as the `postgres` user:

```console
sudo -u postgres psql
```

Create a user, a database it owns, and the two extensions TextPAIR needs:

```sql
CREATE USER your_user WITH PASSWORD 'your_password';
CREATE DATABASE your_db_name OWNER your_user;
\c your_db_name
CREATE EXTENSION pg_trgm;
CREATE EXTENSION vector;
\q
```

TextPAIR connects through the local socket with a password, so you may need to change local connections from `peer` to `md5` (or `scram-sha-256`) in `pg_hba.conf`, then restart PostgreSQL. The path to that file depends on your PostgreSQL version:

```console
sudo nano /etc/postgresql/16/main/pg_hba.conf
sudo systemctl restart postgresql
```

## 3. Create the directory for web applications

```console
sudo mkdir -p /var/www/html/text-pair/
sudo chown -R $(whoami) /var/www/html/text-pair/
```

## 4. Run the install script

From the root of the repository:

```console
./install.sh
```

Options: `-p` to choose the Python executable (`python3` by default), `-c` to install with CUDA support.

The script installs TextPAIR in `/var/lib/text-pair`, puts the `textpair` command in `/usr/local/bin`, and sets up the `textpair` systemd service, which serves the web applications and their API.

## 5. Point TextPAIR at your database

Edit `/etc/text-pair/global_settings.ini`:

- under `[DATABASE]`, the database name, user, and password you created in step 2
- under `[WEB_APP]`, `web_app_path` (the directory from step 3) and `api_server` (the URL where the API is served)

Then restart the service:

```console
sudo systemctl restart textpair
```

## Web server

The `textpair` service runs `/var/lib/text-pair/api_server/web_server.sh`, which serves on port 80. Edit that script to change the port, the number of workers, or to add your SSL key and certificate. If another web server already listens on port 80, set another port there and proxy requests to it.
