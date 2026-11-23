---
title: Installation
nav_order: 2
has_children: true
---

# Installation

TextPAIR runs on 64-bit Linux. There are three ways to install it:

| Method | Use it for |
|:-------|:-----------|
| [Docker](install-docker.html) | The recommended install: TextPAIR and its database in one container. |
| [Linux](install-linux.html) | Installing on the host machine with the install script. |
| [macOS](install-macos.html) | Experimental: local development and research runs, without the web application by default. |

## Requirements

The Docker image provides all of these. For a host install you need:

- Python 3.11 or later
- Node and npm, to build each database's web application
- PostgreSQL, with a dedicated database, a user with read/write permissions on it, and the `pg_trgm` and `vector` ([pgvector](https://github.com/pgvector/pgvector)) extensions created on it
