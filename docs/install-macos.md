---
title: macOS (experimental)
parent: Installation
nav_order: 3
---

# macOS (experimental)

TextPAIR officially supports 64-bit Linux, and [Docker](install-docker.html) is the recommended path for production use. For local development and research runs on a Mac (Apple Silicon or Intel), an opt-in installer sets up the full sequence-alignment pipeline natively, without Docker.

**Prerequisite:** [Homebrew](https://brew.sh). The script stops with instructions if it is missing.

```console
./install_bare_metal_mac.sh
```

The script:

- installs `pyenv` and `lz4` with Homebrew if absent, then installs Python with pyenv and creates the environment TextPAIR runs in. It takes the newest 3.11.x by default; pass `-p 3.12` or an exact `-p 3.12.4` to choose another, anything 3.11 or later.
- installs the TextPAIR Python package into that environment
- patches the installed PhiloLogic dependency's `line_count.py` (its non-lz4 code path is broken, and BSD `wc` output differs from GNU)
- seeds `~/.text-pair/global_settings.ini` and a starter `my_config.ini` (copied from `config/sa_config.ini`) if you do not already have them

Then edit `my_config.ini` (at minimum, set `source_file_path` to your corpus directory) and run:

```console
textpair --config=my_config.ini --skip_web_app --output_path=/tmp/textpair-out --workers=8 my_run_name
```

## Known limitations

- Avoid corpus and output paths containing spaces. This is a PhiloLogic limitation; note that iCloud-synced folders live under a path with spaces, so copy corpora to `/tmp` or similar first.
- The PostgreSQL credentials in `~/.text-pair/global_settings.ini` are only needed if you drop `--skip_web_app` to build the web application.

This mode was developed for the corpus-scale alignment runs in Tarah Wheeler's (2026) DPhil thesis at the University of Oxford.
