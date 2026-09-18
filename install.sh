#!/bin/bash

# Default values
PYTHON_VERSION="python3"
USE_CUDA=false
SKIP_LEMMATIZER="no"

# Parse command line arguments
while getopts "p:cL" opt; do
  case $opt in
    p) PYTHON_VERSION="$OPTARG"
    ;;
    c) USE_CUDA=true
    ;;
    L) SKIP_LEMMATIZER="yes"
    ;;
    *) echo "Usage: $0 [-p python_version] [-c] [-L]"
       echo "  -p: Specify Python version (default: python3)"
       echo "  -c: Install with CUDA support (default: CPU only)"
       echo "  -L: Skip the graph lemmatizer environment (a second torch copy)."
       echo "      Without it, thematic-graph term extraction falls back to regex"
       echo "      tokenization, which is worse on archaic spelling but still works."
       exit 1
    ;;
  esac
done

echo "Using Python version: $PYTHON_VERSION"
if [ "$USE_CUDA" = true ]; then
    echo "Installing with CUDA support"
else
    echo "Installing with CPU-only PyTorch"
fi

# Check if uv is installed, install if not
if ! command -v uv &> /dev/null; then
    echo "uv could not be found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    # Verify installation
    if ! command -v uv &> /dev/null; then
        echo "ERROR: Failed to install uv. Please install uv manually and try again."
        echo "Visit https://docs.astral.sh/uv/getting-started/installation/ for installation instructions."
        exit 1
    else
        echo "uv installed successfully"
    fi
else
    echo "uv is already installed"
fi

# Delete virtual environment if it already exists
if [ -d /var/lib/text-pair/textpair_env ]; then
    echo "Deleting existing TextPAIR installation..."
    sudo rm -rf /var/lib/text-pair/textpair_env
fi


# Give current user permission to write to /var/lib/textpair
sudo mkdir -p /var/lib/text-pair
sudo chown -R $USER:$(id -gn) /var/lib/text-pair

# Create the virtual environment using uv
echo "Creating virtual environment with uv..."
uv venv -p $PYTHON_VERSION /var/lib/text-pair/textpair_env
source /var/lib/text-pair/textpair_env/bin/activate

# Install textpair and textpair_llm together
if [ "$USE_CUDA" = true ]; then
    echo "Installing textpair with CUDA support..."
    uv pip install -e lib/.[cuda]
else
    echo "Installing textpair with CPU-only PyTorch..."
    uv pip install -e lib/.[cpu]
fi

deactivate

# Pack the modernization maps. They are stored as editable TSV; the loader builds
# this cache on first use anyway, but doing it here means the first alignment does
# not pay for it, and it surfaces a write-permission problem now rather than later.
echo ""
echo "Packing modernization maps..."
source /var/lib/text-pair/textpair_env/bin/activate
python -c "import textpair.preprocessing.modernize as m; m.pack_all()"
deactivate

# Numba cache for the sequence aligner's kernels. Wiped on every install so it is
# always repopulated from empty: mode 1777 forbids renaming over another user's
# files, which is what reusing a stale entry would require.
echo ""
echo "Setting up the sequence aligner's numba cache..."
sudo rm -rf /var/lib/text-pair/numba_cache
sudo mkdir -p /var/lib/text-pair/numba_cache
sudo chmod 1777 /var/lib/text-pair/numba_cache

# Pre-compile the kernels so the first real alignment does not pay for it. The
# three-document fixture warms the whole njit call graph, JSON and binary paths both.
source /var/lib/text-pair/textpair_env/bin/activate
warm_fixture="lib/textpair/sequence_alignment/tests/fixtures/no_byte_range"
warm_out=$(mktemp -d)
if python -m textpair.sequence_alignment.aligner \
        --source_files="$warm_fixture/ngrams" \
        --source_metadata="$warm_fixture/metadata/metadata.json" \
        --output_path="$warm_out" > /dev/null 2>&1; then
    echo "Aligner kernels pre-compiled into /var/lib/text-pair/numba_cache"
else
    echo "WARNING: could not pre-compile the aligner kernels; the first alignment will."
fi
rm -rf "$warm_out"
deactivate

# Create separate virtual environment for graph building
echo ""
echo "Creating separate virtual environment for graph building..."
if [ -d /var/lib/text-pair/graph ]; then
    echo "Deleting existing graph environment..."
    rm -rf /var/lib/text-pair/graph
fi

uv venv -p $PYTHON_VERSION /var/lib/text-pair/graph
source /var/lib/text-pair/graph/bin/activate


# Install textpair_graph and textpair_llm together
echo "Installing textpair_graph..."
if [ "$USE_CUDA" = true ]; then
    echo "Installing with GPU acceleration (cuML)..."
    uv pip install -e lib/textpair_graph[cuda] || {
        echo "WARNING: Failed to install CUDA graph libraries. Falling back to CPU alternatives..."
        uv pip install -e lib/textpair_graph
    }
else
    echo "Installing CPU-only graph dependencies..."
    uv pip install -e lib/textpair_graph
fi

deactivate
echo "Graph building environment created at /var/lib/text-pair/graph"

# Separate environment because spacy-transformers pins transformers<4.53.3,
# which the graph environment's label model (>=5.5) cannot share. Optional:
# without it term extraction falls back to regex tokenization.
if [ "$SKIP_LEMMATIZER" = "yes" ]; then
    echo "Skipping the lemmatizer environment (-L)."
else
    echo "Installing lemmatizer environment..."
    uv venv -p $PYTHON_VERSION /var/lib/text-pair/lemmatizer
    source /var/lib/text-pair/lemmatizer/bin/activate
    # torch LAST: installing it first does not hold, because the
    # spacy-transformers install re-resolves it to a CUDA-13 build that a
    # CUDA-12 driver refuses.
    # TODO: simplify all this
    uv pip install "spacy>=3.8.5,<3.9" spacy-transformers orjson lz4 scikit-learn tqdm
    if [ "$USE_CUDA" = true ]; then
        uv pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.8.0" || {
            echo "WARNING: CUDA torch install failed; falling back to CPU torch."
            uv pip install torch
        }
        uv pip install "cupy-cuda12x<14" || echo "WARNING: cupy install failed; lemmatizer will run on CPU."
    else
        uv pip install torch
    fi
    deactivate
    echo "Lemmatizer environment created at /var/lib/text-pair/lemmatizer"
    echo "Set spacy_model in the [GRAPH] config section to a spaCy model path to use it."
fi

# Install the textpair script
sudo cp textpair /usr/local/bin/

# A compareNgrams binary from an earlier install is left where it is, but nothing
# invokes it: the sequence aligner is textpair.sequence_alignment.aligner.

# Install the web application components
echo -e "\nMoving web application components into place..."
sudo mkdir -p /var/lib/text-pair
if [ ! -f /var/lib/text-pair/api_server/web_server.sh ]; then
    sudo cp -Rf api_server /var/lib/text-pair/api_server/
else
    echo "/var/lib/text-pair/api_server/web_server.sh already exists, not modifying..."
    # Always update service files on reinstall
    sudo cp api_server/textpair.service /var/lib/text-pair/api_server/textpair.service
    sudo cp api_server/com.textpair.server.plist /var/lib/text-pair/api_server/com.textpair.server.plist
fi

if [ -d web/web_app/node_modules ]
    then
        sudo rm -rf web/web_app/node_modules
fi
sudo cp -Rf api /var/lib/text-pair/
sudo cp -Rf web-app /var/lib/text-pair/
sudo cp -Rf config /var/lib/text-pair/

echo -e "\nMoving global configuration into place..."
sudo mkdir -p /etc/text-pair
if [ ! -f /etc/text-pair/global_settings.ini ]
    then
        # Seeded from the template so config/global_settings.ini stays the single
        # source of truth for the default settings. -m 644 is required: textpair
        # runs unprivileged and parse_config.py reads this file as the invoking user.
        sudo install -m 644 config/global_settings.ini /etc/text-pair/global_settings.ini
        echo "Make sure you create a PostgreSQL database with a user with read/write access to that database and configure /etc/text-pair/global_settings.ini accordingly."
else
    echo "/etc/text-pair/global_settings.ini already exists, not modifying..."
fi

# Check if pgvector extension is available in PostgreSQL
echo ""
echo "Checking for pgvector extension in PostgreSQL..."
if [ -f /usr/share/postgresql/*/extension/vector.control ] || [ -f /usr/local/share/postgresql/*/extension/vector.control ]; then
    echo "✓ pgvector extension found in PostgreSQL"
    echo ""
    echo "IMPORTANT: Before using TextPAIR, you must enable the vector extension in your database."
    echo "As a PostgreSQL superuser, run:"
    echo "  psql -d your_textpair_database -c 'CREATE EXTENSION IF NOT EXISTS vector;'"
else
    echo ""
    echo "ERROR: pgvector extension not found!"
    echo "TextPAIR requires the pgvector extension for PostgreSQL."
    echo ""
    echo "To install pgvector:"
    echo "  See https://github.com/pgvector/pgvector?tab=readme-ov-file#installation for installation instructions."
    echo ""
    echo "After installing pgvector, run this install script again."
    echo ""
    exit 1
fi

# Install and enable the web server service
echo -e "\nInstalling TextPAIR web server service..."
OS="$(uname -s)"
if [ "$OS" = "Linux" ]; then
    if command -v systemctl &> /dev/null; then
        sudo cp /var/lib/text-pair/api_server/textpair.service /etc/systemd/system/textpair.service
        sudo systemctl daemon-reload
        sudo systemctl enable textpair
        sudo systemctl restart textpair
        echo "TextPAIR service installed and (re)started via systemd."
        echo "  Manage with: systemctl {start|stop|restart|status} textpair"
    else
        echo "WARNING: systemctl not found. You can start the server manually:"
        echo "  /var/lib/text-pair/api_server/web_server.sh"
    fi
elif [ "$OS" = "Darwin" ]; then
    PLIST_SRC="/var/lib/text-pair/api_server/com.textpair.server.plist"
    PLIST_DST="$HOME/Library/LaunchAgents/com.textpair.server.plist"
    mkdir -p "$HOME/Library/LaunchAgents"
    cp "$PLIST_SRC" "$PLIST_DST"
    launchctl bootout gui/$(id -u) "$PLIST_DST" 2>/dev/null || true
    launchctl bootstrap gui/$(id -u) "$PLIST_DST"
    launchctl kickstart -k gui/$(id -u)/com.textpair.server 2>/dev/null || true
    echo "TextPAIR service installed and (re)started via launchd."
    echo "  Stop:    launchctl bootout gui/\$(id -u) $PLIST_DST"
    echo "  Start:   launchctl bootstrap gui/\$(id -u) $PLIST_DST"
    echo "  Logs:    /var/lib/text-pair/api_server/launchd_stdout.log"
else
    echo "WARNING: Unsupported OS ($OS). Start the server manually:"
    echo "  /var/lib/text-pair/api_server/web_server.sh"
fi

echo -e "\n## INSTALLATION COMPLETE ##"
echo "TextPAIR has been installed successfully using uv!"
if [ "$USE_CUDA" = true ]; then
    echo "- PyTorch was installed with CUDA support"
else
    echo "- PyTorch was installed with CPU-only support"
    echo "- To install with CUDA support in the future, run: $0 -c"
fi