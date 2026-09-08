#!/bin/bash
# TextPAIR Mac Bare-Metal Install Script
# For use without Docker, without web app
# Forked from ARTFL-Project/text-pair
# 
# Usage: ./install_bare_metal_mac.sh [-p X.Y | -p X.Y.Z]
#   -p  Python to install via pyenv: a series (3.12 -> newest 3.12.x) or an exact version.
#       Default: newest 3.11.x. Must be >= 3.11 (requires-python in lib/pyproject.toml).

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Python version selection. The floor tracks requires-python in lib/pyproject.toml; the
# default is the series we've tested on macOS. Override with -p (see usage above).
MIN_PY_SERIES="3.11"
PY_SERIES="$MIN_PY_SERIES"
while getopts "p:" opt; do
    case $opt in
        p) PY_SERIES="$OPTARG" ;;
        *) echo "Usage: $0 [-p X.Y | -p X.Y.Z]"; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

echo -e "${GREEN}TextPAIR Mac Bare-Metal Installer${NC}"
echo "=================================="
echo ""

# =============================================================================
# PATCH PHILOLOGIC WC -L (macOS compatibility)
# =============================================================================
patch_philologic_wc() {
    echo "Patching PhiloLogic line_count.py for macOS..."
    
    # Find the philologic installation
    local philologic_path=$("$PYTHON_BIN" -c "import philologic; print(philologic.__path__[0])" 2>/dev/null)
    
    if [ -z "$philologic_path" ]; then
        echo -e "${YELLOW}  PhiloLogic not yet installed, will patch after pip install${NC}"
        return 0
    fi
    
    local line_count_file="${philologic_path}/utils/line_count.py"
    
    if [ -f "$line_count_file" ]; then
        # Check if already patched
        if grep -q 'wc -l < {file_path}' "$line_count_file"; then
            echo "  Already patched"
        else
            # Rewrite the entire file - the upstream non-lz4 branch is broken
            # (runs cut on empty stdin instead of wc -l on the file)
            cat > /tmp/_line_count_patch.py << 'PATCH'
#!/usr/bin/env python3
"""Count number of lines in a file using subprocess module."""
import subprocess
def count_lines(file_path, lz4=False):
    """Count number of lines in a file."""
    if lz4:
        cmd = f"lz4 -dc {file_path} | wc -l"
    else:
        cmd = f"wc -l < {file_path}"
    process = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    count = int(process.stdout.strip())
    return count
PATCH
            sudo cp /tmp/_line_count_patch.py "$line_count_file"
            rm /tmp/_line_count_patch.py
            echo "  Patched: rewrote line_count.py (upstream non-lz4 branch was broken)"
        fi
    else
        echo -e "${YELLOW}  line_count.py not found at expected path${NC}"
    fi
    
    echo ""
}

# =============================================================================
# ARCHITECTURE CHECK
# =============================================================================
check_architecture() {
    local arch=$(uname -m)
    echo "Checking architecture..."
    echo "  Detected: $arch"
    
    if [ "$arch" == "arm64" ]; then
        BINARY_ARCH="aarch64"
        echo "  Binary:   aarch64 (Apple Silicon)"
    elif [ "$arch" == "x86_64" ]; then
        BINARY_ARCH="x86_64"
        echo "  Binary:   x86_64 (Intel)"
    else
        echo -e "${RED}Unsupported architecture: $arch${NC}"
        exit 1
    fi
    echo ""
}

# =============================================================================
# DEPENDENCY CHECK
# =============================================================================
check_dependencies() {
    echo "Checking dependencies..."

    # Homebrew (required to auto-install pyenv/Go below)
    if command -v brew &> /dev/null; then
        echo "  Homebrew: found"
    else
        echo -e "${RED}  ERROR: Homebrew not found. TextPAIR needs it to install pyenv and the Go toolchain.${NC}"
        echo -e "${YELLOW}  Install Homebrew first, then re-run this script:${NC}"
        echo '    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        exit 1
    fi

    # pyenv (guarantees a Python that satisfies requires-python regardless of whatever
    # python3 happens to be on the ambient PATH - relying on system/Homebrew python3
    # directly is what let a Python 3.9 slip past the old version check below)
    if command -v pyenv &> /dev/null; then
        echo "  pyenv: found"
    else
        echo "  pyenv: not found, installing with Homebrew..."
        brew install pyenv
        echo "  pyenv installed"
    fi

    # Validate -p: an exact X.Y.Z is used as-is; an X.Y series resolves to the newest X.Y.Z
    # pyenv knows about. Enforce the floor up front so a too-old -p fails fast instead of
    # after a slow pyenv build (pip would reject it anyway via requires-python).
    if [[ ! "$PY_SERIES" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
        echo -e "${RED}  ERROR: -p expects X.Y or X.Y.Z (got '$PY_SERIES')${NC}"
        exit 1
    fi
    IFS=. read -r want_major want_minor _ <<< "$PY_SERIES"
    IFS=. read -r min_major min_minor _ <<< "$MIN_PY_SERIES"
    if [ "$want_major" -lt "$min_major" ] || { [ "$want_major" -eq "$min_major" ] && [ "$want_minor" -lt "$min_minor" ]; }; then
        echo -e "${RED}  ERROR: Python $PY_SERIES is below the minimum $MIN_PY_SERIES required by lib/pyproject.toml${NC}"
        exit 1
    fi
    if [[ "$PY_SERIES" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        TARGET_PY_VERSION="$PY_SERIES"
    else
        TARGET_PY_VERSION=$(pyenv install --list | grep -E "^\s*${PY_SERIES//./\\.}\.[0-9]+$" | tail -1 | xargs)
    fi
    if [ -z "$TARGET_PY_VERSION" ]; then
        echo -e "${RED}  ERROR: could not find a Python ${PY_SERIES}.x version via pyenv${NC}"
        exit 1
    fi
    # Install it if missing, then pin this directory to it. pyenv local writes .python-version,
    # which is gitignored: it's machine-local state (and uv reads it too, so committing it would
    # override whatever Python upstream's uv-based install asks for).
    if ! pyenv versions --bare | grep -qx "$TARGET_PY_VERSION"; then
        echo "  Installing Python $TARGET_PY_VERSION via pyenv (this can take a few minutes)..."
        pyenv install "$TARGET_PY_VERSION"
    fi
    pyenv local "$TARGET_PY_VERSION"
    PYTHON_BIN="$(pyenv root)/versions/$TARGET_PY_VERSION/bin/python3"
    echo "  Python: $("$PYTHON_BIN" --version) (pyenv, pinned to this directory via .python-version)"

    # Confirm pyenv's shims are wired into the user's shell so `textpair`/`python3` keep
    # resolving to this pinned version in future terminal sessions, not just this script run.
    local shell_rc=""
    case "$SHELL" in
        */zsh) shell_rc="$HOME/.zshrc" ;;
        */bash) shell_rc="$HOME/.bash_profile" ;;
        *) shell_rc="$HOME/.profile" ;;
    esac
    if [ -f "$shell_rc" ] && grep -q 'pyenv init' "$shell_rc"; then
        echo "  pyenv shell integration: found in $shell_rc"
    else
        echo -e "${YELLOW}  pyenv shell integration not found in $shell_rc${NC}"
        echo -e "${YELLOW}  Add this line to $shell_rc, then restart your terminal:${NC}"
        echo '    eval "$(pyenv init -)"'
    fi

    # ripgrep (optional but recommended)
    if command -v rg &> /dev/null; then
        echo "  ripgrep: $(rg --version | head -1)"
    else
        echo -e "${YELLOW}  ripgrep: not found (optional, install with: brew install ripgrep)${NC}"
    fi

    # Homebrew (required to auto-install Go below)
    if command -v brew &> /dev/null; then
        echo "  Homebrew: found"
    else
        echo -e "${RED}  ERROR: Homebrew not found. TextPAIR needs it to install the Go toolchain (used to build compareNgrams).${NC}"
        echo -e "${YELLOW}  Install Homebrew first, then re-run this script:${NC}"
        echo '    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        exit 1
    fi

    # Go (required to build compareNgrams; the bundled binaries are Linux-only)
    if command -v go &> /dev/null; then
        echo "  Go: $(go version)"
    else
        echo "  Go: not found, installing with Homebrew..."
        brew install go
        echo "  Go installed"
    fi

    # lz4 CLI (used to merge alignment result batches)
    if command -v lz4 &> /dev/null; then
        echo "  lz4: found"
    else
        echo "  lz4: not found, installing with Homebrew..."
        brew install lz4
    fi

    echo ""
}

# =============================================================================
# INSTALL
# =============================================================================
install_textpair() {
    echo "Installing TextPAIR..."

    # Install textpair_llm first (local dependency)
    if [ -d "lib/textpair_llm" ]; then
        echo "  Installing textpair_llm..."
        "$PYTHON_BIN" -m pip install -e lib/textpair_llm/. --break-system-packages --quiet
    fi

    # Install main package
    echo "  Installing textpair..."
    "$PYTHON_BIN" -m pip install -e lib/. --break-system-packages

    echo ""
}

# =============================================================================
# SEED GLOBAL SETTINGS (user-level path, no sudo/root needed on macOS)
# =============================================================================
setup_global_settings() {
    local target="$HOME/.text-pair/global_settings.ini"
    echo "Setting up $target..."

    if [ -f "$target" ] || [ -f /etc/text-pair/global_settings.ini ]; then
        echo "  Already exists, leaving as-is"
    else
        mkdir -p "$HOME/.text-pair"
        cp config/global_settings.ini "$target"
        echo "  Seeded from config/global_settings.ini"
        echo -e "${YELLOW}  Edit $target with your actual PostgreSQL credentials (only needed if you drop --skip_web_app)${NC}"
    fi

    echo ""
}

# =============================================================================
# SCAFFOLD STARTER CORPUS CONFIG
# =============================================================================
scaffold_config() {
    local target="my_config.ini"
    echo "Setting up $target..."

    if [ -f "$target" ]; then
        echo "  Already exists, leaving as-is ($target)"
    else
        cp config/sa_config.ini "$target"
        echo "  Seeded from config/sa_config.ini"
        echo -e "${YELLOW}  Edit $target and set source_file_path to your corpus directory before running textpair${NC}"
    fi

    echo ""
}

# =============================================================================
# INSTALL BINARY
# =============================================================================
install_binary() {
    echo "Installing compareNgrams binary..."

    # The prebuilt binaries under lib/core/binary are Linux ELF executables (upstream only
    # targets Linux) and cannot run on macOS at all, even when the CPU architecture matches.
    # Build a native Mach-O binary from source instead. Go is guaranteed present at this point
    # (installed by check_dependencies if it was missing).
    echo "  Building compareNgrams from source with Go..."
    (cd lib/core/src/compareNgrams && go build -o /tmp/compareNgrams_build .)
    sudo cp /tmp/compareNgrams_build /usr/local/bin/compareNgrams
    rm -f /tmp/compareNgrams_build
    sudo chmod +x /usr/local/bin/compareNgrams
    echo "  Built and installed to /usr/local/bin/compareNgrams"

    echo ""
}

# =============================================================================
# VERIFY
# =============================================================================
verify_install() {
    echo "Verifying installation..."

    # Check the pyenv-pinned interpreter directly, since `command -v textpair` only
    # works if pyenv's shims are already wired into this shell's PATH (see the
    # shell-integration note printed by check_dependencies).
    local textpair_bin="$(pyenv root)/versions/$TARGET_PY_VERSION/bin/textpair"
    if [ -x "$textpair_bin" ]; then
        echo -e "${GREEN}  textpair command found ($textpair_bin)${NC}"
    else
        echo -e "${RED}  ERROR: textpair command not found at $textpair_bin${NC}"
        exit 1
    fi
    if ! command -v textpair &> /dev/null; then
        echo -e "${YELLOW}  Note: textpair isn't on PATH in this shell yet - see the pyenv shell integration note above${NC}"
    fi
    
    if command -v compareNgrams &> /dev/null; then
        echo -e "${GREEN}  compareNgrams binary found${NC}"
    else
        echo -e "${RED}  ERROR: compareNgrams binary not found${NC}"
        exit 1
    fi
    
    echo ""
    echo -e "${GREEN}Installation complete!${NC}"
    echo ""
    echo "Usage:"
    echo "  textpair --config=my_config.ini --skip_web_app --output_path=/tmp/textpair-out --workers=4 alignment_name"
    echo ""
    echo "Notes:"
    echo "  - Edit my_config.ini first and set source_file_path to your corpus directory"
    echo "  - ulimit is automatically increased on macOS (no manual fix needed)"
    echo "  - Use absolute paths in my_config.ini for source_file_path"
    echo "  - Avoid paths with spaces (copy corpus to /tmp if on iCloud)"
    echo "  - Input files should be TEI XML format"
    echo "  - Downloading a Spacy model (python -m spacy download <model>) is only needed if"
    echo "    you enable POS/entity filtering or spacy-based lemmatization in my_config.ini"
}

# =============================================================================
# MAIN
# =============================================================================
main() {
    check_architecture
    check_dependencies
    install_textpair
    patch_philologic_wc
    setup_global_settings
    scaffold_config
    install_binary
    verify_install
}

main "$@"