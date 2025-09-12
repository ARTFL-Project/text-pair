#!/bin/bash

# Default values
PYTHON_VERSION="python3"
USE_CUDA=false

# Parse command line arguments
while getopts "p:c" opt; do
  case $opt in
    p) PYTHON_VERSION="$OPTARG"
    ;;
    c) USE_CUDA=true
    ;;
    *) echo "Usage: $0 [-p python_version] [-c]"
       echo "  -p: Specify Python version (default: python3)"
       echo "  -c: Install with CUDA support (default: CPU only)"
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
sudo chown -R $USER:$USER /var/lib/text-pair

# Create the virtual environment using uv
echo "Creating virtual environment with uv..."
uv venv -p $PYTHON_VERSION /var/lib/text-pair/textpair_env
source /var/lib/text-pair/textpair_env/bin/activate

# Install textpair with appropriate PyTorch configuration
if [ "$USE_CUDA" = true ]; then
    echo "Installing textpair with CUDA support..."
    uv pip install -e lib/.[cuda]
else
    echo "Installing textpair with CPU-only PyTorch..."
    uv pip install -e lib/.[cpu]
fi

deactivate

# Install the textpair script
sudo cp textpair /usr/local/bin/

# Install compareNgrams binary
arch=$(uname -m)
if [ "$arch" = "x86_64" ]; then
    binary_path="lib/core/binary/x86_64/compareNgrams"
elif [ "$arch" = "aarch64" ]; then
    binary_path="lib/core/binary/aarch64/compareNgrams"
else
    echo "Only x86_64 and ARM are supported at this time."
    exit 1
fi
sudo rm -f /usr/local/bin/compareNgrams
sudo cp "$binary_path" /usr/local/bin/compareNgrams
sudo chmod +x /usr/local/bin/compareNgrams


# Install the web application components
echo -e "\nMoving web application components into place..."
sudo mkdir -p /var/lib/text-pair
if [ ! -f /var/lib/text-pair/api_server/web_server.sh ]
    then
        sudo cp -Rf api_server /var/lib/text-pair/api_server/
else
    echo "/var/lib/text-pair/api_server/web_server.sh already exists, not modifying..."
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
        sudo touch /etc/text-pair/global_settings.ini
        echo "[WEB_APP]" | sudo tee -a /etc/text-pair/global_settings.ini > /dev/null
        echo "api_server = http://localhost/text-pair-api" | sudo tee -a /etc/text-pair/global_settings.ini > /dev/null
        echo "web_app_path =" | sudo tee -a /etc/text-pair/global_settings.ini > /dev/null
        echo "[DATABASE]" | sudo tee -a /etc/text-pair/global_settings.ini > /dev/null
        echo "database_name =" | sudo tee -a /etc/text-pair/global_settings.ini > /dev/null
        echo "database_user =" | sudo tee -a /etc/text-pair/global_settings.ini > /dev/null
        echo "database_password =" | sudo tee -a /etc/text-pair/global_settings.ini > /dev/null
        echo "Make sure you create a PostgreSQL database with a user with read/write access to that database and configure /etc/text-pair/global_settings.ini accordingly."
else
    echo "/etc/text-pair/global_settings.ini already exists, not modifying..."
fi

echo -e "\n## INSTALLATION COMPLETE ##"
echo "TextPAIR has been installed successfully using uv!"
if [ "$USE_CUDA" = true ]; then
    echo "- PyTorch was installed with CUDA support"
else
    echo "- PyTorch was installed with CPU-only support"
    echo "- To install with CUDA support in the future, run: $0 -c"
fi
echo ""
echo "## NEXT STEPS ##"
echo "In order to start the TextPAIR web app, you need to configure and start up the web_server.sh script."
echo "You can either:"
echo "- Start it manually at /var/lib/text-pair/api_server/web_server.sh"
echo "- Use the systemd init script located at /var/lib/text-pair/api_server/textpair.service: for this you will need to copy it to your OS systemd folder (usually /etc/systemd/system/) and run 'systemctl enable textpair && systemctl start textpair' as root"