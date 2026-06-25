#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Color definitions for terminal output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== FaceRec AI Worker Native Launcher ===${NC}"
echo ""

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR] python3 is not installed or not in PATH!${NC}"
    echo "Please install Python 3.9+ using Homebrew or from python.org"
    exit 1
fi

VENV_DIR="venv_native"

# 1. Create a virtual environment for native running if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating native virtual environment in $VENV_DIR...${NC}"
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR"/bin/activate

# 2. Upgrade pip and install dependencies
echo -e "${YELLOW}Upgrading pip and installing python packages...${NC}"
pip install --upgrade pip setuptools wheel
pip install -r backend/requirements.txt
pip install -U ultralytics

# 3. Compile gRPC protobuf definitions locally
echo -e "${YELLOW}Compiling gRPC protobuf definitions...${NC}"
python3 -m grpc_tools.protoc -Ibackend --python_out=backend --grpc_python_out=backend backend/facerec.proto

echo -e "${GREEN}Dependencies installed and compiled successfully.${NC}"
echo -e "${YELLOW}Launching GUI application natively...${NC}"

# 4. Launch the GUI
python3 backend/ai_worker_gui.py

# Deactivate virtual environment after exit
deactivate
