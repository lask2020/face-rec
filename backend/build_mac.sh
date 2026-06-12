#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Color definitions for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}  FaceRec AI Worker Binary Builder for macOS${NC}"
echo -e "${BLUE}====================================================${NC}"
echo ""

# Function to show usage
show_usage() {
    echo "Usage: ./build_mac.sh [cpu|openvino]"
    echo ""
    echo "Options:"
    echo "  cpu       Build with standard CPU execution provider (default, supports Apple Silicon CoreML)"
    echo "  openvino  Build with Intel OpenVINO execution provider"
    echo ""
}

# Determine target provider
TARGET=${1:-"cpu"}
case "$TARGET" in
    cpu)
        ONNX_PACKAGE="onnxruntime>=1.19.0"
        SUFFIX="CPU"
        ;;
    openvino)
        ONNX_PACKAGE="onnxruntime-openvino>=1.19.0"
        SUFFIX="OpenVINO"
        ;;
    help|--help|-h)
        show_usage
        exit 0
        ;;
    *)
        echo -e "${RED}Error: Unknown target provider '$TARGET'${NC}"
        show_usage
        exit 1
        ;;
esac

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR] python3 is not installed or not in PATH!${NC}"
    echo "Please install Python 3.9+ using Homebrew or from python.org"
    exit 1
fi

# Determine python version
PYTHON_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "Using Python version: ${GREEN}${PYTHON_VER}${NC}"
echo -e "Target hardware platform: ${GREEN}${SUFFIX}${NC}"

# Define build environment directory
VENV_DIR="venv_build"

# Clean up any previous virtual environment or builds
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Cleaning up previous build virtual environment...${NC}"
    rm -rf "$VENV_DIR"
fi

# 1. Create a virtual environment for building
echo -e "${YELLOW}[1/4] Creating virtual environment for clean build...${NC}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR"/bin/activate

# 2. Upgrade pip and install pyinstaller
echo -e "${YELLOW}[2/4] Installing/Upgrading PyInstaller and packaging tools...${NC}"
pip install --upgrade pip setuptools wheel
pip install --upgrade pyinstaller

# 3. Install dependencies from requirements.txt, replacing onnxruntime with target package
echo -e "${YELLOW}[3/4] Installing dependencies...${NC}"
# Copy requirements but strip out standard onnxruntime to avoid conflict
grep -v "onnxruntime" requirements.txt > temp_requirements.txt
pip install -r temp_requirements.txt
rm temp_requirements.txt

# Install the selected target onnxruntime package
echo -e "${YELLOW}Installing specified ONNX Runtime package: ${ONNX_PACKAGE}...${NC}"
pip install "$ONNX_PACKAGE"

# Compile gRPC protobuf definitions
echo -e "${YELLOW}Compiling gRPC protobuf definitions...${NC}"
python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. facerec.proto

# 4. Compile GUI application to macOS App Bundle & Executable
BINARY_NAME="FaceRec_AI_Worker_macOS_${SUFFIX}"
echo -e "${YELLOW}[4/4] Compiling GUI application with PyInstaller to '${BINARY_NAME}'...${NC}"

# Clean up previous outputs to prevent directory non-empty errors
rm -rf "dist/${BINARY_NAME}" "dist/${BINARY_NAME}.app" "build/${BINARY_NAME}"

# Use ":" as the directory separator on macOS/Linux and add --noconfirm
pyinstaller --noconfirm --noconsole --onefile --name="${BINARY_NAME}" --add-data "app:app" --add-data "facerec.proto:." ai_worker_gui.py

echo ""
echo -e "${BLUE}====================================================${NC}"
echo -e "${GREEN}  Compilation successful!${NC}"
echo -e "  Your executable is located at: ${GREEN}dist/${BINARY_NAME}${NC}"
echo -e "  Your macOS bundle is located at: ${GREEN}dist/${BINARY_NAME}.app${NC}"
echo -e "${BLUE}====================================================${NC}"


# Deactivate virtual env
deactivate

# Clean up build virtual environment and generated protobuf files
echo -e "${YELLOW}Cleaning up build environment...${NC}"
rm -rf "$VENV_DIR"
rm -f facerec_pb2.py facerec_pb2_grpc.py

echo -e "${GREEN}Done!${NC}"
