@echo off
setlocal enabledelayedexpansion

echo ===========================================
echo FaceRec Training - NVIDIA GPU (CUDA)
echo ===========================================

cd /d "%~dp0"

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.9, 3.10, or 3.11 from python.org
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist "venv_nvidia\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment for NVIDIA GPU...
    python -m venv venv_nvidia
)

:: Activate virtual environment
call venv_nvidia\Scripts\activate.bat

:: Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

:: Install requirements
echo [INFO] Installing dependencies...
python -m pip install -r backend\requirements.txt

:: Remove default ONNX Runtime and install GPU version for NVIDIA
echo [INFO] Setting up NVIDIA CUDA support...
python -m pip uninstall -y onnxruntime onnxruntime-directml onnxruntime-openvino >nul 2>&1
python -m pip install onnxruntime-gpu>=1.19.0

:: Install PyTorch with CUDA support for training
echo [INFO] Installing PyTorch with CUDA support...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: Compile gRPC protobuf definitions
echo [INFO] Compiling gRPC protobuf definitions...
python -m grpc_tools.protoc -I. --python_out=backend --grpc_python_out=backend facerec.proto

echo ===========================================
echo Ready for training on NVIDIA GPU
echo ===========================================
echo.
echo Usage:
echo   python backend/finetune_char_model.py ^
echo     --data /path/to/data.yaml ^
echo     --base-model /path/to/thai_char_yolo26s.pt ^
echo     --output /path/to/output.pt ^
echo     --epochs 30
echo.
echo Or run training interactively:
cmd /k

pause
