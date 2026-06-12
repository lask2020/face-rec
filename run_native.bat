@echo off
setlocal enabledelayedexpansion

echo ===========================================
echo FaceRec AI Worker Node - Windows Launcher
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
if not exist "venv_native\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment...
    python -m venv venv_native
)

:: Activate virtual environment
call venv_native\Scripts\activate.bat

:: Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

:: Install requirements
:: Note: For AMD GPU on Windows, we automatically replace onnxruntime with onnxruntime-directml
echo [INFO] Installing dependencies (this may take a while the first time)...
python -m pip install -r backend\requirements.txt

echo [INFO] Patching for Windows AMD/Intel GPU (DirectML)...
python -m pip uninstall -y onnxruntime >nul 2>&1
python -m pip install onnxruntime-directml>=1.19.0

echo [INFO] Compiling gRPC protobuf definitions...
python -m grpc_tools.protoc -I. --python_out=backend --grpc_python_out=backend facerec.proto

echo ===========================================
echo Launching GUI application natively...
echo ===========================================
python backend\ai_worker_gui.py

pause
