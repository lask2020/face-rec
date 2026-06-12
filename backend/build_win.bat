@echo off
echo ====================================================
echo  FaceRec AI Worker Binary Builder for Windows
echo ====================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.9+ from python.org and check "Add Python to PATH" during installation.
    pause
    exit /b %errorlevel%
)

REM Determine target provider
SET TARGET=%1
IF "%TARGET%"=="" SET TARGET=cpu

IF "%TARGET%"=="cpu" (
    SET ONNX_PACKAGE=onnxruntime^>=1.19.0
    SET SUFFIX=CPU
) ELSE IF "%TARGET%"=="gpu" (
    SET ONNX_PACKAGE=onnxruntime-gpu^>=1.19.0
    SET SUFFIX=CUDA
) ELSE IF "%TARGET%"=="directml" (
    SET ONNX_PACKAGE=onnxruntime-directml^>=1.19.0
    SET SUFFIX=DirectML
) ELSE IF "%TARGET%"=="openvino" (
    SET ONNX_PACKAGE=onnxruntime-openvino^>=1.19.0
    SET SUFFIX=OpenVINO
) ELSE (
    echo [ERROR] Unknown target provider '%TARGET%'!
    echo Usage: build_win.bat [cpu^|gpu^|directml^|openvino]
    pause
    exit /b 1
)

echo Target hardware platform: %SUFFIX%
echo.

REM Define build environment directory
SET VENV_DIR=venv_build

REM Clean up any previous virtual environment or builds
if exist "%VENV_DIR%" (
    echo Cleaning up previous build virtual environment...
    rmdir /s /q "%VENV_DIR%"
)

echo [1/4] Creating virtual environment for clean build...
python -m venv "%VENV_DIR%"
call "%VENV_DIR%"\Scripts\activate.bat

echo [2/4] Installing/Upgrading PyInstaller and packaging tools...
python -m pip install --upgrade pip setuptools wheel
pip install --upgrade pyinstaller

echo [3/4] Installing dependencies...
pip install -r requirements.txt
echo Uninstalling standard CPU onnxruntime to avoid conflict...
pip uninstall -y onnxruntime
echo Installing target package: %ONNX_PACKAGE%...
pip install %ONNX_PACKAGE%

echo Compiling gRPC protobuf definitions...
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. facerec.proto

echo [4/4] Compiling GUI application with PyInstaller...
pyinstaller --noconfirm --noconsole --onefile --name="FaceRec_AI_Worker_Windows_%SUFFIX%" --add-data "app;app" --add-data "facerec.proto;." --collect-binaries onnxruntime ai_worker_gui.py

if %errorlevel% neq 0 (
    echo [ERROR] PyInstaller compilation failed!
    call deactivate
    del /f /q facerec_pb2.py facerec_pb2_grpc.py
    rmdir /s /q "%VENV_DIR%"
    pause
    exit /b %errorlevel%
)

echo.
echo ====================================================
echo  Compilation successful!
echo  Your executable is located at: dist\FaceRec_AI_Worker_Windows_%SUFFIX%.exe
echo ====================================================

call deactivate
echo Cleaning up build environment...
rmdir /s /q "%VENV_DIR%"
del /f /q facerec_pb2.py facerec_pb2_grpc.py
pause
