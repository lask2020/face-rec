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
    exit /b 1
)

REM ── Parse arguments ────────────────────────────────────────────────────────
SET TARGET=%1
SET CLEAN_BUILD=%2
IF "%TARGET%"=="" SET TARGET=cpu

IF "%TARGET%"=="cpu" (
    SET ONNX_PACKAGE=onnxruntime>=1.19.0
    SET SUFFIX=CPU
) ELSE IF "%TARGET%"=="gpu" (
    SET ONNX_PACKAGE=onnxruntime-gpu>=1.19.0
    SET SUFFIX=CUDA
) ELSE IF "%TARGET%"=="directml" (
    SET ONNX_PACKAGE=onnxruntime-directml>=1.19.0
    SET SUFFIX=DirectML
) ELSE IF "%TARGET%"=="openvino" (
    SET ONNX_PACKAGE=onnxruntime-openvino>=1.19.0
    SET SUFFIX=OpenVINO
) ELSE (
    echo [ERROR] Unknown target '%TARGET%'
    echo Usage: build_win.bat [cpu^|gpu^|directml^|openvino] [--clean]
    pause
    exit /b 1
)

echo Target: %SUFFIX%
echo.

SET VENV_DIR=venv_build
SET MARKER_FILE=%VENV_DIR%\.build_marker

REM ── Decide whether to recreate venv ────────────────────────────────────────
SET NEED_INSTALL=0

IF "%CLEAN_BUILD%"=="--clean" (
    echo [--clean] Removing previous venv...
    if exist "%VENV_DIR%" rmdir /s /q "%VENV_DIR%"
    SET NEED_INSTALL=1
    GOTO :create_venv
)

REM Re-use existing venv if marker matches current target + requirements hash
IF NOT EXIST "%VENV_DIR%\Scripts\activate.bat" (
    SET NEED_INSTALL=1
    GOTO :create_venv
)

IF NOT EXIST "%MARKER_FILE%" (
    SET NEED_INSTALL=1
    GOTO :create_venv
)

REM Read saved marker (target|req_hash)
SET /p SAVED_MARKER=<"%MARKER_FILE%"

REM Compute a quick requirements.txt fingerprint (file size + date is fast enough)
FOR %%F IN (requirements.txt) DO SET REQ_STAMP=%%~zF_%%~tF
SET CURRENT_MARKER=%TARGET%^|%REQ_STAMP%

IF NOT "%SAVED_MARKER%"=="%CURRENT_MARKER%" (
    echo Dependencies or target changed — reinstalling packages...
    SET NEED_INSTALL=1
) ELSE (
    echo [CACHE HIT] Reusing existing venv for %SUFFIX% — skipping install.
)
GOTO :build

:create_venv
IF NOT EXIST "%VENV_DIR%\Scripts\activate.bat" (
    echo [1/4] Creating virtual environment...
    python -m venv "%VENV_DIR%"
)

:build
call "%VENV_DIR%\Scripts\activate.bat"

IF "%NEED_INSTALL%"=="1" (
    echo [2/4] Installing build tools...
    python -m pip install --quiet pip setuptools wheel pyinstaller

    echo [3/4] Installing dependencies...
    pip install --quiet -r requirements.txt

    echo Switching to %SUFFIX% onnxruntime...
    pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml onnxruntime-openvino >nul 2>&1
    pip install --quiet "%ONNX_PACKAGE%"

    IF "%TARGET%"=="directml" (
        echo Installing torch-directml for AMD/Intel GPU training...
        pip install --quiet torch-directml
    )

    REM Save marker so next build can skip this step
    FOR %%F IN (requirements.txt) DO SET REQ_STAMP=%%~zF_%%~tF
    echo %TARGET%^|%REQ_STAMP%>"%MARKER_FILE%"
) ELSE (
    echo [2/4] Skipped ^(cached^).
    echo [3/4] Skipped ^(cached^).
)

echo Compiling protobuf...
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. facerec.proto

echo [4/4] Compiling with PyInstaller...
pyinstaller --noconfirm --noconsole --onefile ^
    --name="FaceRec_AI_Worker_Windows_%SUFFIX%" ^
    --add-data "app;app" ^
    --add-data "facerec.proto;." ^
    --hidden-import finetune_char_model ^
    --collect-binaries onnxruntime ^
    ai_worker_gui.py

if %errorlevel% neq 0 (
    echo [ERROR] PyInstaller compilation failed!
    call deactivate
    del /f /q facerec_pb2.py facerec_pb2_grpc.py 2>nul
    pause
    exit /b 1
)

call deactivate
del /f /q facerec_pb2.py facerec_pb2_grpc.py 2>nul

echo.
echo ====================================================
echo  Done! Output: dist\FaceRec_AI_Worker_Windows_%SUFFIX%.exe
echo ====================================================
echo.
echo TIP: Next build will reuse the cached venv (fast).
echo      Run "build_win.bat %TARGET% --clean" to force full rebuild.
pause
