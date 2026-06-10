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

echo [1/3] Installing/Upgrading PyInstaller...
pip install --upgrade pyinstaller

echo [2/3] Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo [3/3] Compiling GUI application to EXE...
pyinstaller --noconsole --onefile --name="FaceRec_AI_Worker" --add-data "app;app" --add-data "facerec.proto;." ai_worker_gui.py

if %errorlevel% neq 0 (
    echo [ERROR] PyInstaller compilation failed!
    pause
    exit /b %errorlevel%
)

echo.
echo ====================================================
echo  Compilation successful!
echo  Your executable is located at: dist\FaceRec_AI_Worker.exe
echo ====================================================
pause
