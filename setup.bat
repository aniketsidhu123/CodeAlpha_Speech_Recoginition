@echo off
setlocal enabledelayedexpansion
title Setup - Speech Emotion Recognition (SER) System

echo ======================================================================
echo    🎙️  SPEECH EMOTION RECOGNITION (SER) - ENVIRONMENT SETUP
echo ======================================================================
echo.

:: 1. Check Python installation
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not added to PATH.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do set PYTHON_VER=%%i
echo [OK] Found %PYTHON_VER%
echo.

:: 2. Create and prepare project directories
echo [2/5] Creating project directories...
if not exist "data\raw" mkdir "data\raw"
if not exist "data\processed" mkdir "data\processed"
if not exist "models" mkdir "models"
if not exist "reports" mkdir "reports"
if not exist "demo_samples" mkdir "demo_samples"
echo [OK] Required directories verified.
echo.

:: 3. Optional Virtual Environment
echo [3/5] Setting up virtual environment...
set /p USE_VENV="Do you want to create/use a dedicated virtual environment (venv)? (y/n, default: y): "
if /i "%USE_VENV%"=="" set USE_VENV=y

if /i "%USE_VENV%"=="y" (
    if not exist "venv\Scripts\activate.bat" (
        echo Creating virtual environment 'venv'...
        python -m venv venv
        if %ERRORLEVEL% NEQ 0 (
            echo [WARNING] Failed to create venv. Proceeding with global python...
        ) else (
            echo [OK] Virtual environment created.
        )
    )
    if exist "venv\Scripts\activate.bat" (
        echo Activating virtual environment...
        call venv\Scripts\activate.bat
    )
) else (
    echo Using system Python environment.
)
echo.

:: 4. Upgrade pip and install requirements
echo [4/5] Installing dependencies from requirements.txt...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [WARNING] Some packages failed to install. Please review errors above.
) else (
    echo [OK] All dependencies installed successfully.
)
echo.

:: 5. Validate core libraries
echo [5/5] Validating installation...
python -c "import tensorflow, librosa, soundfile, sklearn, fastapi, streamlit; print('[OK] Core libraries (TensorFlow, Librosa, Soundfile, Scikit-learn, FastAPI, Streamlit) verified successfully!')"
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Verification encountered an issue. Review the above message.
) else (
    echo [SUCCESS] Environment setup is complete!
)

echo.
echo ======================================================================
echo    SETUP COMPLETE!
echo    You can now double-click 'run.bat' to launch the web app or train models.
echo ======================================================================
echo.
pause
