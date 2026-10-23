@echo off
setlocal enabledelayedexpansion
title Speech Emotion Recognition (SER) System

:: Activate virtual environment if present
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

:: Check CLI argument
if not "%~1"=="" (
    if /i "%~1"=="web" goto START_WEB
    if /i "%~1"=="serve" goto START_WEB
    if /i "%~1"=="streamlit" goto START_STREAMLIT
    if /i "%~1"=="train" goto TRAIN_CNN
    if /i "%~1"=="train-lstm" goto TRAIN_LSTM
    if /i "%~1"=="train-crnn" goto TRAIN_CRNN
    if /i "%~1"=="eval" goto EVALUATE
    if /i "%~1"=="preprocess" goto PREPROCESS
    if /i "%~1"=="predict" goto PREDICT_CUSTOM
)

:MENU
cls
echo ======================================================================
echo    🎙️  SPEECH EMOTION RECOGNITION (SER) - CONTROL CENTER
echo ======================================================================
echo    Deep Learning & Signal Processing on RAVDESS (MFCCs / Mel-Spectrogram)
echo ======================================================================
echo.
echo   [1] Start Modern Web UI (FastAPI + Glassmorphism Interface)
echo   [2] Start Streamlit Interactive Dashboard
echo   [3] Preprocess Dataset (Extract MFCCs & Acoustic Features)
echo   [4] Train 1D CNN Model (Deep Conv1D)
echo   [5] Train BiLSTM Model (Bidirectional Recurrent Neural Network)
echo   [6] Train Hybrid CRNN with Self-Attention Model
echo   [7] Evaluate Model (Generate Confusion Matrix Heatmap & Report)
echo   [8] Predict Emotion on RAVDESS Demo Sample
echo   [9] Predict Emotion on Custom Audio File
echo   [10] Exit
echo.
echo ======================================================================
set /p CHOICE="Enter your choice [1-10]: "

if "%CHOICE%"=="1" goto START_WEB
if "%CHOICE%"=="2" goto START_STREAMLIT
if "%CHOICE%"=="3" goto PREPROCESS
if "%CHOICE%"=="4" goto TRAIN_CNN
if "%CHOICE%"=="5" goto TRAIN_LSTM
if "%CHOICE%"=="6" goto TRAIN_CRNN
if "%CHOICE%"=="7" goto EVALUATE
if "%CHOICE%"=="8" goto PREDICT_DEMO
if "%CHOICE%"=="9" goto PREDICT_CUSTOM
if "%CHOICE%"=="10" goto EXIT_PROMPT

echo [ERROR] Invalid selection, please try again.
timeout /t 2 >nul
goto MENU

:START_WEB
echo.
echo ======================================================================
echo Starting FastAPI Web Application at http://127.0.0.1:8000
echo ======================================================================
start http://127.0.0.1:8000
python main.py serve --port 8000
pause
goto MENU

:START_STREAMLIT
echo.
echo ======================================================================
echo Launching Streamlit Interactive Dashboard at http://localhost:8501
echo ======================================================================
python main.py streamlit --port 8501
pause
goto MENU

:PREPROCESS
echo.
echo ======================================================================
echo Extracting MFCCs, Chroma, Mel-Spectrogram, ZCR, RMS from RAVDESS...
echo ======================================================================
python main.py preprocess --force
echo.
pause
goto MENU

:TRAIN_CNN
echo.
echo ======================================================================
echo Training Deep 1D CNN Model on Acoustic Features...
echo ======================================================================
python main.py train --model cnn --epochs 40 --batch-size 32
echo.
pause
goto MENU

:TRAIN_LSTM
echo.
echo ======================================================================
echo Training Bidirectional LSTM Sequential Model...
echo ======================================================================
python main.py train --model lstm --epochs 40 --batch-size 32
echo.
pause
goto MENU

:TRAIN_CRNN
echo.
echo ======================================================================
echo Training Hybrid CRNN with Attention Model...
echo ======================================================================
python main.py train --model crnn --epochs 40 --batch-size 32
echo.
pause
goto MENU

:EVALUATE
echo.
echo ======================================================================
echo Evaluating Model on Unseen Test Set & Generating Confusion Matrix...
echo ======================================================================
python main.py evaluate
echo.
pause
goto MENU

:PREDICT_DEMO
echo.
echo Available Demo Emotion Samples:
echo   1. Happy
echo   2. Angry
echo   3. Sad
echo   4. Calm
echo   5. Fearful
echo   6. Disgust
echo   7. Surprised
echo   8. Neutral
set /p DEMO_CHOICE="Select emotion number to test [1-8]: "
if "%DEMO_CHOICE%"=="1" set SAMPLE_FILE=demo_samples\happy.wav
if "%DEMO_CHOICE%"=="2" set SAMPLE_FILE=demo_samples\angry.wav
if "%DEMO_CHOICE%"=="3" set SAMPLE_FILE=demo_samples\sad.wav
if "%DEMO_CHOICE%"=="4" set SAMPLE_FILE=demo_samples\calm.wav
if "%DEMO_CHOICE%"=="5" set SAMPLE_FILE=demo_samples\fearful.wav
if "%DEMO_CHOICE%"=="6" set SAMPLE_FILE=demo_samples\disgust.wav
if "%DEMO_CHOICE%"=="7" set SAMPLE_FILE=demo_samples\surprised.wav
if "%DEMO_CHOICE%"=="8" set SAMPLE_FILE=demo_samples\neutral.wav

if not defined SAMPLE_FILE (
    echo [ERROR] Invalid sample selection.
    pause
    goto MENU
)

python main.py predict --file "%SAMPLE_FILE%"
echo.
pause
goto MENU

:PREDICT_CUSTOM
echo.
set /p USER_AUDIO="Enter the full path to your audio file (.wav or .mp3): "
if not exist "%USER_AUDIO%" (
    echo [ERROR] File does not exist: %USER_AUDIO%
    pause
    goto MENU
)
python main.py predict --file "%USER_AUDIO%"
echo.
pause
goto MENU

:EXIT_PROMPT
echo Thank you for using Speech Emotion Recognition!
exit /b 0
