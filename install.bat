@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1
REM ============================================================
REM PDF Translator Enhanced v2.0 - Windows Installation Script
REM © 2025 Sven Kalinowski with small help of Lino Casu
REM Licensed under the Anti-Capitalist Software License v1.4
REM ============================================================

title PDF Translator - Installation

echo.
echo ============================================================
echo    PDF Translator Enhanced v2.0 - Installation
echo ============================================================
echo.

REM ============================================================
REM [1/5] Check Python
REM ============================================================
echo [1/5] Checking Python...
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
echo OK - Python %PYTHON_VERSION% found.

REM ============================================================
REM [2/5] Create Virtual Environment
REM ============================================================
echo [2/5] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo OK - Virtual environment created.
) else (
    echo OK - Virtual environment already exists.
)

REM Activate venv
call venv\Scripts\activate.bat

REM ============================================================
REM [3/5] Install Dependencies
REM ============================================================
echo [3/5] Installing Python dependencies...
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
echo OK - Dependencies installed.

REM ============================================================
REM [4/5] Check/Install Ollama
REM ============================================================
echo [4/5] Checking Ollama...
where ollama >nul 2>&1
if errorlevel 1 (
    echo.
    echo Ollama not found. Please install from:
    echo   https://ollama.ai/download
    echo.
    echo After installing, run this script again.
    echo.
) else (
    echo OK - Ollama found.
    
    REM Start Ollama if not running
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if errorlevel 1 (
        echo Starting Ollama...
        start "" /B ollama serve >nul 2>&1
        timeout /t 3 /nobreak >nul
    )
)

REM ============================================================
REM [5/5] Model Selection
REM ============================================================
echo.
echo ============================================================
echo    Select a Model to Download
echo ============================================================
echo.
echo  SMALL (4 GB VRAM):
echo    1) gemma2:2b        - 1.6 GB - Basic, fast
echo    2) phi3:mini        - 2.3 GB - Good quality
echo    3) llama3.2:3b      - 2.0 GB - Good, 128K context
echo.
echo  MEDIUM (8 GB VRAM):
echo    4) mistral:7b       - 4.1 GB - Very good
echo    5) qwen2.5:7b       - 4.4 GB - BEST for translations [RECOMMENDED]
echo    6) llama3.1:8b      - 4.7 GB - Very good, 128K context
echo.
echo  LARGE (16 GB VRAM):
echo    7) qwen2.5:14b      - 9.0 GB - Excellent, multilingual
echo    8) deepseek-coder-v2:16b - 9.0 GB - Technical/Code
echo.
echo  XL (24+ GB VRAM):
echo    9) mistral-small:22b - 13 GB - Premium quality
echo   10) qwen2.5:32b      - 19 GB - Top multilingual
echo   11) gpt-oss:20b      - 12 GB - Strong general
echo.
echo  CLOUD (No GPU needed):
echo   12) gpt-oss:120b-cloud     - Cloud, requires 'ollama signin'
echo   13) deepseek-v3.1:671b-cloud - Maximum quality, cloud
echo.
echo    0) Skip model download
echo.

set /p MODEL_CHOICE="Enter choice (0-13) [5]: "
if "%MODEL_CHOICE%"=="" set MODEL_CHOICE=5

if "%MODEL_CHOICE%"=="0" goto :skip_model
if "%MODEL_CHOICE%"=="1" set MODEL=gemma2:2b
if "%MODEL_CHOICE%"=="2" set MODEL=phi3:mini
if "%MODEL_CHOICE%"=="3" set MODEL=llama3.2:3b
if "%MODEL_CHOICE%"=="4" set MODEL=mistral:7b
if "%MODEL_CHOICE%"=="5" set MODEL=qwen2.5:7b
if "%MODEL_CHOICE%"=="6" set MODEL=llama3.1:8b
if "%MODEL_CHOICE%"=="7" set MODEL=qwen2.5:14b
if "%MODEL_CHOICE%"=="8" set MODEL=deepseek-coder-v2:16b
if "%MODEL_CHOICE%"=="9" set MODEL=mistral-small:22b
if "%MODEL_CHOICE%"=="10" set MODEL=qwen2.5:32b
if "%MODEL_CHOICE%"=="11" set MODEL=gpt-oss:20b
if "%MODEL_CHOICE%"=="12" set MODEL=gpt-oss:120b-cloud
if "%MODEL_CHOICE%"=="13" set MODEL=deepseek-v3.1:671b-cloud

if not defined MODEL (
    echo Invalid choice. Skipping model download.
    goto :skip_model
)

echo.
echo Downloading %MODEL%...
echo This may take several minutes depending on model size.
echo.

ollama pull %MODEL%

if errorlevel 1 (
    echo.
    echo WARNING: Model download failed.
    echo For cloud models, run: ollama signin
) else (
    echo.
    echo OK - Model %MODEL% downloaded successfully!
)

:skip_model

echo.
echo ============================================================
echo    Installation Complete!
echo ============================================================
echo.
echo To start the translator, run:
echo   run.bat
echo.
echo For public URL (share with others):
echo   run.bat --share
echo.
echo ============================================================
echo.

pause
endlocal
