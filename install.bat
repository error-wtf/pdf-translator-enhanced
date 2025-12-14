@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1
REM ============================================================
REM PDF Translator - Windows Installation Script
REM 2025 Sven Kalinowski with small help of Lino Casu
REM Licensed under the Anti-Capitalist Software License v1.4
REM ============================================================

echo.
echo ============================================================
echo    PDF Translator - Installation
echo    Now with Marker support for scientific PDFs
echo ============================================================
echo.

REM Check for --clean flag to force venv recreation
if "%1"=="--clean" (
    echo Cleaning up old virtual environment...
    if exist "venv" rmdir /s /q venv
    echo Done. Will create fresh venv.
)

REM ============================================================
REM [1/11] Check Python
REM ============================================================
echo [1/11] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.10+ from https://python.org
    echo.
    echo Or with winget:
    echo   winget install Python.Python.3.12
    pause
    exit /b 1
)
python --version

REM ============================================================
REM [2/11] Check/Install Ollama
REM ============================================================
echo.
echo [2/11] Checking Ollama...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo Ollama not installed. Attempting installation via winget...
    winget --version >nul 2>&1
    if errorlevel 1 (
        echo WARNING: winget not available.
        echo Please install Ollama manually: https://ollama.ai
    ) else (
        winget install Ollama.Ollama --accept-package-agreements --accept-source-agreements
    )
) else (
    echo Ollama is installed.
    ollama --version
)

REM ============================================================
REM [3/11] Start Ollama service
REM ============================================================
echo.
echo [3/11] Starting Ollama service...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo Ollama is not running. Starting in background...
    start /B ollama serve >nul 2>&1
    timeout /t 3 /nobreak >nul
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if errorlevel 1 (
        echo WARNING: Could not start Ollama.
    ) else (
        echo Ollama started!
    )
) else (
    echo Ollama is already running.
)

REM ============================================================
REM [4/11] Create virtual environment
REM ============================================================
echo.
echo [4/11] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created.
) else (
    REM Check if venv is corrupted (temp files from failed pip installs)
    if exist "venv\Lib\site-packages\~*" (
        echo Virtual environment appears corrupted. Recreating...
        rmdir /s /q venv
        python -m venv venv
        echo Virtual environment recreated.
    ) else (
        echo Virtual environment already exists.
    )
)

REM ============================================================
REM [5/11] Activate venv
REM ============================================================
echo.
echo [5/11] Activating virtual environment...
call venv\Scripts\activate.bat

REM ============================================================
REM [6/11] Check/Install PyTorch with CUDA for GPU acceleration
REM ============================================================
echo.
echo [6/11] Checking PyTorch CUDA support for GPU acceleration...
echo.

REM Check for NVIDIA GPU first
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo No NVIDIA GPU detected. Will use CPU mode.
    echo   Marker will work but be slower (~5-10 min per PDF^)
    set "USE_CUDA=0"
) else (
    echo NVIDIA GPU detected! Installing PyTorch with CUDA support...
    echo This enables fast Marker extraction (~30-60 sec per PDF^)
    echo.
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo WARNING: PyTorch CUDA installation failed.
        set "USE_CUDA=0"
    ) else (
        set "USE_CUDA=1"
        echo PyTorch CUDA installed successfully!
    )
)

REM ============================================================
REM [7/11] Install remaining dependencies
REM ============================================================
echo.
echo [7/11] Upgrading pip and installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Re-install PyTorch CUDA if it was overwritten by marker-pdf
if "!USE_CUDA!"=="1" (
    echo.
    echo Checking if CUDA was overwritten...
    python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
    if errorlevel 1 (
        echo CUDA was disabled by marker-pdf. Re-installing PyTorch CUDA...
        echo.
        echo NOTE: This downloads ~2.4GB. Please wait...
        echo Cleaning up old torch installation completely...
        REM Uninstall ALL torch packages first
        pip uninstall -y torch torchvision torchaudio sympy 2>nul
        pip uninstall -y torch torchvision torchaudio 2>nul
        REM Force remove all torch-related directories
        for /d %%d in ("venv\Lib\site-packages\torch*") do rmdir /s /q "%%d" 2>nul
        for /d %%d in ("venv\Lib\site-packages\~orch*") do rmdir /s /q "%%d" 2>nul
        echo Installing PyTorch CUDA from scratch...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
        if errorlevel 1 (
            echo WARNING: PyTorch CUDA installation failed. Using CPU mode.
        ) else (
            echo PyTorch CUDA re-installed successfully!
        )
    ) else (
        echo CUDA is still working!
    )
)

REM ============================================================
REM [8/11] Check LaTeX
REM ============================================================
echo.
echo [8/11] Checking LaTeX...

REM First check if pdflatex is in PATH
where pdflatex >nul 2>&1
if not errorlevel 1 (
    echo LaTeX found in PATH!
    goto :latex_done
)

REM Check common MiKTeX locations
set "MIKTEX_PATH="
if exist "%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" (
    set "MIKTEX_PATH=%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64"
)
if exist "%PROGRAMFILES%\MiKTeX\miktex\bin\x64\pdflatex.exe" (
    set "MIKTEX_PATH=%PROGRAMFILES%\MiKTeX\miktex\bin\x64"
)
if exist "C:\MiKTeX\miktex\bin\x64\pdflatex.exe" (
    set "MIKTEX_PATH=C:\MiKTeX\miktex\bin\x64"
)

if defined MIKTEX_PATH (
    echo MiKTeX found at: !MIKTEX_PATH!
    set "PATH=!MIKTEX_PATH!;%PATH%"
    goto :latex_done
)

REM Not found - try to install
echo pdflatex not found. Installing MiKTeX...
winget install MiKTeX.MiKTeX --accept-package-agreements --accept-source-agreements
timeout /t 5 /nobreak >nul
echo.
echo NOTE: Please restart your terminal after installation.

:latex_done

REM ============================================================
REM [9/11] Check Pandoc
REM ============================================================
echo.
echo [9/11] Checking Pandoc...
pandoc --version >nul 2>&1
if errorlevel 1 (
    echo Pandoc not found. Installing...
    winget install JohnMacFarlane.Pandoc --accept-package-agreements --accept-source-agreements >nul 2>&1
    echo Pandoc installation attempted.
) else (
    echo Pandoc found!
)

REM ============================================================
REM [10/11] Download Marker Models (for scientific PDF extraction)
REM ============================================================
echo.
echo [10/11] Downloading Marker models for scientific PDF extraction...
echo.
echo This downloads ~2GB of AI models for PDF formula extraction.
echo This is OPTIONAL - skip with 'n' if you only use Standard mode.
echo.
set /p "DOWNLOAD_MARKER=Download Marker models now? [y/n]: "
if /i "!DOWNLOAD_MARKER!"=="y" (
    echo.
    python download_marker_models.py
    if errorlevel 1 (
        echo.
        echo WARNING: Marker model download failed.
        echo Models will be downloaded on first use instead.
    )
) else (
    echo Skipping Marker models. They will download on first use.
)

REM ============================================================
REM [11/11] Ollama Model Setup
REM ============================================================
echo.
echo [11/11] Ollama Model Setup...
echo.
echo ============================================================
echo    Ollama Model Setup
echo ============================================================
echo.

curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo Ollama not running. Skipping model setup.
    goto :model_done
)

echo Installed models:
ollama list 2>nul
echo.
echo ============================================================
echo    Recommended models for translation (all FREE, run locally):
echo ============================================================
echo.
echo    VRAM     Model              Size    Quality
echo    ----     -----              ----    -------
echo     4 GB    qwen2.5:3b         1.9 GB  Good
echo     8 GB    llama3.2:3b        2 GB    Good
echo    12 GB    qwen2.5:7b         4.4 GB  Very Good (recommended)
echo    16 GB    mistral-nemo:12b   7.1 GB  Excellent
echo    24 GB    qwen2.5:14b        9 GB    Excellent
echo.
echo Enter model name to download, u to update, or n to skip:
echo.
set /p "PULL_MODEL=Choice [qwen2.5:7b]: "

if "!PULL_MODEL!"=="" set "PULL_MODEL=qwen2.5:7b"
if /i "!PULL_MODEL!"=="n" (
    echo Skipping model setup.
    goto :model_done
)
if /i "!PULL_MODEL!"=="u" (
    set /p "UPDATE_MODEL=Which model to update? "
    if not "!UPDATE_MODEL!"=="" (
        echo Updating !UPDATE_MODEL!...
        ollama pull !UPDATE_MODEL!
    )
    goto :model_done
)

echo.
echo Downloading !PULL_MODEL!...
ollama pull !PULL_MODEL!

:model_done

REM ============================================================
REM Clear cache
REM ============================================================
echo.
echo Clearing Python cache...
if exist "__pycache__" rmdir /s /q "__pycache__" 2>nul
for /d %%d in (*) do (
    if exist "%%d\__pycache__" rmdir /s /q "%%d\__pycache__" 2>nul
)

REM ============================================================
REM Done
REM ============================================================
echo.
echo ============================================================
echo    Installation Complete!
echo ============================================================
echo.
echo Start the app with:
echo   run.bat
echo.
echo Or manually:
echo   venv\Scripts\activate.bat
echo   python gradio_app.py
echo.
echo ============================================================
echo.

pause
endlocal
