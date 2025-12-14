@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1
REM ============================================================
REM PDF Translator - Windows Start Script
REM 2025 Sven Kalinowski with small help of Lino Casu
REM Licensed under the Anti-Capitalist Software License v1.4
REM ============================================================

title PDF Translator

echo.
echo ============================================================
echo    PDF Translator - Starting
echo ============================================================
echo.

REM Check for --share flag
set "SHARE_FLAG="
if "%1"=="--share" set "SHARE_FLAG=--share"
if "%1"=="-s" set "SHARE_FLAG=--share"

REM ============================================================
REM [1/5] Check Virtual Environment
REM ============================================================
echo [1/5] Checking virtual environment...
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run install.bat first.
    pause
    exit /b 1
)
echo OK - Virtual environment found.

REM ============================================================
REM [2/5] Clear Python Cache
REM ============================================================
echo [2/5] Clearing Python cache...
set "CACHE_COUNT=0"
if exist "__pycache__" (
    rmdir /s /q "__pycache__" 2>nul
    set /a CACHE_COUNT+=1
)
for /d %%d in (*) do (
    if exist "%%d\__pycache__" (
        rmdir /s /q "%%d\__pycache__" 2>nul
        set /a CACHE_COUNT+=1
    )
)
echo OK - Cleared !CACHE_COUNT! cache directories.

REM ============================================================
REM [3/5] Setup LaTeX PATH
REM ============================================================
echo [3/5] Checking LaTeX...

where pdflatex >nul 2>&1
if not errorlevel 1 (
    echo OK - LaTeX found in PATH.
    goto :latex_done
)

if exist "%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" (
    set "PATH=%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64;%PATH%"
    echo OK - MiKTeX added to PATH.
    goto :latex_done
)
if exist "%PROGRAMFILES%\MiKTeX\miktex\bin\x64\pdflatex.exe" (
    set "PATH=%PROGRAMFILES%\MiKTeX\miktex\bin\x64;%PATH%"
    echo OK - MiKTeX added to PATH.
    goto :latex_done
)
if exist "C:\MiKTeX\miktex\bin\x64\pdflatex.exe" (
    set "PATH=C:\MiKTeX\miktex\bin\x64;%PATH%"
    echo OK - MiKTeX added to PATH.
    goto :latex_done
)

echo WARNING: LaTeX not found. PDF generation may fail.

:latex_done

REM ============================================================
REM [4/5] Check/Start Ollama
REM ============================================================
echo [4/5] Checking Ollama service...

call venv\Scripts\activate.bat >nul 2>&1

curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo Ollama not running. Starting...
    start "" /B ollama serve >nul 2>&1
    timeout /t 3 /nobreak >nul
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if errorlevel 1 (
        echo WARNING: Could not start Ollama.
    ) else (
        echo OK - Ollama started.
    )
) else (
    echo OK - Ollama is running.
)

REM ============================================================
REM [5/5] Free Port and Start Gradio
REM ============================================================
echo [5/5] Starting Gradio server...

REM Kill any process using port 7860
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":7860" ^| findstr "LISTENING"') do (
    echo Freeing port 7860 - killing PID %%a
    taskkill /F /PID %%a >nul 2>&1
    timeout /t 2 /nobreak >nul
)

echo.
echo ============================================================
echo    Application Starting
echo ============================================================
echo.
if defined SHARE_FLAG (
    echo   Local URL:  http://127.0.0.1:7860
    echo   Public URL: Will be generated...
) else (
    echo   URL:  http://127.0.0.1:7860
    echo.
    echo   TIP: Use 'run.bat --share' for a public URL
)
echo.
echo   Press Ctrl+C to stop the server.
echo.
echo ============================================================
echo.

python gradio_app.py %SHARE_FLAG%

if errorlevel 1 (
    echo.
    echo ============================================================
    echo    Application Error
    echo ============================================================
    echo.
    echo Common fixes:
    echo   1. Run install.bat to reinstall dependencies
    echo   2. Check if port 7860 is free
    echo   3. Restart your terminal
    echo.
)

pause
endlocal
