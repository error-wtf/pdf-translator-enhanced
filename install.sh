#!/bin/bash
# ============================================================
# PDF Translator - Linux/macOS Installation Script
# 2025 Sven Kalinowski with small help of Lino Casu
# Licensed under the Anti-Capitalist Software License v1.4
# ============================================================

set -e

echo ""
echo "============================================================"
echo "   PDF Translator - Installation"
echo "   Now with Marker support for scientific PDFs"
echo "============================================================"
echo ""

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     PLATFORM=Linux;;
    Darwin*)    PLATFORM=Mac;;
    *)          PLATFORM="UNKNOWN"
esac
echo "Detected system: $PLATFORM"

# ============================================================
# [1/9] Check Python
# ============================================================
echo ""
echo "[1/9] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found!"
    echo "Please install Python 3.10+:"
    if [ "$PLATFORM" = "Linux" ]; then
        echo "  sudo apt install python3 python3-venv python3-pip"
    else
        echo "  brew install python3"
    fi
    exit 1
fi
python3 --version

# ============================================================
# [2/9] Check/Install Ollama
# ============================================================
echo ""
echo "[2/9] Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "Ollama not installed."
    read -p "Install Ollama now? (y/n): " INSTALL_OLLAMA
    if [[ "$INSTALL_OLLAMA" =~ ^[yY]$ ]]; then
        echo "Installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
    fi
else
    echo "Ollama is installed."
    ollama --version
fi

# ============================================================
# [3/9] Start Ollama service
# ============================================================
echo ""
echo "[3/9] Starting Ollama service..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama is already running."
else
    if command -v ollama &> /dev/null; then
        echo "Starting Ollama in background..."
        nohup ollama serve > /dev/null 2>&1 &
        sleep 3
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "Ollama started!"
        else
            echo "WARNING: Could not start Ollama."
        fi
    fi
fi

# ============================================================
# [4/9] Create virtual environment
# ============================================================
echo ""
echo "[4/9] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# ============================================================
# [5/9] Activate venv
# ============================================================
echo ""
echo "[5/9] Activating virtual environment..."
source venv/bin/activate

# ============================================================
# [6/9] Install dependencies
# ============================================================
echo ""
echo "[6/9] Upgrading pip and installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# ============================================================
# [7/9] Check LaTeX
# ============================================================
echo ""
echo "[7/9] Checking LaTeX..."
if ! command -v pdflatex &> /dev/null; then
    echo "pdflatex not found."
    read -p "Install LaTeX now? (y/n): " INSTALL_LATEX
    if [[ "$INSTALL_LATEX" =~ ^[yY]$ ]]; then
        if [ "$PLATFORM" = "Linux" ]; then
            echo "Installing TeX Live..."
            sudo apt update
            sudo apt install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended
        elif [ "$PLATFORM" = "Mac" ]; then
            if command -v brew &> /dev/null; then
                echo "Installing BasicTeX..."
                brew install --cask basictex
            fi
        fi
    fi
else
    echo "LaTeX found!"
fi

# ============================================================
# [8/9] Check Pandoc
# ============================================================
echo ""
echo "[8/9] Checking Pandoc..."
if ! command -v pandoc &> /dev/null; then
    echo "Pandoc not found."
    if [ "$PLATFORM" = "Linux" ]; then
        sudo apt install -y pandoc 2>/dev/null || echo "Pandoc installation skipped."
    elif [ "$PLATFORM" = "Mac" ]; then
        brew install pandoc 2>/dev/null || echo "Pandoc installation skipped."
    fi
else
    echo "Pandoc found!"
fi

# ============================================================
# [9/9] Ollama Model Setup
# ============================================================
echo ""
echo "[9/9] Ollama Model Setup..."
echo ""
echo "============================================================"
echo "   Ollama Model Setup"
echo "============================================================"
echo ""

if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama not running. Skipping model setup."
else
    echo "Installed models:"
    ollama list 2>/dev/null || echo "  (none)"
    echo ""
    echo "Recommended models for translation:"
    echo "   8 GB VRAM:  llama3.2:3b      2 GB"
    echo "  12 GB VRAM:  qwen2.5:7b       4.4 GB  BEST"
    echo "  16 GB VRAM:  mistral-nemo:12b 7.1 GB"
    echo "  24 GB VRAM:  qwen2.5:14b      9 GB"
    echo ""
    echo "Enter model name to download, u to update, or n to skip:"
    read -p "Choice [qwen2.5:7b]: " PULL_MODEL
    
    if [ -z "$PULL_MODEL" ]; then
        PULL_MODEL="qwen2.5:7b"
    fi
    
    if [[ "$PULL_MODEL" =~ ^[nN]$ ]]; then
        echo "Skipping model setup."
    elif [[ "$PULL_MODEL" =~ ^[uU]$ ]]; then
        read -p "Which model to update? " UPDATE_MODEL
        if [ -n "$UPDATE_MODEL" ]; then
            echo "Updating $UPDATE_MODEL..."
            ollama pull "$UPDATE_MODEL"
        fi
    else
        echo "Downloading $PULL_MODEL..."
        ollama pull "$PULL_MODEL"
    fi
fi

# ============================================================
# Clear cache
# ============================================================
echo ""
echo "Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# ============================================================
# Done
# ============================================================
echo ""
echo "============================================================"
echo "   Installation Complete!"
echo "============================================================"
echo ""
echo "Start the app with:"
echo "  ./run.sh"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  python gradio_app.py"
echo ""
echo "============================================================"
echo ""
