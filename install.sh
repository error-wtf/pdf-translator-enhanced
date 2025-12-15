#!/bin/bash
# ============================================================
# PDF Translator Enhanced v2.0 - Linux/macOS Installation Script
# © 2025 Sven Kalinowski with small help of Lino Casu
# Licensed under the Anti-Capitalist Software License v1.4
# ============================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}   PDF Translator Enhanced v2.0 - Installation${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo ""
}

print_step() {
    echo -e "${YELLOW}[$1]${NC} $2"
}

print_ok() {
    echo -e "${GREEN}OK${NC} - $1"
}

print_error() {
    echo -e "${RED}ERROR:${NC} $1"
}

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Linux*)     OS="Linux";;
        Darwin*)    OS="macOS";;
        *)          OS="Unknown";;
    esac
    echo "$OS"
}

print_header
OS=$(detect_os)
echo -e "System: ${BOLD}$OS${NC}"
echo ""

# ============================================================
# [1/5] Check Python
# ============================================================
print_step "1/5" "Checking Python..."

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found!"
    echo ""
    if [ "$OS" = "Linux" ]; then
        echo -e "Install with: ${CYAN}sudo apt install python3 python3-venv python3-pip${NC}"
    elif [ "$OS" = "macOS" ]; then
        echo -e "Install with: ${CYAN}brew install python3${NC}"
    fi
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
print_ok "Python $PYTHON_VERSION found."

# ============================================================
# [2/5] Create Virtual Environment
# ============================================================
print_step "2/5" "Creating virtual environment..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_ok "Virtual environment created."
else
    print_ok "Virtual environment already exists."
fi

# Activate venv
source venv/bin/activate

# ============================================================
# [3/5] Install Dependencies
# ============================================================
print_step "3/5" "Installing Python dependencies..."

pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt

print_ok "Dependencies installed."

# ============================================================
# [4/5] Check/Install Ollama
# ============================================================
print_step "4/5" "Checking Ollama..."

if ! command -v ollama &> /dev/null; then
    echo ""
    echo -e "${YELLOW}Ollama not found.${NC}"
    echo ""
    read -p "Install Ollama now? (y/n) [y]: " INSTALL_OLLAMA
    INSTALL_OLLAMA=${INSTALL_OLLAMA:-y}
    
    if [[ "$INSTALL_OLLAMA" =~ ^[Yy]$ ]]; then
        echo "Installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
        print_ok "Ollama installed."
    else
        echo "Skipping Ollama installation."
        echo -e "Install later from: ${CYAN}https://ollama.ai${NC}"
    fi
else
    print_ok "Ollama found."
    
    # Start Ollama if not running
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Starting Ollama..."
        nohup ollama serve > /dev/null 2>&1 &
        sleep 3
    fi
fi

# ============================================================
# [5/5] Model Selection
# ============================================================
echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}   Select a Model to Download${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""
echo " SMALL (4 GB VRAM):"
echo "   1) gemma2:2b        - 1.6 GB - Basic, fast"
echo "   2) phi3:mini        - 2.3 GB - Good quality"
echo "   3) llama3.2:3b      - 2.0 GB - Good, 128K context"
echo ""
echo " MEDIUM (8 GB VRAM):"
echo "   4) mistral:7b       - 4.1 GB - Very good"
echo -e "   ${GREEN}5) qwen2.5:7b       - 4.4 GB - BEST for translations [RECOMMENDED]${NC}"
echo "   6) llama3.1:8b      - 4.7 GB - Very good, 128K context"
echo ""
echo " LARGE (16 GB VRAM):"
echo "   7) qwen2.5:14b      - 9.0 GB - Excellent, multilingual"
echo "   8) deepseek-coder-v2:16b - 9.0 GB - Technical/Code"
echo ""
echo " XL (24+ GB VRAM):"
echo "   9) mistral-small:22b - 13 GB - Premium quality"
echo "  10) qwen2.5:32b      - 19 GB - Top multilingual"
echo "  11) gpt-oss:20b      - 12 GB - Strong general"
echo ""
echo " CLOUD (No GPU needed):"
echo "  12) gpt-oss:120b-cloud     - Cloud, requires 'ollama signin'"
echo "  13) deepseek-v3.1:671b-cloud - Maximum quality, cloud"
echo ""
echo "   0) Skip model download"
echo ""

read -p "Enter choice (0-13) [5]: " MODEL_CHOICE
MODEL_CHOICE=${MODEL_CHOICE:-5}

case $MODEL_CHOICE in
    0) MODEL="";;
    1) MODEL="gemma2:2b";;
    2) MODEL="phi3:mini";;
    3) MODEL="llama3.2:3b";;
    4) MODEL="mistral:7b";;
    5) MODEL="qwen2.5:7b";;
    6) MODEL="llama3.1:8b";;
    7) MODEL="qwen2.5:14b";;
    8) MODEL="deepseek-coder-v2:16b";;
    9) MODEL="mistral-small:22b";;
    10) MODEL="qwen2.5:32b";;
    11) MODEL="gpt-oss:20b";;
    12) MODEL="gpt-oss:120b-cloud";;
    13) MODEL="deepseek-v3.1:671b-cloud";;
    *) MODEL="";;
esac

if [ -n "$MODEL" ]; then
    echo ""
    echo "Downloading $MODEL..."
    echo "This may take several minutes depending on model size."
    echo ""
    
    if ollama pull "$MODEL"; then
        print_ok "Model $MODEL downloaded successfully!"
    else
        echo ""
        echo -e "${YELLOW}WARNING:${NC} Model download failed."
        echo "For cloud models, run: ollama signin"
    fi
fi

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}   Installation Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "To start the translator, run:"
echo -e "  ${CYAN}./run.sh${NC}"
echo ""
echo "For public URL (share with others):"
echo -e "  ${CYAN}./run.sh --share${NC}"
echo ""
echo -e "${GREEN}============================================================${NC}"
echo ""
echo -e "${YELLOW}OPTIONAL: For OCR (text in images), install Tesseract:${NC}"
if [ "$OS" = "Linux" ]; then
    echo -e "  ${CYAN}sudo apt install tesseract-ocr tesseract-ocr-deu${NC}"
elif [ "$OS" = "macOS" ]; then
    echo -e "  ${CYAN}brew install tesseract tesseract-lang${NC}"
fi
echo ""
echo -e "${GREEN}============================================================${NC}"
echo ""
