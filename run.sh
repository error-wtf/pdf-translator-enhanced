#!/bin/bash
# ============================================================
# PDF Translator Enhanced v2.0 - Linux/macOS Start Script
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
NC='\033[0m' # No Color

# Parse arguments
SHARE_FLAG=""
if [ "$1" = "--share" ] || [ "$1" = "-s" ]; then
    SHARE_FLAG="--share"
fi

# Helper functions
print_header() {
    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}   PDF Translator v2.0 - Starting${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo ""
}

print_step() {
    echo -e "${YELLOW}[$1]${NC} $2"
}

print_ok() {
    echo -e "${GREEN}OK${NC} - $1"
}

print_warn() {
    echo -e "${YELLOW}WARNING:${NC} $1"
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
# [1/5] Check Virtual Environment
# ============================================================
print_step "1/5" "Checking virtual environment..."

if [ ! -f "venv/bin/activate" ]; then
    print_error "Virtual environment not found!"
    echo ""
    echo -e "Please run ${CYAN}./install.sh${NC} first."
    echo ""
    exit 1
fi
print_ok "Virtual environment found."

# ============================================================
# [2/5] Clear Python Cache
# ============================================================
print_step "2/5" "Clearing Python cache..."

CACHE_COUNT=$(find . -type d -name "__pycache__" 2>/dev/null | wc -l)
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

print_ok "Cleared $CACHE_COUNT cache directories."

# ============================================================
# [3/5] Check LaTeX
# ============================================================
print_step "3/5" "Checking LaTeX..."

if command -v pdflatex &> /dev/null; then
    LATEX_VERSION=$(pdflatex --version 2>/dev/null | head -1)
    print_ok "LaTeX found: $LATEX_VERSION"
else
    print_warn "LaTeX not found. PDF generation may fail."
    if [ "$OS" = "Linux" ]; then
        echo -e "         Install with: ${CYAN}sudo apt install texlive-latex-base${NC}"
    elif [ "$OS" = "macOS" ]; then
        echo -e "         Install with: ${CYAN}brew install --cask basictex${NC}"
    fi
fi

# ============================================================
# [4/5] Check/Start Ollama
# ============================================================
print_step "4/5" "Checking Ollama service..."

# Activate venv
source venv/bin/activate

check_ollama() {
    curl -s http://localhost:11434/api/tags > /dev/null 2>&1
}

if check_ollama; then
    print_ok "Ollama is running."
else
    if command -v ollama &> /dev/null; then
        echo "     Ollama not running. Starting in background..."
        nohup ollama serve > /dev/null 2>&1 &
        OLLAMA_PID=$!
        
        # Wait for Ollama (max 10 seconds)
        OLLAMA_READY=0
        for i in {1..10}; do
            sleep 1
            if check_ollama; then
                OLLAMA_READY=1
                break
            fi
        done
        
        if [ "$OLLAMA_READY" = "1" ]; then
            print_ok "Ollama started successfully (PID: $OLLAMA_PID)."
        else
            print_warn "Could not start Ollama."
            echo -e "         Local LLM translation not available."
        fi
    else
        print_warn "Ollama not installed."
        echo -e "         Install from: ${CYAN}https://ollama.ai${NC}"
        if [ "$OS" = "Linux" ]; then
            echo -e "         Or run: ${CYAN}curl -fsSL https://ollama.ai/install.sh | sh${NC}"
        elif [ "$OS" = "macOS" ]; then
            echo -e "         Or run: ${CYAN}brew install ollama${NC}"
        fi
    fi
fi

# ============================================================
# [5/5] Free Port 7860 and Start Gradio
# ============================================================
print_step "5/5" "Starting Gradio server..."

# Kill any process using port 7860
PORT_KILLED=0
if command -v lsof &> /dev/null; then
    PID=$(lsof -ti:7860 2>/dev/null || true)
    if [ -n "$PID" ]; then
        echo "     Freeing port 7860 (killing PID: $PID)..."
        kill -9 $PID 2>/dev/null || true
        PORT_KILLED=1
    fi
elif command -v fuser &> /dev/null; then
    if fuser 7860/tcp 2>/dev/null; then
        echo "     Freeing port 7860..."
        fuser -k 7860/tcp 2>/dev/null || true
        PORT_KILLED=1
    fi
elif command -v ss &> /dev/null; then
    PID=$(ss -tlnp 2>/dev/null | grep ':7860' | grep -oP '(?<=pid=)\d+' || true)
    if [ -n "$PID" ]; then
        echo "     Freeing port 7860 (killing PID: $PID)..."
        kill -9 $PID 2>/dev/null || true
        PORT_KILLED=1
    fi
fi

if [ "$PORT_KILLED" = "1" ]; then
    sleep 2
fi

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}   Application Starting${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
if [ -n "$SHARE_FLAG" ]; then
    echo -e "   Local URL:  ${CYAN}http://127.0.0.1:7860${NC}"
    echo -e "   Public URL: ${CYAN}Will be generated...${NC}"
else
    echo -e "   URL:  ${CYAN}http://127.0.0.1:7860${NC}"
    echo ""
    echo -e "   TIP: Use ${YELLOW}./run.sh --share${NC} for a public URL"
fi
echo ""
echo -e "   Press ${YELLOW}Ctrl+C${NC} to stop the server."
echo ""
echo -e "${GREEN}============================================================${NC}"
echo ""

# Start Gradio with error handling
python gradio_app.py $SHARE_FLAG
EXIT_CODE=$?

if [ "$EXIT_CODE" != "0" ]; then
    echo ""
    echo -e "${RED}============================================================${NC}"
    echo -e "${RED}   Application Error (Exit Code: $EXIT_CODE)${NC}"
    echo -e "${RED}============================================================${NC}"
    echo ""
    echo "Common fixes:"
    echo -e "  1. Run ${CYAN}./install.sh${NC} to reinstall dependencies"
    echo "  2. Check if port 7860 is free"
    echo "  3. Restart your terminal"
    echo ""
    exit $EXIT_CODE
fi
