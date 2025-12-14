# 📄 PDF Translator (Enhanced Fork)

**Translate scientific PDFs while preserving LaTeX formulas and document structure**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/error-wtf/pdf-translator-enhanced/blob/main/PDF_Translator_Colab.ipynb)

> 🔗 **Based on:** [thelanguagenerd/pdf-translator](https://github.com/thelanguagenerd/pdf-translator)  
> This is an enhanced fork with improved formula preservation, table detection, and multi-language support.

© 2025 Sven Kalinowski with small help of Lino Casu  
Licensed under the **Anti-Capitalist Software License v1.4**

---

## 🆕 What's New in This Fork

- **🔬 Unified Pipeline** - Combines Marker + PyMuPDF for best extraction quality
- **📊 Table Detection** - Automatically detects and preserves table structure
- **🖼️ Caption Anchoring** - Figures and captions stay together
- **🧹 Text Normalization** - Removes garbage characters (￾, soft hyphens, etc.)
- **📐 Enhanced Formula Protection** - 60+ LaTeX patterns protected during translation
- **🌍 20 Languages** - Now includes Arabic, Hebrew, Ukrainian, Hindi, and more
- **✅ Regression Tests** - Automated quality checks for translated PDFs
- **☁️ Ollama Cloud Support** - Run large models (up to 671B) without local GPU

---

## 🌟 Features

- **🔒 100% Local Processing** - With Ollama local models, no data leaves your computer
- **☁️ Ollama Cloud Option** - Use huge models without GPU via `ollama signin`
- **🧠 AI-Powered Translation** - Uses state-of-the-art LLMs (Llama, Mistral, GPT-4)
- **📐 LaTeX Preservation** - Mathematical formulas remain intact
- **🎨 Beautiful UI** - Modern Gradio interface with dark/light mode
- **🔧 Auto GPU Detection** - Automatically detects your VRAM and suggests suitable models
- **🌍 20 Languages** - English, German, French, Spanish, Italian, Japanese, Chinese, Arabic, Hebrew, Ukrainian, and more

---

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
2. [System Requirements](#-system-requirements)
3. [Detailed Installation](#-detailed-installation)
4. [Running the Application](#-running-the-application)
5. [Using the Application](#-using-the-application)
6. [LLM Backend Options](#-llm-backend-options)
7. [VRAM & Model Guide](#-vram--model-guide)
8. [API Reference](#-api-reference)
9. [Troubleshooting](#-troubleshooting)
10. [Project Structure](#-project-structure)
11. [Security](#-security)
12. [License](#-license)

---

## 🚀 Quick Start

### Windows (One-Click Install)

1. **Download or clone this repository**
2. **Double-click `install.bat`**
3. **Wait for installation to complete** (5-10 minutes)
4. **Double-click `run.bat`** to start the application
5. **Open http://127.0.0.1:7860** in your browser

### Linux / macOS (One-Click Install)

```bash
# Clone the repository
git clone https://github.com/error-wtf/pdf-translator-enhanced.git
cd pdf-translator-enhanced

# Make scripts executable
chmod +x install.sh run.sh

# Install everything
./install.sh

# Start the application
./run.sh
```

---

## 💻 System Requirements

### Minimum Requirements

| Component | Minimum | Notes |
|-----------|---------|-------|
| **Operating System** | Windows 10/11, Ubuntu 20.04+, macOS 12+ | 64-bit required |
| **Python** | 3.10 or higher | 3.11+ recommended |
| **RAM** | 8 GB | For small models only |
| **GPU VRAM** | 4 GB | NVIDIA recommended |
| **Storage** | 10 GB free | More for larger models |
| **LaTeX** | TeX Live or MiKTeX | Required for PDF output |

### Recommended Requirements

| Component | Recommended | Notes |
|-----------|-------------|-------|
| **Python** | 3.11 or 3.12 | Best performance |
| **RAM** | 16 GB or more | For larger models |
| **GPU VRAM** | 8-24 GB | NVIDIA RTX series |
| **Storage** | 50 GB free | For multiple models |

### GPU Compatibility

| GPU Brand | Support Level | Notes |
|-----------|---------------|-------|
| **NVIDIA** | ✅ Full Support | Best performance, auto-detected |
| **AMD** | ⚠️ Partial | ROCm required on Linux |
| **Apple Silicon** | ✅ Good | M1/M2/M3 unified memory |
| **Intel** | ⚠️ Limited | CPU fallback available |
| **No GPU** | ☁️ Ollama Cloud | Use cloud models with `ollama signin` |

---

## 📦 Detailed Installation

### Step 1: Install Python

#### Windows
```powershell
# Option A: Using winget (recommended)
winget install Python.Python.3.11

# Option B: Download from python.org
# Go to https://www.python.org/downloads/
# Download Python 3.11 or 3.12
# Run installer, CHECK "Add Python to PATH"
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

#### macOS
```bash
# Using Homebrew
brew install python@3.11
```

**Verify installation:**
```bash
python --version
# Should show: Python 3.11.x or higher
```

---

### Step 2: Install Ollama (Recommended for Local AI)

Ollama allows you to run AI models locally without sending data to the cloud.

#### Windows
```powershell
# Using winget
winget install Ollama.Ollama

# Or download from https://ollama.ai/download
```

#### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### macOS
```bash
brew install ollama
```

**Start Ollama service:**
```bash
ollama serve
```

**Download a model:**
```bash
# Recommended for 16GB VRAM
ollama pull llama3.1:8b

# For 8GB VRAM
ollama pull llama3.2:3b

# For 24GB+ VRAM
ollama pull mistral-small:22b
```

---

### Step 3: Install LaTeX

LaTeX is required to compile the translated PDF output.

#### Windows
```powershell
# Using winget (recommended)
winget install MiKTeX.MiKTeX

# After installation, open MiKTeX Console and:
# 1. Click "Check for updates"
# 2. Install all updates
# 3. Restart your terminal
```

**Alternative: TeX Live**
- Download from https://tug.org/texlive/
- Run the installer (takes 30-60 minutes)

#### Linux (Ubuntu/Debian)
```bash
# Basic installation (smaller, ~500MB)
sudo apt install texlive-latex-base texlive-latex-extra

# Full installation (larger, ~3GB, recommended)
sudo apt install texlive-full

# For multilingual support
sudo apt install texlive-lang-english texlive-lang-german texlive-lang-french
```

#### macOS
```bash
# Basic installation
brew install --cask basictex

# Full installation (recommended)
brew install --cask mactex
```

**Verify installation:**
```bash
pdflatex --version
# Should show version information
```

---

### Step 4: Set Up the Application

#### Automatic Setup (Recommended)

**Windows:**
```batch
install.bat
```

**Linux/macOS:**
```bash
./install.sh
```

The install script will:
1. ✅ Create a Python virtual environment
2. ✅ Install all Python dependencies
3. ✅ Check for Ollama and offer to install it
4. ✅ Start Ollama service
5. ✅ Check for LaTeX and offer to install it
6. ✅ Download an AI model (interactive selection)

#### Manual Setup

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "import gradio; print('Gradio OK')"
python -c "import requests; print('Requests OK')"
```

---

## ▶️ Running the Application

### Method 1: Using Run Scripts (Recommended)

**Windows:**
```batch
run.bat
```

**Linux/macOS:**
```bash
./run.sh
```

The run script will:
1. Activate the virtual environment
2. Check if Ollama is running (start if needed)
3. Add MiKTeX to PATH if needed
4. Launch the Gradio web interface
5. Open your browser automatically

### Method 2: Manual Start

```bash
# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Start Ollama (in a separate terminal)
ollama serve

# Start the application
python gradio_app.py
```

### Method 3: With Share URL (Access from Other Devices)

```bash
python gradio_app.py --share
```

This generates a public URL like `https://abc123.gradio.live` that:
- Is valid for 72 hours
- Can be accessed from any device
- Requires your PC to stay on

### Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--share` | Generate public URL for remote access | `python gradio_app.py --share` |
| `--port` | Use a different port (default: 7860) | `python gradio_app.py --port 8080` |
| `--no-browser` | Don't open browser automatically | `python gradio_app.py --no-browser` |

### Access the Application

After starting, open your browser and go to:
- **Local:** http://127.0.0.1:7860
- **Network:** http://YOUR_IP:7860 (if using `--share` or `0.0.0.0`)

---

## 📖 Using the Application

### Step-by-Step Translation Guide

#### 1. Upload Your PDF

- Click the **"Upload PDF"** area or drag & drop your file
- Supported: Scientific papers, articles, documentation
- Maximum size: 50 MB
- Best results with text-based PDFs (not scanned images)

#### 2. Select Target Language

Choose from 20 languages:

| Language | Code | | Language | Code |
|----------|------|-|----------|------|
| 🇬🇧 English | en | | 🇵🇱 Polish | pl |
| 🇩🇪 German | de | | 🇹🇷 Turkish | tr |
| 🇫🇷 French | fr | | 🇸🇪 Swedish | sv |
| 🇪🇸 Spanish | es | | 🇨🇿 Czech | cs |
| 🇮🇹 Italian | it | | 🇬🇷 Greek | el |
| 🇯🇵 Japanese | ja | | 🇮🇳 Hindi | hi |
| 🇨🇳 Chinese | zh | | 🇸🇦 Arabic | ar |
| 🇵🇹 Portuguese | pt | | 🇺🇦 Ukrainian | uk |
| 🇷🇺 Russian | ru | | 🇮🇱 Hebrew | he |
| 🇰🇷 Korean | ko | | 🇳🇱 Dutch | nl |

#### 3. Choose Backend

**Ollama (Local) - Recommended**
- ✅ 100% private - no data leaves your PC
- ✅ Free to use
- ✅ Works offline
- ⚠️ Requires GPU with sufficient VRAM

**Ollama Cloud - For Low-End PCs**
- ✅ No GPU required
- ✅ Access to huge models (up to 671B parameters)
- ⚠️ Requires internet connection
- ⚠️ Data sent to Ollama servers

**OpenAI (Cloud)**
- ✅ Best translation quality
- ✅ No GPU required
- ⚠️ Costs ~$0.01-0.05 per page
- ⚠️ Requires internet connection

**No Translation**
- Only extracts and reformats the PDF
- Useful for testing LaTeX output

#### 4. Configure Ollama (if selected)

1. **Check VRAM Detection** - The app shows your detected GPU VRAM
2. **Select VRAM** - Choose your GPU's VRAM from the dropdown
3. **Select Model** - Choose a model that fits your VRAM
   - 💾 = Already installed
   - ✅ = Fits comfortably
   - ⚠️ = Might be tight
4. **Download Model** - Click if model isn't installed yet

#### 5. Translate

1. Click the **"🚀 Translate"** button
2. Wait for processing:
   - Analyzing PDF... (10%)
   - Translating blocks... (30-70%)
   - Compiling LaTeX... (70-95%)
   - Done! (100%)
3. Download the translated PDF

#### 6. Download Result

- Click the download button next to "Translated PDF"
- The file path is shown in the status area
- Original formatting and formulas are preserved

---

## 🤖 LLM Backend Options

### Option 1: Ollama (Local) ⭐ Recommended
**Advantages:**
- 🔒 Complete privacy - data never leaves your computer
- 💰 Free to use - no API costs
- 🌐 Works offline - no internet required
- ⚡ Fast for repeated use - model stays in memory

**Disadvantages:**
- 🎮 Requires GPU with sufficient VRAM
- 📦 Initial model download (2-40 GB)
- 🔧 Slightly more complex setup

**Best for:**
- Privacy-conscious users
- Frequent translations
- Users with gaming GPUs (RTX 3060+)

### Option 2: Ollama Cloud (Preview) ☁️ NEW

**Advantages:**
- 🚀 Access to huge models (20B, 120B, 480B, 671B parameters)
- 💻 No GPU required - runs on Ollama's servers
- 🔧 Easy setup - just `ollama signin`
- 💰 Currently free during preview

**Disadvantages:**
- 🌐 Requires internet connection
- 🔓 Data is sent to Ollama servers (ollama.com)
- ⏳ May be slower than local models (network latency)

**Best for:**
- Users without GPU or with low VRAM
- Google Colab users (T4 only has 16GB)
- When you need maximum model quality

**Available Cloud Models:**

| Model | Parameters | Description |
|-------|------------|-------------|
| `gpt-oss:20b-cloud` | 20B | Good quality, fast |
| `gpt-oss:120b-cloud` | 120B | Excellent quality |
| `qwen3-coder:480b-cloud` | 480B | Best for technical texts |
| `deepseek-v3.1:671b-cloud` | 671B | Maximum quality |

**Setup:**
```bash
# 1. Install Ollama v0.12+
# https://ollama.com/download

# 2. Sign in to Ollama
ollama signin

# 3. Pull a cloud model
ollama pull gpt-oss:120b-cloud

# 4. Use it like any other model
ollama run gpt-oss:120b-cloud
```

**Privacy Note:** When using cloud models, your PDF content is sent to Ollama's servers for processing. Use local models if you need complete privacy.

### Option 3: OpenAI (Cloud)

**Advantages:**
- 🏆 Best translation quality (GPT-4)
- 💻 No GPU required
- 🚀 Instant start - no model download

**Disadvantages:**
- 💰 Costs money (~$0.01-0.05 per page)
- 🌐 Requires internet connection
- 🔓 Data sent to OpenAI servers

**Best for:**
- Occasional translations
- Users without GPU
- When quality is paramount

**Setup:**
1. Go to https://platform.openai.com
2. Create an account and add payment method
3. Generate an API key
4. Enter the key in the app (NOT stored permanently!)

### Comparison Table

| Feature | Ollama Local | Ollama Cloud | OpenAI |
|---------|--------------|--------------|--------|
| **Privacy** | ✅ 100% local | ⚠️ Data to Ollama | ⚠️ Data to OpenAI |
| **GPU Required** | ✅ Yes | ❌ No | ❌ No |
| **Internet** | ❌ Offline OK | ✅ Required | ✅ Required |
| **Cost** | 💰 Free | 💰 Free (preview) | 💰 $0.01-0.05/page |
| **Max Model Size** | ~70B (48GB VRAM) | 671B | GPT-4 |
| **Speed** | ⚡ Fast | 🐢 Network latency | ⚡ Fast |

---

## 🎮 VRAM & Model Guide

### Understanding VRAM Requirements

| Your VRAM | What You Can Run | Quality Level |
|-----------|------------------|---------------|
| 4 GB | Small models (2-3B parameters) | Basic |
| 6 GB | Small models, some 7B quantized | Good |
| 8 GB | 7B models quantized (Q4) | Good |
| 12 GB | 7B models full, some 12B | Very Good |
| 16 GB | 8B-12B models comfortably | Very Good |
| 24 GB | 22B-32B models | Excellent |
| 32 GB+ | 70B models quantized | Premium |
| 48 GB+ | 70B full precision | Maximum |
| **No GPU** | ☁️ Use Ollama Cloud models | Up to 671B! |

### Recommended Models by VRAM

#### 4-6 GB VRAM (GTX 1650, GTX 1060)

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `llama3.2:3b` | 2 GB | Good | Fast |
| `phi3:mini` | 2.3 GB | Good | Fast |
| `gemma2:2b` | 1.6 GB | Basic | Very Fast |

```bash
ollama pull llama3.2:3b
```

#### 8 GB VRAM (RTX 3060, GTX 1080)

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `mistral:7b-instruct-q4_0` | 4.1 GB | Good | Medium |
| `llama3.2:3b` | 2 GB | Good | Fast |
| `phi3:mini` | 2.3 GB | Good | Fast |

```bash
ollama pull mistral:7b-instruct-q4_0
```

#### 12-16 GB VRAM (RTX 3080, RTX 4070) ⭐ Sweet Spot

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `llama3.1:8b` ⭐ | 4.7 GB | Very Good | Medium |
| `mistral:7b` | 4.1 GB | Very Good | Medium |
| `mistral-nemo:12b` | 7.1 GB | Very Good | Slower |
| `openchat:7b` | 4.1 GB | Very Good | Medium |
| `gpt-oss:20b` | 12 GB | Excellent | Slower |

```bash
ollama pull llama3.1:8b
```

#### 24 GB VRAM (RTX 3090, RTX 4090)

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `mistral-small:22b` | 13 GB | Excellent | Medium |
| `codestral:22b` | 13 GB | Excellent | Medium |
| `qwen2.5:32b` | 19 GB | Excellent | Slower |
| `gpt-oss:20b` | 12 GB | Excellent | Medium |

```bash
ollama pull mistral-small:22b
```

#### 32+ GB VRAM (Workstation GPUs)

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `mixtral:8x7b` | 26 GB | Excellent | Medium |
| `llama3.1:70b-q4` | 40 GB | Premium | Slow |
| `qwen2.5:72b` | 43 GB | Premium | Slow |

```bash
ollama pull mixtral:8x7b
```

#### No GPU? Use Ollama Cloud! ☁️

| Model | Parameters | Quality | Setup |
|-------|------------|---------|-------|
| `gpt-oss:20b-cloud` | 20B | Excellent | `ollama signin` |
| `gpt-oss:120b-cloud` | 120B | Premium | `ollama signin` |
| `qwen3-coder:480b-cloud` | 480B | Maximum | `ollama signin` |
| `deepseek-v3.1:671b-cloud` | 671B | Maximum | `ollama signin` |

```bash
ollama signin
ollama pull gpt-oss:120b-cloud
```

### ChatGPT-like Open Source Alternatives

These models are specifically trained to be conversational like ChatGPT:

| Model | VRAM | Description |
|-------|------|-------------|
| `openchat:7b` | 8 GB | ChatGPT-like responses |
| `neural-chat:7b` | 8 GB | Intel optimized |
| `openchat:8b` | 10 GB | Improved version |
| `command-r:35b` | 24 GB | Cohere's instruction model |

---

## 🔌 API Reference

### FastAPI Endpoints

Start the FastAPI server:
```bash
uvicorn main:app --reload
# API: http://127.0.0.1:8000
# Docs: http://127.0.0.1:8000/docs
```

### Upload and Translate

#### With Ollama (Local)
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "pdf=@your_paper.pdf" \
  -F "target_language=en" \
  -F "use_ollama=true" \
  -F "ollama_model=llama3.1:8b"
```

#### With OpenAI (Cloud)
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "pdf=@your_paper.pdf" \
  -F "target_language=en" \
  -F "use_openai=true" \
  -F "openai_api_key=sk-your-key-here"
```

### Check Job Status

```bash
curl "http://localhost:8000/job/{job_id}"
```

**Response:**
```json
{
  "job_id": "abc123",
  "status": "translating",
  "progress": 45,
  "message": "Translating block 15 of 33..."
}
```

### Download Result

```bash
curl -O "http://localhost:8000/job/{job_id}/pdf"
```

### Status Values

| Status | Description |
|--------|-------------|
| `queued` | Job is waiting to start |
| `analyzing` | PDF is being parsed |
| `translating` | Text is being translated |
| `latex_build` | LaTeX is being compiled |
| `done` | Translation complete |
| `error` | An error occurred |

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### "Ollama not reachable"

**Cause:** Ollama service is not running.

**Solution:**
```bash
# Start Ollama
ollama serve

# Check if running
curl http://localhost:11434/api/tags
```

#### "Model not found"

**Cause:** The selected model is not downloaded.

**Solution:**
```bash
# Download the model
ollama pull llama3.1:8b

# List installed models
ollama list
```

#### "Cloud model not working"

**Cause:** Not signed in to Ollama Cloud.

**Solution:**
```bash
# Sign in (requires Ollama v0.12+)
ollama signin

# Then pull the cloud model
ollama pull gpt-oss:120b-cloud
```

#### "Out of Memory" / "CUDA out of memory"

**Cause:** Model is too large for your GPU VRAM.

**Solutions:**
1. Choose a smaller model
2. Close other GPU-intensive applications
3. Use a quantized version (e.g., `q4_0`)
4. **Use Ollama Cloud models** - no GPU needed!

```bash
# Use quantized version
ollama pull mistral:7b-instruct-q4_0

# Or use cloud model (no GPU needed)
ollama signin
ollama pull gpt-oss:120b-cloud
```

#### "PDF compilation failed" / "pdflatex not found"

**Cause:** LaTeX is not installed or not in PATH.

**Solution (Windows):**
```batch
winget install MiKTeX.MiKTeX
# Restart your terminal after installation
```

**Solution (Linux):**
```bash
sudo apt install texlive-latex-base texlive-latex-extra
```

**Solution (macOS):**
```bash
brew install --cask basictex
```

#### "GPU not detected"

**Cause:** NVIDIA drivers not installed or GPU not supported.

**Solutions:**
1. Install/update NVIDIA drivers
2. Select VRAM manually in the dropdown
3. Use CPU mode (slower)
4. **Use Ollama Cloud models** - no GPU needed!

```bash
# Check GPU (Windows)
nvidia-smi

# Check GPU (Linux)
nvidia-smi
# or
rocm-smi  # for AMD
```

#### "Translation quality is poor"

**Solutions:**
1. Use a larger model if you have enough VRAM
2. Switch to OpenAI for best quality
3. Try a different model (Mistral often works well for translations)
4. Use Ollama Cloud for larger models without GPU

### Getting Help

If you encounter issues not listed here:

1. Check the console output for error messages
2. Look at the `data/jobs/{job_id}/main.log` for LaTeX errors
3. Open an issue on GitHub with:
   - Your operating system
   - Python version
   - GPU model and VRAM
   - Full error message

---

## 📁 Project Structure

```
pdf-translator/
│
├── 📄 install.bat          # Windows installation script
├── 📄 install.sh           # Linux/macOS installation script
├── 📄 run.bat              # Windows start script
├── 📄 run.sh               # Linux/macOS start script
├── 📄 requirements.txt     # Python dependencies
│
├── 🎨 gradio_app.py        # Main application (Gradio UI)
├── 🌐 index.html           # Alternative HTML frontend
├── ⚡ main.py              # FastAPI server
│
├── 🔧 ollama_backend.py    # Ollama integration & VRAM detection
├── 📑 pdf_processing.py    # PDF parsing & translation logic
├── 📝 latex_build.py       # LaTeX generation & compilation
├── 📊 models.py            # Data models (Block, JobInfo)
├── 📋 jobs.py              # Job management & state tracking
│
├── 📂 data/                # Runtime data directory
│   └── jobs/               # Translation jobs
│       └── {job_id}/       # Individual job folder
│           ├── original.pdf
│           ├── main.tex
│           ├── main.pdf    # Result
│           └── main.log    # LaTeX log
│
└── 📖 README.md            # This file
```

---

## 🔒 Security

### API Key Handling

- **User supplies own key:** Each user provides their own OpenAI API key
- **No storage:** Keys are NEVER stored (no file, no database, no `.env`)
- **No logging:** Keys are never logged; only `api_key_present=True/False`
- **Request-local only:** Keys are used only for the current request

### Data Privacy by Backend

| Backend | Data Location | Privacy Level |
|---------|---------------|---------------|
| **Ollama Local** | Your PC only | 🔒 Maximum |
| **Ollama Cloud** | Ollama servers | ⚠️ Moderate |
| **OpenAI** | OpenAI servers | ⚠️ Moderate |

### Security Measures

| Measure | Description |
|---------|-------------|
| **Path Traversal Protection** | Job IDs validated as UUIDs |
| **CORS Policy** | Credentials disabled with wildcard origins |
| **Upload Size Limit** | Maximum 50 MB per PDF |
| **Process Timeout** | pdflatex limited to 5 minutes |
| **Input Sanitization** | LaTeX special characters escaped |

### Deployment Recommendations

- ✅ **Local/Single-user:** Safe to use as-is
- ⚠️ **Shared environments:** Add authentication
- ⚠️ **Production:** Add rate limiting and user auth

---

## 📜 License

**© 2025 Sven Kalinowski with small help of Lino Casu**

Licensed under the **Anti-Capitalist Software License v1.4**

This software may be used for any purpose **except** for commercial use by capitalist enterprises. This includes but is not limited to:
- Personal use ✅
- Academic research ✅
- Non-profit organizations ✅
- Worker-owned cooperatives ✅
- For-profit corporations ❌

For the full license text, see: https://anticapitalist.software/

---

## 🙏 Acknowledgments

- **Ollama** - For making local LLMs accessible
- **Gradio** - For the beautiful UI framework
- **Meta AI** - For the Llama models
- **Mistral AI** - For the Mistral models
- **The LaTeX Project** - For the typesetting system

---

*Made with ❤️ for the open-source community*
