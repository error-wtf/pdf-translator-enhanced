# 📄 PDF Translator Enhanced v2.0

**Translate scientific PDFs with 100% formula preservation and professional quality**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/error-wtf/pdf-translator-enhanced/blob/main/PDF_Translator_Colab.ipynb)
[![CI](https://github.com/error-wtf/pdf-translator-enhanced/actions/workflows/ci.yml/badge.svg)](https://github.com/error-wtf/pdf-translator-enhanced/actions)
[![License](https://img.shields.io/badge/license-Anti--Capitalist-red)](LICENSE)

> 🔗 **Based on:** [thelanguagenerd/pdf-translator](https://github.com/thelanguagenerd/pdf-translator)  
> This is a **major enhanced fork** with 100% formula preservation, 25 languages, and professional translation quality.

© 2025 Sven Kalinowski with small help of Lino Casu  
Licensed under the **Anti-Capitalist Software License v1.4**

---

## 🆕 What's New in v2.0

### Core Quality
- **🔬 100% Formula Preservation** - Hash-based isolation ensures NO formula corruption
- **📊 Quality Assurance** - Automatic 0-100 scoring with back-translation validation
- **🎯 Domain-Specific Prompts** - Optimized for physics, math, chemistry, biology, CS
- **📐 Layout Engine** - Exact font matching, columns, and text reflow

### Infrastructure
- **⚡ Batch Processing** - Translate multiple PDFs in parallel
- **💾 Translation Caching** - SQLite-based cache for speed and consistency
- **🔄 Resumable** - Checkpoint system to resume interrupted translations
- **🌍 25 Languages** - Including CJK, RTL, Cyrillic

### DevOps
- **🖥️ CLI Tool** - Full-featured command line interface
- **🐳 Docker** - One-click deployment with docker-compose
- **🧪 40+ Tests** - Comprehensive unit test suite
- **📦 Modern Packaging** - PEP 517/518 compliant (pyproject.toml)

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| **100% Formula Preservation** | Hash-based placeholder isolation |
| **25 Languages** | Latin, Cyrillic, CJK, RTL scripts |
| **Quality Scoring** | Automatic 0-100 evaluation |
| **Batch Processing** | Parallel multi-PDF translation |
| **Caching** | SQLite-based persistent cache |
| **Resume** | Checkpoint-based resume |
| **CLI** | Full command line interface |
| **Docker** | One-click deployment |
| **Ollama/OpenAI** | Local or cloud LLMs |

---

## 🚀 Quick Start

### Option 1: CLI (Recommended)

```bash
# Install
git clone https://github.com/error-wtf/pdf-translator-enhanced.git
cd pdf-translator-enhanced
pip install -e .

# Translate single PDF
python cli.py translate paper.pdf -l German

# Batch translate
python cli.py batch ./papers/ -l German -w 2

# Resume interrupted
python cli.py resume --all
```

### Option 2: Docker

```bash
# Start with Ollama
docker-compose up -d

# Access UI at http://localhost:7860
```

### Option 3: Gradio UI

```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama
ollama serve

# Run UI
python gradio_app.py
# → http://127.0.0.1:7860
```

---

## 🖥️ CLI Reference

```bash
# Single file translation
python cli.py translate input.pdf -l German -m qwen2.5:7b

# Batch translation (parallel)
python cli.py batch ./papers/ -l German -o ./output/ -w 2

# Resume interrupted translations
python cli.py resume --all

# Cache management
python cli.py cache stats
python cli.py cache clear --language German

# List supported languages
python cli.py languages

# List recommended models
python cli.py models
```

### CLI Options

| Option | Description |
|--------|-------------|
| `-l, --language` | Target language (default: German) |
| `-m, --model` | Ollama model (default: qwen2.5:7b) |
| `-o, --output` | Output directory |
| `-w, --workers` | Parallel workers for batch (default: 2) |
| `--force` | Force restart (ignore checkpoint) |
| `--qa` | Run quality check after translation |

---

## 🌍 Supported Languages (25)

### Latin Script
German, French, Spanish, Italian, Portuguese, Dutch, Swedish, Norwegian, Danish, Finnish, Polish, Czech, Hungarian, Romanian, Turkish, Vietnamese

### Cyrillic
Russian, Ukrainian, Bulgarian

### Other Scripts
Greek, Chinese, Japanese, Korean, Arabic (RTL), Hebrew (RTL), Thai, Hindi

---

## 📊 Quality Assurance

Every translation gets an automatic quality score (0-100):

| Score | Level | Description |
|-------|-------|-------------|
| 90-100 | Excellent | Publication ready |
| 75-89 | Good | Minor review needed |
| 60-74 | Acceptable | Review recommended |
| 40-59 | Poor | Significant issues |
| 0-39 | Failed | Re-translation needed |

### Scoring Components
- **Formula Integrity** (35%) - All LaTeX preserved correctly
- **Semantic Accuracy** (30%) - Back-translation similarity
- **Completeness** (20%) - No missing content
- **Terminology** (15%) - Consistent technical terms

---

## 🔧 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PDF Translator v2.0                       │
├─────────────────────────────────────────────────────────────┤
│  Input: PDF                                                  │
│    ↓                                                         │
│  Extraction: Marker + PyMuPDF + Nougat OCR                  │
│    ↓                                                         │
│  Formula Isolation: Hash-based placeholders (100% safe)     │
│    ↓                                                         │
│  Translation: Ollama/OpenAI + Domain Prompts + Glossary     │
│    ↓                                                         │
│  Formula Restoration: Verify integrity                       │
│    ↓                                                         │
│  Layout: Font matching + Column detection + Reflow          │
│    ↓                                                         │
│  QA: Back-translation + Scoring (0-100)                     │
│    ↓                                                         │
│  Output: Translated PDF                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Module Reference

### Core Modules

| Module | Purpose |
|--------|---------|
| `formula_isolator.py` | 100% formula preservation with hash placeholders |
| `translation_prompts.py` | Domain-specific optimized prompts |
| `glossary.py` | Terminology management (never-translate terms) |
| `layout_engine.py` | Precise layout reconstruction |
| `quality_assurance.py` | Automatic QA scoring (0-100) |
| `validation.py` | LaTeX syntax validation |
| `two_pass_translator.py` | Two-pass consistency refinement |

### Infrastructure Modules

| Module | Purpose |
|--------|---------|
| `batch_processor.py` | Multi-PDF parallel processing |
| `translation_cache.py` | SQLite caching with TTL |
| `progress_tracker.py` | Checkpoint-based resume |
| `languages.py` | 25 language configurations |
| `cli.py` | Command line interface |

### Extraction Modules

| Module | Purpose |
|--------|---------|
| `unified_translator.py` | Main translation pipeline |
| `nougat_extractor.py` | Nougat OCR for complex formulas |
| `formula_ocr.py` | pix2tex for formula images |
| `table_detector.py` | ML-based table detection |

---

## 💻 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10+ | 3.11+ |
| RAM | 8 GB | 16 GB |
| GPU VRAM | 4 GB (or Ollama Cloud) | 8-24 GB |
| Storage | 10 GB | 50 GB |

### Recommended Models

| VRAM | Model | Quality |
|------|-------|---------|
| 4-6 GB | `llama3.2:3b` | Good |
| 8 GB | `mistral:7b` | Very Good |
| 12-16 GB | `qwen2.5:7b` ⭐ | Excellent |
| 24 GB | `qwen2.5:32b` | Premium |
| No GPU | `gpt-oss:120b-cloud` | Premium |

---

## 🐳 Docker Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f pdf-translator

# Stop
docker-compose down
```

### Services
- **pdf-translator** - Main app on port 7860
- **ollama** - LLM server on port 11434

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

---

## 📝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📜 License

**© 2025 Sven Kalinowski with small help of Lino Casu**

Licensed under the **Anti-Capitalist Software License v1.4**

- ✅ Personal use
- ✅ Academic research
- ✅ Non-profit organizations
- ✅ Worker-owned cooperatives
- ❌ For-profit corporations

See [LICENSE](LICENSE) for full text.

---

## 🙏 Acknowledgments

- **Ollama** - Local LLM runtime
- **Gradio** - UI framework
- **Meta AI** - Llama models
- **Mistral AI** - Mistral models

---

*Made with ❤️ for the open-source community*
