"""
Gradio Frontend for PDF-Translator
With Ollama and OpenAI Support

Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple, List

# Setup logging FIRST before any other imports
from logging_config import setup_logging, LOG_FILE, print_log_analysis
LOG_PATH = setup_logging()

import gradio as gr

# Backend imports
from ollama_backend import (
    OLLAMA_MODELS,
    get_models_for_vram,
    get_models_for_vram_with_installed,
    check_ollama_installed,
    get_installed_models,
    is_model_installed,
    pull_model,
    get_vram_recommendations,
    detect_gpu_vram,
    get_max_vram_for_system,
)
from pdf_processing import analyze_pdf, translate_blocks
from latex_build import build_and_compile, JOBS_DIR, cleanup_old_jobs
from latex_translator import translate_latex_file
from pdf_marker_translator import translate_pdf_with_marker
from page_by_page_translator import translate_pdf_page_by_page
from unified_translator import translate_pdf_unified
from uuid import uuid4
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_translator.gradio")

# VRAM options - dynamically filtered based on detected hardware
VRAM_OPTIONS_ALL = [
    ("4 GB (GTX 1650, GTX 1050 Ti)", 4),
    ("6 GB (GTX 1060, RTX 2060)", 6),
    ("8 GB (GTX 1070/1080, RTX 3060)", 8),
    ("12 GB (RTX 3060 12GB, RTX 4070)", 12),
    ("16 GB (RTX 4060 Ti 16GB, RTX 4080)", 16),
    ("24 GB (RTX 3090, RTX 4090)", 24),
    ("32 GB (Dual GPU / Workstation)", 32),
    ("48 GB (RTX A6000, Dual 24GB)", 48),
    ("64 GB (Multi-GPU Setup)", 64),
    ("96 GB (Enterprise / Multi-GPU)", 96),
]

# Ollama cloud models (remote inference; requires internet + Ollama signin)
OLLAMA_CLOUD_MODELS: List[str] = [
    "gpt-oss:20b-cloud",
    "gpt-oss:120b-cloud",
    "deepseek-v3.1:671b-cloud",
    "qwen3-coder:480b-cloud",
]


def get_available_vram_options() -> List[Tuple[str, int]]:
    """Returns VRAM options up to the next tier above detected VRAM."""
    detected = detect_gpu_vram()
    
    if detected is None:
        # No GPU detected - show all options with warning
        logger.warning("GPU VRAM not detected - showing all options")
        return VRAM_OPTIONS_ALL
    
    # Round up to nearest standard VRAM tier (Windows reserves some VRAM)
    # 15 GB detected -> 16 GB actual, 11 GB detected -> 12 GB actual, etc.
    vram_tiers = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
    actual_vram = detected
    for tier in vram_tiers:
        if detected <= tier:
            actual_vram = tier
            break
    
    # Filter to options <= actual VRAM (rounded up)
    available = [(label, vram) for label, vram in VRAM_OPTIONS_ALL if vram <= actual_vram]
    
    # If no option fits (very small GPU), offer at least 4GB
    if not available:
        available = [VRAM_OPTIONS_ALL[0]]
    
    logger.info("Detected %d GB VRAM (rounded to %d GB) - showing %d options", detected, actual_vram, len(available))
    return available


def get_detected_vram_info() -> str:
    """Returns info about detected VRAM with page capacity estimate."""
    from ollama_backend import get_page_estimate_for_vram
    
    detected = detect_gpu_vram()
    if detected:
        # Round up to nearest tier
        vram_tiers = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
        actual_vram = detected
        for tier in vram_tiers:
            if detected <= tier:
                actual_vram = tier
                break
        
        pages = get_page_estimate_for_vram(actual_vram)
        return f"🎮 Detected GPU VRAM: **{detected} GB** (≈{actual_vram} GB usable) → **~{pages} pages** consistent translation"
    return "⚠️ GPU VRAM could not be detected. Please select manually."


def get_vram_capacity_info(vram_gb: int, model_name: str = None) -> str:
    """Returns capacity info for selected VRAM and model."""
    from ollama_backend import (
        get_page_estimate_for_model, get_token_limit_for_model,
        get_page_estimate_for_vram, get_token_limit_for_vram,
        get_model_context_length, DEFAULT_CONTEXT_LENGTH
    )

    if model_name and model_name.endswith("-cloud"):
        return "☁️ **Cloud model** | VRAM independent (requires internet + `ollama signin`)"
    
    if model_name:
        # Model-specific capacity
        pages = get_page_estimate_for_model(model_name, vram_gb)
        tokens = get_token_limit_for_model(model_name, vram_gb)
        context = get_model_context_length(model_name)
        
        # Show model context info
        if context > DEFAULT_CONTEXT_LENGTH:
            context_str = f" (model: {context//1024}K)"
        else:
            context_str = ""
        
        return f"📄 **~{pages} pages** consistent | 🔤 {tokens:,} tokens{context_str}"
    else:
        # VRAM-only fallback
        pages = get_page_estimate_for_vram(vram_gb)
        tokens = get_token_limit_for_vram(vram_gb)
        return f"📄 **~{pages} pages** consistent | 🔤 {tokens:,} tokens max"


# For compatibility
VRAM_OPTIONS = VRAM_OPTIONS_ALL

# Languages
LANGUAGES = {
    "German": "de",
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
    "Japanese": "ja",
    "Chinese": "zh",
    "Portuguese": "pt",
    "Russian": "ru",
    "Korean": "ko",
    "Arabic": "ar",
    "Ukrainian": "uk",
    "Hebrew": "he",
    "Dutch": "nl",
    "Polish": "pl",
    "Turkish": "tr",
    "Swedish": "sv",
    "Czech": "cs",
    "Greek": "el",
    "Hindi": "hi",
}


def get_model_choices(vram_gb: int) -> List[Tuple[str, str]]:
    """Returns model choices - INSTALLED models first, then others."""
    from ollama_backend import get_installed_models, OLLAMA_MODELS, DEFAULT_CONTEXT_LENGTH
    
    # Get actually installed models from Ollama
    installed = get_installed_models()
    
    choices = []
    seen_models = set()
    
    # FIRST: Add all installed models (regardless of VRAM - user has them)
    for model_name in installed:
        # Get model info if available
        base_name = model_name.split(":")[0]
        model_info = None
        
        # Try exact match first
        if model_name in OLLAMA_MODELS:
            model_info = OLLAMA_MODELS[model_name]
        else:
            # Try base name match
            for name, info in OLLAMA_MODELS.items():
                if name.startswith(base_name):
                    model_info = info
                    break
        
        if model_info:
            desc = model_info.get("description", "Installed model")
            size = model_info.get("size_gb", "?")
        else:
            desc = "Installed model"
            size = "?"
        
        label = f"💾 {model_name} ({size} GB) [INSTALLED] - {desc}"
        choices.append((label, model_name))
        seen_models.add(model_name)
        seen_models.add(base_name)
    
    # SECOND: Add recommended models that fit VRAM (not installed)
    models = get_models_for_vram_with_installed(vram_gb)
    for m in models:
        name = m["name"]
        base = name.split(":")[0]
        
        # Skip if already added
        if name in seen_models or base in seen_models:
            continue
        
        if m["fits_comfortably"]:
            status_marker = "⬇️"  # Can download
        else:
            status_marker = "⚠️"  # Tight fit
        
        label = f"{status_marker} {m['name']} ({m['size_gb']} GB) - {m['description']}"
        choices.append((label, m["name"]))
        seen_models.add(name)

    # THIRD: Add Ollama cloud models (remote inference) as extra options
    for cloud_model in OLLAMA_CLOUD_MODELS:
        if cloud_model in seen_models:
            continue
        label = f"☁️ {cloud_model} [CLOUD] - runs on ollama.com (requires internet + ollama signin)"
        choices.append((label, cloud_model))
        seen_models.add(cloud_model)
    
    return choices


def update_model_dropdown(vram_label: str):
    """Updates model dropdown based on VRAM selection."""
    vram_gb = dict(VRAM_OPTIONS).get(vram_label, 16)
    choices = get_model_choices(vram_gb)
    # Use first model for initial capacity info
    first_model = choices[0][1] if choices else None
    capacity_info = get_vram_capacity_info(vram_gb, first_model)
    if choices:
        return gr.Dropdown(choices=choices, value=choices[0][1], interactive=True), capacity_info
    return gr.Dropdown(choices=[("No models available", "")], value="", interactive=False), capacity_info


def update_capacity_for_model(model_name: str, vram_label: str):
    """Updates capacity info when model selection changes."""
    vram_gb = dict(VRAM_OPTIONS).get(vram_label, 16)
    return get_vram_capacity_info(vram_gb, model_name)


def check_ollama_status():
    """Checks Ollama status and returns info."""
    if check_ollama_installed():
        installed = get_installed_models()
        if installed:
            return f"✅ Ollama running! Installed models: {', '.join(installed[:5])}{'...' if len(installed) > 5 else ''}"
        return "✅ Ollama running, but no models installed. Select a model and click 'Download Model'."
    return "❌ Ollama not reachable. Please start Ollama or install it from https://ollama.ai"


def pull_ollama_model(model_name: str, force_update: bool = False, progress=gr.Progress()):
    """Downloads or updates an Ollama model."""
    if not model_name:
        return "❌ No model selected!"
    
    if not check_ollama_installed():
        return "❌ Ollama not reachable! Please start Ollama first."

    if model_name.endswith("-cloud"):
        if not is_model_installed(model_name):
            msg = pull_model(model_name)
            if not msg:
                return "☁️ Cloud model: pull failed. Ensure you are signed in: run `ollama signin` (Ollama v0.12+)."
        return "☁️ Cloud model selected. If it doesn't work, run `ollama signin` and try again."
    
    # Check if already installed
    if is_model_installed(model_name) and not force_update:
        installed = get_installed_models()
        for m in installed:
            if m == model_name or m.startswith(model_name.split(":")[0] + ":"):
                return f"✅ Model already installed: {m}" 
    
    # Progress callback
    def progress_callback(status: str, percent: int):
        progress(percent / 100, desc=status)
    
    # Pull model
    success = pull_model(model_name, progress_callback)
    if success:
        return f"✅ Model downloaded: {model_name}" 
    return f"❌ Failed to download model: {model_name}" 


def update_ollama_model(model_name: str, progress=gr.Progress()):
    """Forces update of an Ollama model."""
    return pull_ollama_model(model_name, force_update=True, progress=progress)


def translate_pdf(
    pdf_file,
    target_language: str,
    extraction_mode: str,
    backend: str,
    openai_api_key: str,
    ollama_model: str,
    latex_source_dir=None,
    progress=gr.Progress(),
):
    """Main translation function."""
    
    # Validate inputs
    if not pdf_file:
        return None, "❌ Please upload a PDF file!"
    
    if backend == "OpenAI" and not openai_api_key:
        return None, "❌ Please enter your OpenAI API key!"
    
    if backend == "Ollama (Local)" and not ollama_model:
        return None, "❌ Please select an Ollama model!"

    if backend == "Ollama (Local)" and ollama_model and ollama_model.endswith("-cloud"):
        return None, "☁️ Selected a cloud model. Ensure internet access and run `ollama signin` if needed."
    
    # Create job directory
    job_id = str(uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy uploaded PDF
    pdf_path = job_dir / "original.pdf"
    if hasattr(pdf_file, "name"):
        # Gradio file object
        import shutil
        shutil.copy(pdf_file.name, pdf_path)
    else:
        # Path string
        import shutil
        shutil.copy(pdf_file, pdf_path)
    
    logger.info(f"Starting translation job {job_id}")
    
    try:
        # Analyze PDF
        progress(0.05, desc="🔍 Analyzing PDF structure...")
        pdf_info = analyze_pdf(str(pdf_path))
        
        # Translate based on mode
        if extraction_mode == "Unified (Recommended)":
            progress(0.10, desc="🚀 Starting unified translation...")
            output_path = translate_pdf_unified(
                str(pdf_path),
                target_language,
                use_openai=(backend == "OpenAI"),
                openai_api_key=openai_api_key,
                use_ollama=(backend == "Ollama (Local)"),
                ollama_model=ollama_model,
                job_dir=str(job_dir),
                progress=progress,
            )
        elif extraction_mode == "Marker (Best Quality)":
            progress(0.10, desc="🔬 Starting Marker translation...")
            output_path = translate_pdf_with_marker(
                str(pdf_path),
                target_language,
                use_openai=(backend == "OpenAI"),
                openai_api_key=openai_api_key,
                use_ollama=(backend == "Ollama (Local)"),
                ollama_model=ollama_model,
                job_dir=str(job_dir),
                progress=progress,
            )
        elif extraction_mode == "Page-by-Page (Fast)":
            progress(0.10, desc="⚡ Starting page-by-page translation...")
            output_path = translate_pdf_page_by_page(
                str(pdf_path),
                target_language,
                use_openai=(backend == "OpenAI"),
                openai_api_key=openai_api_key,
                use_ollama=(backend == "Ollama (Local)"),
                ollama_model=ollama_model,
                job_dir=str(job_dir),
                progress=progress,
            )
        else:
            progress(0.10, desc="📝 Starting translation...")
            output_path = translate_blocks(
                str(pdf_path),
                target_language,
                use_openai=(backend == "OpenAI"),
                openai_api_key=openai_api_key,
                use_ollama=(backend == "Ollama (Local)"),
                ollama_model=ollama_model,
                job_dir=str(job_dir),
                progress=progress,
            )
        
        progress(1.0, desc="✅ Translation complete!")
        
        if output_path and Path(output_path).exists():
            return output_path, f"✅ Translation complete! Job ID: {job_id}"
        return None, f"❌ Translation failed! Check logs. Job ID: {job_id}"
    
    except Exception as e:
        logger.exception(f"Translation error: {e}")
        return None, f"❌ Error: {str(e)}"


def create_gradio_app():
    """Creates the Gradio interface."""
    
    # Detect VRAM
    detected_vram = detect_gpu_vram() or 16
    
    # Round up to nearest tier
    vram_tiers = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
    initial_vram = detected_vram
    for tier in vram_tiers:
        if detected_vram <= tier:
            initial_vram = tier
            break

    with gr.Blocks(
        title="PDF Translator",
        theme=gr.themes.Soft(),
        css="""
        .warning {
            background: #fffbeb;
            border: 1px solid #f59e0b;
            border-radius: 8px;
            padding: 12px;
            margin: 10px 0;
        }
        """,
    ) as app:
        
        gr.Markdown("""
        # 📄 PDF Translator
        
        Translate scientific PDFs while preserving LaTeX formulas and document structure.
        """)
        
        with gr.Tabs():
            # === TAB 1: Main Translator ===
            with gr.TabItem("🚀 Translate"):
                
                with gr.Row():
                    with gr.Column(scale=2):
                        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"], file_count="single")
                        
                        target_lang = gr.Dropdown(
                            choices=list(LANGUAGES.keys()),
                            value="German",
                            label="Target Language",
                        )
                        
                        extraction_mode = gr.Dropdown(
                            choices=[
                                "Unified (Recommended)",
                                "Marker (Best Quality)",
                                "Page-by-Page (Fast)",
                                "Classic (Legacy)",
                            ],
                            value="Unified (Recommended)",
                            label="Extraction Mode",
                        )
                        
                        backend_choice = gr.Radio(
                            choices=["Ollama (Local)", "OpenAI"],
                            value="Ollama (Local)",
                            label="LLM Backend",
                        )
                        
                        # OpenAI group
                        with gr.Group(visible=False) as openai_group:
                            openai_key = gr.Textbox(
                                label="OpenAI API Key",
                                placeholder="sk-...",
                                type="password",
                            )
                        
                        # Ollama group
                        with gr.Group(visible=True) as ollama_group:
                            gr.Markdown(get_detected_vram_info())
                            
                            vram_options = get_available_vram_options()
                            vram_select = gr.Dropdown(
                                choices=[label for label, _ in vram_options],
                                value=[label for label, vram in vram_options if vram == initial_vram][0],
                                label="Your GPU VRAM",
                            )
                            
                            capacity_info = gr.Markdown(
                                value=get_vram_capacity_info(initial_vram),
                                elem_id="capacity_info"
                            )
                            
                            # Models based on detected VRAM
                            initial_choices = get_model_choices(initial_vram)
                            initial_model = initial_choices[0][1] if initial_choices else ""
                            
                            ollama_model_select = gr.Dropdown(
                                choices=initial_choices,
                                value=initial_model,
                                label="Ollama Model (only fitting your VRAM)",
                            )
                            
                            ollama_status = gr.Textbox(
                                value=check_ollama_status(),
                                label="Ollama Status",
                                interactive=False,
                            )
                            
                            with gr.Row():
                                refresh_btn = gr.Button("🔄 Check Status", size="sm")
                                pull_btn = gr.Button("⬇️ Download Model", size="sm", variant="secondary")
                                update_btn = gr.Button("🔄 Update Model", size="sm", variant="secondary")
                        
                        translate_btn = gr.Button("🚀 Translate", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        output_file = gr.File(
                            label="📥 Download Translated File",
                            file_count="single",
                            interactive=False,
                        )
                        
                        # Download button for online mode
                        download_btn = gr.DownloadButton(
                            label="⬇️ Download File",
                            visible=False,
                            variant="primary",
                        )
                        
                        status_output = gr.Textbox(label="Status", lines=8)
                        
                        # Warning for online users
                        gr.Markdown("""
                        <div class="warning">
                        ⚠️ **Online Mode Limitations:**
                        - For .tex files with images: **Local mode only!**
                        - Image folders cannot be accessed from online version
                        - Use `--local` flag or run locally for full functionality
                        </div>
                        """, visible=True)
                
                # Event Handler
                def toggle_backend(choice):
                    return (
                        gr.Group(visible=(choice == "OpenAI")),
                        gr.Group(visible=(choice == "Ollama (Local)")),
                    )
                
                backend_choice.change(
                    toggle_backend,
                    inputs=[backend_choice],
                    outputs=[openai_group, ollama_group],
                )
                
                vram_select.change(
                    update_model_dropdown,
                    inputs=[vram_select],
                    outputs=[ollama_model_select, capacity_info],
                )
                
                # Update capacity when model changes
                ollama_model_select.change(
                    update_capacity_for_model,
                    inputs=[ollama_model_select, vram_select],
                    outputs=[capacity_info],
                )
                
                refresh_btn.click(
                    check_ollama_status,
                    outputs=[ollama_status],
                )
                
                pull_btn.click(
                    pull_ollama_model,
                    inputs=[ollama_model_select],
                    outputs=[ollama_status],
                )
                
                update_btn.click(
                    update_ollama_model,
                    inputs=[ollama_model_select],
                    outputs=[ollama_status],
                )
                
                def translate_and_prepare_download(pdf_file, target_language, extraction_mode, backend, openai_api_key, ollama_model, latex_source_dir, progress=gr.Progress()):
                    """Wrapper that also prepares download button."""
                    output_path, status = translate_pdf(
                        pdf_file, target_language, extraction_mode, backend, 
                        openai_api_key, ollama_model, latex_source_dir, progress
                    )
                    
                    if output_path and Path(output_path).exists():
                        # Return file for both gr.File and gr.DownloadButton
                        return output_path, status, gr.DownloadButton(value=output_path, visible=True)
                    else:
                        return None, status, gr.DownloadButton(visible=False)
                
                translate_btn.click(
                    translate_and_prepare_download,
                    inputs=[pdf_input, target_lang, extraction_mode, backend_choice, openai_key, ollama_model_select, latex_source_dir],
                    outputs=[output_file, status_output, download_btn],
                )
            
            # === TAB 2: Model Recommendations ===
            with gr.TabItem("📊 VRAM Guide"):
                gr.Markdown(get_vram_recommendations())
                
                gr.Markdown("""
                ### Model Selection Tips
                
                | VRAM | Recommendation | Quality |
                |------|----------------|----------|
                | 8 GB | `llama3.2:3b` or `phi3:mini` | Good for simple texts |
                | 16 GB | `llama3.1:8b` or `mistral:7b` | **Best choice for most** |
                | 24 GB | `qwen2.5:32b` | Excellent for scientific texts |
                | 48+ GB | `llama3.1:70b` | Premium quality, near GPT-4 |
                
                **Note:** ✅ = Fits comfortably, ⚠️ = Works, but tight
                """)
            
            # === TAB 3: Help ===
            with gr.TabItem("❓ Help"):
                gr.Markdown("""
                ## Installation
                
                ### Ollama (Recommended for local use)
                
                1. **Install Ollama:** https://ollama.ai
                2. **Start Ollama:** `ollama serve`
                3. **Select a model** and click "Download Model"

                ### Ollama Cloud Models (Preview)

                1. Install latest Ollama (v0.12+): https://ollama.com/download
                2. Sign in: `ollama signin`
                3. Select a `...-cloud` model and click "Download Model"
                
                ### OpenAI (Cloud)
                
                1. Get API key from https://platform.openai.com
                2. Enter key (NOT stored!)
                3. Cost: approx. $0.01-0.05 per page
                
                ## Security
                
                - **API keys are NOT stored**
                - Keys are only used for the current request
                - Ollama runs completely local - no data leaves your PC
                
                ## Troubleshooting
                
                | Problem | Solution |
                |---------|----------|
                | "Ollama not reachable" | Start `ollama serve` |
                | "Model not found" | Download model first |
                | "PDF compilation failed" | Install LaTeX (TeX Live / MiKTeX) |
                | "Out of Memory" | Choose smaller model |
                """)
        
        gr.Markdown("""
        ---
        © 2025 Sven Kalinowski with small help of Lino Casu | Licensed under the Anti-Capitalist Software License v1.4
        """)
    
    return app


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF Translator - Gradio App")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public URL to share with friends",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for the server (default: 7860)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Local only mode (no public URL)",
    )
    args = parser.parse_args()
    
    # Share mode only if explicitly requested (--share flag)
    # Use --local to force local-only mode
    if args.local:
        args.share = False
    
    # Cleanup old job directories on startup
    cleanup_old_jobs(max_age_hours=24)
    logger.info(f"Job directory: {JOBS_DIR}")
    
    app = create_gradio_app()
    
    if args.share:
        print("")
        print("=" * 60)
        print("   SHARE MODE ACTIVE")
        print("=" * 60)
        print("")
        print("A public URL will be generated...")
        print("You can share this with friends!")
        print("")
        print("NOTE: The URL is valid for 72 hours.")
        print("      Your PC must stay on.")
        print("=" * 60)
        print("")
    
    app.launch(
        server_name="0.0.0.0" if args.share else "127.0.0.1",
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
    )
