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
from docx_translator import translate_docx
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
    
    # Check if already installed
    if is_model_installed(model_name) and not force_update:
        installed = get_installed_models()
        for m in installed:
            if m == model_name or m.startswith(model_name.split(":")[0] + ":"):
                return f"💾 Model '{m}' is already installed!\n\nClick 'Update Model' to update to the latest version."
    
    action = "Updating" if force_update else "Downloading"
    progress(0, desc=f"{action} {model_name}...")
    
    def progress_callback(status: str, percent: int):
        progress(percent / 100, desc=f"{status} ({percent}%)")
    
    success = pull_model(model_name, progress_callback)
    
    if success:
        if force_update:
            return f"✅ Model '{model_name}' successfully updated!"
        return f"✅ Model '{model_name}' successfully downloaded!"
    return f"❌ Error downloading '{model_name}'"


def update_ollama_model(model_name: str, progress=gr.Progress()):
    """Updates an Ollama model to the latest version."""
    return pull_ollama_model(model_name, force_update=True, progress=progress)


def translate_latex_file_handler(
    tex_file,
    job_dir: Path,
    target_language: str,
    target_lang_code: str,
    use_openai: bool,
    openai_api_key: str,
    use_ollama: bool,
    ollama_model: str,
    latex_source_dir: str = "",
    progress=gr.Progress(),
) -> Tuple[Optional[str], str]:
    """
    Handles LaTeX file translation - perfect 1:1 output!
    """
    import shutil
    import subprocess
    
    progress(0.1, desc="Processing LaTeX file...")
    
    # Copy original .tex file
    original_name = Path(tex_file.name).stem
    tex_path = job_dir / f"{original_name}.tex"
    shutil.copy(tex_file.name, tex_path)
    
    # Copy all supporting files from source directory (images, bib, sty, etc.)
    # Use user-provided source directory if available, otherwise try parent of uploaded file
    if latex_source_dir and latex_source_dir.strip():
        source_dir = Path(latex_source_dir.strip())
        logger.info(f"Using user-provided source directory: {source_dir}")
    else:
        source_dir = Path(tex_file.name).parent
        logger.info(f"Using uploaded file's parent directory: {source_dir}")
    
    copied_count = 0
    online_mode_warning = ""
    if source_dir.exists():
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.pdf', '*.eps', '*.bib', '*.sty', '*.cls', '*.bst', '*.PNG', '*.JPG', '*.PDF']:
            for file in source_dir.glob(ext):
                if file.name != Path(tex_file.name).name:  # Don't copy the tex file again
                    try:
                        shutil.copy(file, job_dir / file.name)
                        logger.info(f"Copied supporting file: {file.name}")
                        copied_count += 1
                    except Exception as e:
                        logger.warning(f"Could not copy {file.name}: {e}")
        if copied_count > 0:
            logger.info(f"Copied {copied_count} supporting files from {source_dir}")
    else:
        logger.warning(f"Source directory does not exist: {source_dir}")
        online_mode_warning = "\n\n⚠️ **Online Mode:** Image folder not accessible. For .tex files with images, run locally!"
    
    # Output path
    output_tex = job_dir / f"{original_name}_translated.tex"
    
    progress(0.2, desc=f"Translating LaTeX to {target_language}...")
    
    # Translate using latex_translator
    if use_ollama:
        success = translate_latex_file(
            str(tex_path),
            str(output_tex),
            ollama_model,
            target_language,
            progress_callback=lambda c, t, s: progress(0.2 + 0.6 * c / max(t, 1), desc=s),
            use_openai=False
        )
    elif use_openai and openai_api_key:
        success = translate_latex_file(
            str(tex_path),
            str(output_tex),
            "",  # model not used for OpenAI
            target_language,
            progress_callback=lambda c, t, s: progress(0.2 + 0.6 * c / max(t, 1), desc=s),
            use_openai=True,
            openai_api_key=openai_api_key
        )
    else:
        return None, "❌ Please select Ollama or OpenAI backend with valid API key."
    
    if not success:
        return None, "❌ LaTeX translation failed!"
    
    progress(0.85, desc="Compiling PDF (this may take a while)...")
    
    # Try to compile to PDF
    try:
        # Use just the filename, not full path, since we set cwd
        tex_filename = output_tex.name
        
        # Find pdflatex - check common locations on Windows
        pdflatex_cmd = "pdflatex"
        if os.name == 'nt':  # Windows
            miktex_paths = [
                Path(os.environ.get('LOCALAPPDATA', '')) / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64" / "pdflatex.exe",
                Path("C:/MiKTeX/miktex/bin/x64/pdflatex.exe"),
                Path(os.environ.get('ProgramFiles', '')) / "MiKTeX" / "miktex" / "bin" / "x64" / "pdflatex.exe",
            ]
            for miktex_path in miktex_paths:
                if miktex_path.exists():
                    pdflatex_cmd = str(miktex_path)
                    logger.info(f"Found pdflatex at: {pdflatex_cmd}")
                    break
        
        # Run pdflatex twice for references (longer timeout for complex docs)
        for run in range(2):
            progress(0.85 + run * 0.05, desc=f"Compiling PDF (pass {run+1}/2)...")
            result = subprocess.run(
                [pdflatex_cmd, "-interaction=nonstopmode", "-halt-on-error", tex_filename],
                cwd=str(job_dir),
                capture_output=True,
                timeout=300  # 5 minutes per run for complex documents
            )
            logger.info(f"pdflatex run {run+1}: returncode={result.returncode}")
            
            # Check if PDF was created after first run
            output_pdf = job_dir / f"{original_name}_translated.pdf"
            if output_pdf.exists() and run == 0:
                # First run succeeded, do second run for references
                continue
            elif not output_pdf.exists() and run == 0:
                # First run failed, log and try once more
                stderr = result.stderr.decode('utf-8', errors='ignore')
                if stderr:
                    logger.warning(f"pdflatex stderr: {stderr[:1000]}")
        
        output_pdf = job_dir / f"{original_name}_translated.pdf"
        if output_pdf.exists():
            progress(1.0, desc="Complete!")
            return str(output_pdf), f"✅ LaTeX translated and compiled! Output: {output_pdf.name}{online_mode_warning}"
        else:
            # Log error for debugging
            stderr = result.stderr.decode('utf-8', errors='ignore')
            stdout = result.stdout.decode('utf-8', errors='ignore')
            logger.warning(f"PDF not created. returncode={result.returncode}")
            if stderr:
                logger.warning(f"stderr: {stderr[:500]}")
            # Check log file for errors
            log_file = job_dir / f"{original_name}_translated.log"
            if log_file.exists():
                log_content = log_file.read_text(errors='ignore')
                # Find error lines
                error_lines = [l for l in log_content.split('\n') if '!' in l or 'Error' in l]
                if error_lines:
                    logger.warning(f"LaTeX errors: {error_lines[:5]}")
    except subprocess.TimeoutExpired:
        logger.warning("pdflatex timed out after 5 minutes")
    except FileNotFoundError:
        logger.warning("pdflatex not found in PATH or MiKTeX locations")
    except Exception as e:
        logger.warning(f"PDF compilation failed: {e}")
    
    # Return .tex file if PDF compilation failed
    # Check for missing images in log
    log_file = job_dir / f"{original_name}_translated.log"
    missing_files_hint = ""
    if log_file.exists():
        log_content = log_file.read_text(errors='ignore')
        if "not found" in log_content.lower() and (".png" in log_content or ".jpg" in log_content or ".pdf" in log_content):
            missing_files_hint = "\n⚠️ Missing image files! Place images in same folder as .tex file before uploading."
    
    progress(1.0, desc="Complete (PDF compilation skipped)")
    return str(output_tex), f"✅ LaTeX translated! Output: {output_tex.name}\n(PDF compilation failed - compile manually with pdflatex){missing_files_hint}{online_mode_warning}"


def translate_pdf(
    pdf_file,
    target_language: str,
    extraction_mode: str,
    backend: str,
    openai_api_key: str,
    ollama_model: str,
    latex_source_dir: str = "",
    progress=gr.Progress(),
) -> Tuple[Optional[str], str]:
    """
    Main function: Translate PDF or LaTeX file.
    
    Returns:
        Tuple[Optional[str], str]: (Path to output file or None, Status message)
    """
    if pdf_file is None:
        return None, "❌ Please upload a PDF, DOCX, or .tex file!"
    
    # Check file type
    file_lower = pdf_file.name.lower()
    is_latex = file_lower.endswith('.tex')
    is_docx = file_lower.endswith('.docx')
    use_marker = "Marker" in extraction_mode
    
    # Backend validation
    use_openai = backend == "OpenAI"
    use_ollama = backend == "Ollama (Local)"
    
    logger.info(f"translate_pdf: backend='{backend}', use_ollama={use_ollama}, use_marker={use_marker}")
    
    if use_openai and not openai_api_key:
        return None, "❌ OpenAI selected, but no API key provided!"
    
    if use_ollama:
        if not check_ollama_installed():
            return None, "❌ Ollama not reachable! Please start Ollama first."
        if not ollama_model:
            return None, "❌ Ollama selected, but no model chosen!"
    
    target_lang_code = LANGUAGES.get(target_language, "en")
    
    try:
        # Create job ID with timestamp for multi-user support
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        job_id = f"{timestamp}_{uuid4().hex[:8]}"
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created job directory: {job_dir}")
        
        # Handle LaTeX files directly
        if is_latex:
            return translate_latex_file_handler(
                pdf_file, job_dir, target_language, target_lang_code,
                use_openai, openai_api_key, use_ollama, ollama_model, 
                latex_source_dir, progress
            )
        
        # Handle DOCX files directly
        if is_docx:
            if not use_ollama and not use_openai:
                return None, "❌ DOCX translation requires Ollama or OpenAI backend."
            progress(0.02, desc="Starting DOCX translation...")
            import shutil
            docx_path = job_dir / "input.docx"
            shutil.copy(pdf_file.name, docx_path)
            output_docx = job_dir / f"translated_{Path(pdf_file.name).stem}.docx"
            
            result = translate_docx(
                str(docx_path),
                str(output_docx),
                ollama_model if use_ollama else "",
                target_language,
                progress_callback=lambda c, t, s: progress(c / 100, desc=s)
            )
            success = result.success
            
            if success and output_docx.exists():
                return str(output_docx), f"✅ DOCX translated! Output: {output_docx.name}"
            else:
                return None, "❌ DOCX translation failed!"
        
        # Use Unified mode for best quality (combines all methods)
        use_unified = "Unified" in extraction_mode
        if use_unified:
            if not use_ollama and not use_openai:
                return None, "❌ Unified mode requires Ollama or OpenAI backend."
            progress(0.02, desc="Starting unified translation (best quality)...")
            output_path, status = translate_pdf_unified(
                pdf_file.name,
                str(job_dir),
                ollama_model if use_ollama else "",
                target_language,
                progress_callback=lambda c, t, s: progress(c / 100, desc=s),
                use_openai=use_openai,
                openai_api_key=openai_api_key
            )
            return output_path, status
        
        # Use Page-by-Page mode for layout preservation with images
        use_page_by_page = "Page-by-Page" in extraction_mode
        if use_page_by_page:
            if not use_ollama and not use_openai:
                return None, "❌ Page-by-Page mode requires Ollama or OpenAI backend."
            progress(0.02, desc="Starting page-by-page translation...")
            output_path, status = translate_pdf_page_by_page(
                pdf_file.name,
                str(job_dir),
                ollama_model if use_ollama else "",
                target_language,
                progress_callback=lambda c, t, s: progress(c / 100, desc=s),
                use_openai=use_openai,
                openai_api_key=openai_api_key
            )
            return output_path, status
        
        # Use Marker pipeline for scientific PDFs (works with both Ollama and OpenAI)
        if use_marker:
            if not use_ollama and not use_openai:
                return None, "❌ Marker mode requires Ollama or OpenAI backend."
            progress(0.02, desc="Loading Marker models (first run may take 5-10 min to download)...")
            output_path, status = translate_pdf_with_marker(
                pdf_file.name,
                str(job_dir),
                ollama_model if use_ollama else "",
                target_language,
                progress_callback=lambda c, t, s: progress(c / 100, desc=s),
                use_openai=use_openai,
                openai_api_key=openai_api_key
            )
            return output_path, status
        
        progress(0.1, desc="Analyzing PDF...")
        
        # Copy PDF
        pdf_path = job_dir / "original.pdf"
        with open(pdf_file.name, "rb") as f:
            pdf_path.write_bytes(f.read())
        
        # Analyze PDF
        blocks, detected_language = analyze_pdf(pdf_path)
        logger.info(f"Analyzed PDF: {len(blocks)} blocks, detected language: {detected_language}")
        
        progress(0.3, desc=f"Translating {len(blocks)} blocks...")
        
        # Translate
        translated_blocks = translate_blocks(
            blocks,
            source_language=detected_language,
            target_language=target_lang_code,
            use_openai=use_openai,
            openai_api_key=openai_api_key if use_openai else None,
            use_ollama=use_ollama,
            ollama_model=ollama_model if use_ollama else None,
        )
        
        progress(0.7, desc="Compiling LaTeX...")
        
        # Build LaTeX
        build_and_compile(
            job_id=job_id,
            blocks=translated_blocks,
            target_language=target_lang_code,
            source_language=detected_language,
        )
        
        progress(0.95, desc="Done!")
        
        # Result PDF
        result_pdf = job_dir / "main.pdf"
        if result_pdf.exists():
            # Gradio needs the absolute path as string
            absolute_path = str(result_pdf.resolve())
            logger.info(f"PDF created at: {absolute_path}")
            return absolute_path, f"✅ Translation successful! ({len(blocks)} blocks translated)\n\n📁 File: {absolute_path}"
        else:
            # Check if LaTeX log exists for error message
            log_file = job_dir / "main.log"
            error_msg = "PDF compilation failed."
            if log_file.exists():
                try:
                    log_content = log_file.read_text(encoding="utf-8", errors="ignore")
                    # Search for LaTeX errors
                    for line in log_content.split("\n"):
                        if line.startswith("!"):
                            error_msg += f"\n\nLaTeX error: {line}"
                            break
                except:
                    pass
            return None, f"❌ {error_msg}\n\nCheck LaTeX installation (MiKTeX/TeX Live)."
    
    except Exception as e:
        logger.exception("Translation failed")
        return None, f"❌ Error: {str(e)}"


def create_gradio_app():
    """Creates the Gradio app."""
    
    with gr.Blocks(
        title="PDF Translator",
        theme=gr.themes.Soft(primary_hue="emerald"),
        css="""
        .container { max-width: 900px; margin: auto; }
        .header { text-align: center; margin-bottom: 20px; }
        .warning { background-color: #fef3c7; padding: 10px; border-radius: 8px; margin: 10px 0; }
        """
    ) as app:
        
        gr.Markdown("""
        # 📄 PDF Translator
        ### Translate scientific PDFs with AI
        
        Supports **OpenAI GPT-4** and **local Ollama models**.
        LaTeX formulas and structure are preserved.
        """)
        
        with gr.Tabs():
            # === TAB 1: Translate ===
            with gr.TabItem("🔄 Translate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        pdf_input = gr.File(
                            label="Upload PDF, DOCX, or LaTeX",
                            file_types=[".pdf", ".docx", ".tex"],
                            type="filepath",
                        )
                        
                        latex_source_dir = gr.Textbox(
                            label="LaTeX Source Directory (optional)",
                            placeholder="E:\\path\\to\\folder\\with\\images",
                            info="For .tex files: folder containing images/bib files",
                            visible=True,
                        )
                        
                        target_lang = gr.Dropdown(
                            choices=list(LANGUAGES.keys()),
                            value="English",
                            label="Target Language",
                        )
                        
                        # Perfect Mode - always uses best quality (Unified)
                        extraction_mode = gr.State("Unified (Best Quality - Combines All Methods)")
                        
                        gr.Markdown("""
                        ### ⭐ Perfect Mode (1:1 Translation)
                        - ✅ Marker for formulas (100% accurate)
                        - ✅ PyMuPDF for layout & images
                        - ✅ All images preserved in place
                        - ✅ Tables fully translated
                        """)
                        
                        backend_choice = gr.Radio(
                            choices=["Ollama (Local)", "OpenAI", "No Translation"],
                            value="Ollama (Local)",
                            label="Backend",
                        )
                        
                        # OpenAI Options
                        with gr.Group(visible=False) as openai_group:
                            openai_key = gr.Textbox(
                                label="OpenAI API Key",
                                type="password",
                                placeholder="sk-...",
                                info="NOT stored!",
                            )
                        
                        # Ollama Options
                        with gr.Group(visible=True) as ollama_group:
                            # Dynamic VRAM options based on detected hardware
                            available_vram = get_available_vram_options()
                            detected_vram = detect_gpu_vram()
                            
                            # Select best matching option as default
                            if detected_vram:
                                # Find highest option that is <= detected
                                default_option = available_vram[-1][0] if available_vram else VRAM_OPTIONS_ALL[2][0]
                            else:
                                default_option = available_vram[2][0] if len(available_vram) > 2 else available_vram[-1][0]
                            
                            gr.Markdown(get_detected_vram_info())
                            
                            vram_select = gr.Dropdown(
                                choices=[v[0] for v in available_vram],
                                value=default_option,
                                label="Your VRAM (only matching options)",
                            )
                            
                            # Capacity info display
                            initial_vram = detected_vram if detected_vram else 16
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
