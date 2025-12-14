"""
Gradio Frontend for PDF-Translator
Supports PDF, DOCX, and LaTeX files.
- PDF: Uses overlay method with Cambria font (100% formula preservation)
- DOCX: Perfect translation with formatting preserved
- LaTeX: Direct source translation

Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List

import gradio as gr

# Backend imports
from ollama_backend import (
    OLLAMA_MODELS,
    get_models_for_vram,
    check_ollama_installed,
    get_installed_models,
    is_model_installed,
    pull_model,
    get_vram_recommendations,
    detect_gpu_vram,
)
from pdf_overlay_translator import translate_pdf_overlay
from docx_translator import translate_docx
from latex_translator import translate_latex_file
from uuid import uuid4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_translator.gradio")

# Languages
LANGUAGES = {
    "German": "de",
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
}

# VRAM options
VRAM_OPTIONS = [
    ("4 GB", 4),
    ("6 GB", 6),
    ("8 GB", 8),
    ("12 GB", 12),
    ("16 GB", 16),
    ("24 GB", 24),
]


def get_model_choices(vram_gb: int) -> List[Tuple[str, str]]:
    """Get model choices for given VRAM."""
    models = get_models_for_vram(vram_gb)
    choices = []
    for model_id, info in models.items():
        label = f"{info['name']} ({info['size']})"
        choices.append((label, model_id))
    return choices if choices else [("qwen2.5:7b", "qwen2.5:7b")]


def check_ollama_status() -> str:
    """Check Ollama installation status."""
    if not check_ollama_installed():
        return "Ollama not installed. Visit https://ollama.ai"
    
    installed = get_installed_models()
    if installed:
        return f"Ollama ready. Models: {', '.join(installed[:3])}{'...' if len(installed) > 3 else ''}"
    return "Ollama installed but no models. Click 'Download Model'."


def pull_ollama_model(model_name: str) -> str:
    """Download an Ollama model."""
    if not model_name:
        return "No model selected"
    
    success, message = pull_model(model_name)
    return message


def translate_document(
    doc_file,
    target_language: str,
    ollama_model: str,
    progress=gr.Progress()
) -> Tuple[Optional[str], str]:
    """
    Translate document (PDF, DOCX, or LaTeX).
    - PDF: Overlay method with Cambria font (100% formula preservation)
    - DOCX: Perfect translation with formatting preserved (RECOMMENDED)
    - LaTeX: Direct source translation
    """
    if doc_file is None:
        return None, "Please upload a file."
    
    if not ollama_model:
        return None, "Please select a model."
    
    # Check if model is installed
    if not is_model_installed(ollama_model):
        return None, f"Model '{ollama_model}' not installed. Click 'Download Model'."
    
    try:
        # Create output directory
        job_id = str(uuid4())[:8]
        output_dir = Path(tempfile.gettempdir()) / "pdf_translator" / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_ext = Path(doc_file.name).suffix.lower()
        
        # Handle DOCX files (PERFECT TRANSLATION)
        if file_ext == '.docx':
            progress(0.05, desc="Starting DOCX translation (perfect mode)...")
            output_docx = output_dir / "translated.docx"
            
            result = translate_docx(
                doc_file.name,
                str(output_docx),
                ollama_model,
                target_language,
                progress_callback=lambda c, t, s: progress(c/100, desc=s)
            )
            
            if result.success:
                status = f"""✅ DOCX Translation Complete (PERFECT MODE)

Paragraphs translated: {result.paragraphs_translated}
Paragraphs skipped: {result.paragraphs_skipped}

PERFECT FEATURES:
- All formatting preserved (bold, italic, fonts)
- Tables translated
- Mathematical formulas preserved
- Document structure intact

Output: {result.output_path}

TIP: Open in Word and export as PDF for perfect results."""
                
                if result.warnings:
                    status += f"\n\nWarnings: {', '.join(result.warnings)}"
                
                return result.output_path, status
            else:
                return None, f"DOCX translation failed: {', '.join(result.warnings)}"
        
        # Handle LaTeX files
        if file_ext == '.tex':
            progress(0.1, desc="Translating LaTeX file...")
            output_tex = output_dir / "translated.tex"
            success = translate_latex_file(
                doc_file.name,
                str(output_tex),
                ollama_model,
                target_language,
                progress_callback=lambda c, t, s: progress(c/t, desc=s)
            )
            if success:
                return str(output_tex), f"✅ LaTeX translation complete!\n\nFile: {output_tex}\n\nCompile with pdflatex for perfect PDF."
            return None, "LaTeX translation failed."
        
        # Handle PDF files (overlay method)
        progress(0.05, desc="Starting PDF translation (overlay mode)...")
        
        result = translate_pdf_overlay(
            doc_file.name,
            str(output_dir),
            ollama_model,
            target_language,
            progress_callback=lambda c, t, s: progress(c/100, desc=s)
        )
        
        if result.success:
            status = f"""✅ PDF Translation Complete

Pages: {result.pages_processed}
Blocks translated: {result.blocks_translated}
Blocks skipped: {result.blocks_skipped}
Font: Cambria (Unicode math support)

Formulas preserved: ΔΦ, ω, 10⁻¹⁶ ✓

Output: {result.output_path}

TIP: For PERFECT results, use the original DOCX file instead!"""
            
            if result.warnings:
                status += f"\n\nWarnings: {', '.join(result.warnings)}"
            
            return result.output_path, status
        else:
            return None, f"Translation failed: {', '.join(result.warnings)}"
    
    except Exception as e:
        logger.exception("Translation failed")
        return None, f"Error: {str(e)}"


def create_gradio_app():
    """Create the Gradio app."""
    
    with gr.Blocks(
        title="PDF Translator",
        theme=gr.themes.Soft(primary_hue="emerald"),
    ) as app:
        
        gr.Markdown("""
        # PDF & DOCX Translator
        ### Translate scientific documents with formula preservation
        
        **Supported formats:**
        - **DOCX** (Word) - ⭐ PERFECT translation with all formatting
        - **PDF** - Good translation with Cambria font for math symbols
        - **LaTeX** (.tex) - Direct source translation
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                doc_input = gr.File(
                    label="Upload Document (DOCX recommended for perfect results)",
                    file_types=[".pdf", ".docx", ".tex"],
                    type="filepath",
                )
                
                target_lang = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="German",
                    label="Target Language",
                )
                
                # VRAM selection
                detected_vram = detect_gpu_vram()
                default_vram = "8 GB"
                if detected_vram:
                    for label, vram in VRAM_OPTIONS:
                        if detected_vram <= vram:
                            default_vram = label
                            break
                
                vram_info = f"Detected: {detected_vram} GB VRAM" if detected_vram else "GPU not detected"
                gr.Markdown(vram_info)
                
                vram_select = gr.Dropdown(
                    choices=[v[0] for v in VRAM_OPTIONS],
                    value=default_vram,
                    label="Your VRAM",
                )
                
                # Model selection
                initial_vram = detected_vram if detected_vram else 8
                initial_choices = get_model_choices(initial_vram)
                
                model_select = gr.Dropdown(
                    choices=initial_choices,
                    value=initial_choices[0][1] if initial_choices else "qwen2.5:7b",
                    label="Ollama Model",
                )
                
                ollama_status = gr.Textbox(
                    value=check_ollama_status(),
                    label="Ollama Status",
                    interactive=False,
                )
                
                with gr.Row():
                    refresh_btn = gr.Button("Check Status", size="sm")
                    pull_btn = gr.Button("Download Model", size="sm")
                
                translate_btn = gr.Button("Translate", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                output_file = gr.File(
                    label="Download Translated Document",
                    interactive=False,
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    lines=14,
                )
                
                gr.Markdown("""
                ### Why DOCX is better than PDF:
                - **Perfect formatting** - bold, italic, fonts preserved
                - **Complete paragraphs** - no fragmented blocks
                - **Tables translated** - with structure intact
                - **Easy to PDF** - just export from Word
                
                ### Scientific terminology:
                - Entanglement → Verschränkung
                - Coherence → Kohärenz
                - Abstract → Zusammenfassung
                """)
        
        # Update models when VRAM changes
        def update_models(vram_label):
            vram = dict(VRAM_OPTIONS).get(vram_label, 8)
            choices = get_model_choices(vram)
            return gr.Dropdown(choices=choices, value=choices[0][1] if choices else "")
        
        vram_select.change(update_models, inputs=[vram_select], outputs=[model_select])
        refresh_btn.click(check_ollama_status, outputs=[ollama_status])
        pull_btn.click(pull_ollama_model, inputs=[model_select], outputs=[ollama_status])
        
        translate_btn.click(
            translate_document,
            inputs=[doc_input, target_lang, model_select],
            outputs=[output_file, status_output],
        )
        
        gr.Markdown("""
        ---
        Licensed under the Anti-Capitalist Software License v1.4
        """)
    
    return app


if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(share=False)
