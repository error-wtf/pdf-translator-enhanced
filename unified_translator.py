"""
Unified PDF Translator - Best Quality Pipeline

Combines all methods intelligently:
1. Marker for formula extraction (best for math)
2. Nougat as fallback (better for complex formulas)
3. PyMuPDF for layout/image extraction (best for structure)
4. Page-by-page processing (best for large docs)
5. Smart model loading/unloading for VRAM efficiency
6. Glossary for consistent terminology
7. Context window for translation consistency

Orchestration:
- Step 1: Split PDF into pages
- Step 2: For each page, extract with Marker/Nougat (formulas) + PyMuPDF (images/layout)
- Step 3: Merge extractions intelligently
- Step 4: Translate with LLM (Ollama/OpenAI) using context
- Step 5: Reconstruct PDF with original layout + translated text + images
- Step 6: Merge all pages

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import logging
import re
import subprocess
import shutil
import time
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import fitz  # PyMuPDF
from text_normalizer import normalize_text, normalize_and_reflow, count_garbage_chars
from caption_anchoring import anchor_captions_to_images, sort_blocks_reading_order, AnchoredFigure
from table_detector import detect_tables_in_page, DetectedTable

logger = logging.getLogger("pdf_translator.unified")


# =============================================================================
# TRANSLATION CONTEXT INTEGRATION
# =============================================================================

def get_translation_context(target_language: str):
    """Get or create translation context for consistent terminology."""
    try:
        from ollama_backend import TranslationContext
        return TranslationContext(target_language)
    except ImportError:
        logger.warning("TranslationContext not available")
        return None


# =============================================================================
# MODEL MANAGER
# =============================================================================

class ModelManager:
    """Manages model loading/unloading for VRAM efficiency."""
    
    def __init__(self):
        self.marker_loaded = False
        self.ollama_model = None
    
    def unload_ollama(self):
        """Unload Ollama models to free VRAM."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/ps", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                for model in models:
                    model_name = model.get("name", "")
                    if model_name:
                        logger.info(f"Unloading Ollama model: {model_name}")
                        requests.post(
                            "http://localhost:11434/api/generate",
                            json={"model": model_name, "keep_alive": 0, "prompt": ""},
                            timeout=30
                        )
                self.ollama_model = None
                time.sleep(2)  # Give time to release VRAM
        except Exception as e:
            logger.warning(f"Could not unload Ollama: {e}")
    
    def load_ollama(self, model_name: str):
        """Pre-load Ollama model."""
        if self.ollama_model == model_name:
            return
        try:
            import requests
            logger.info(f"Pre-loading Ollama model: {model_name}")
            requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model_name, "keep_alive": "5m", "prompt": "Hello"},
                timeout=60
            )
            self.ollama_model = model_name
        except Exception as e:
            logger.warning(f"Could not pre-load Ollama: {e}")
    
    def prepare_for_marker(self):
        """Prepare VRAM for Marker (unload Ollama first)."""
        self.unload_ollama()
        self.marker_loaded = True
    
    def prepare_for_translation(self, model_name: str):
        """Prepare for translation (Marker should be done by now)."""
        self.marker_loaded = False
        if model_name:
            self.load_ollama(model_name)


# Global model manager
model_manager = ModelManager()


# =============================================================================
# PDF PROCESSING FUNCTIONS
# =============================================================================

def split_pdf_into_pages(pdf_path: str, output_dir: Path) -> List[Path]:
    """Split PDF into individual pages."""
    doc = fitz.open(pdf_path)
    page_paths = []
    
    for page_num in range(len(doc)):
        page_pdf = output_dir / f"page_{page_num + 1:03d}.pdf"
        single_doc = fitz.open()
        single_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        single_doc.save(str(page_pdf))
        single_doc.close()
        page_paths.append(page_pdf)
    
    total = len(doc)
    doc.close()
    logger.info(f"Split PDF into {total} pages")
    return page_paths


def extract_with_marker(pdf_path: str, output_dir: Path, timeout: int = 120) -> Optional[str]:
    """Extract text/formulas with Marker (best for math)."""
    try:
        import sys
        import json
        
        worker_script = Path(__file__).parent / "marker_worker.py"
        python_exe = sys.executable
        
        result = subprocess.run(
            [python_exe, str(worker_script), pdf_path, str(output_dir)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path(__file__).parent)
        )
        
        for line in result.stdout.split('\n'):
            if line.startswith('MARKER_RESULT:'):
                result_json = json.loads(line[14:].strip())
                if result_json.get('success'):
                    return result_json.get('output_path')
        return None
    except subprocess.TimeoutExpired:
        logger.warning(f"Marker timed out after {timeout}s")
        return None
    except Exception as e:
        logger.warning(f"Marker extraction failed: {e}")
        return None


def extract_with_nougat_fallback(pdf_path: str, output_dir: Path) -> Optional[str]:
    """Try Nougat extraction as fallback for Marker."""
    try:
        from nougat_extractor import is_nougat_available, extract_with_nougat
        
        if not is_nougat_available():
            logger.info("Nougat not available as fallback")
            return None
        
        logger.info("Using Nougat as fallback extractor")
        markdown = extract_with_nougat(str(pdf_path), str(output_dir))
        
        if markdown and len(markdown) > 50:
            # Save to file
            output_file = output_dir / f"{Path(pdf_path).stem}_nougat.md"
            output_file.write_text(markdown, encoding="utf-8")
            return str(output_file)
        
        return None
    except ImportError:
        logger.debug("nougat_extractor not available")
        return None
    except Exception as e:
        logger.warning(f"Nougat extraction failed: {e}")
        return None


def smart_extract_text(pdf_path: str, output_dir: Path, timeout: int = 120) -> Tuple[Optional[str], str]:
    """
    Smart extraction: tries Marker first, falls back to Nougat.
    
    Returns (markdown_content, method_used)
    """
    # Try Marker first
    marker_result = extract_with_marker(pdf_path, output_dir, timeout=timeout)
    
    if marker_result and Path(marker_result).exists():
        content = Path(marker_result).read_text(encoding='utf-8')
        if len(content) > 50:
            return content, "marker"
    
    # Marker failed or produced empty output - try Nougat
    nougat_result = extract_with_nougat_fallback(pdf_path, output_dir)
    
    if nougat_result and Path(nougat_result).exists():
        content = Path(nougat_result).read_text(encoding='utf-8')
        if len(content) > 50:
            return content, "nougat"
    
    # Both failed - return None
    return None, "none"


def extract_with_pymupdf(pdf_path: Path) -> Dict:
    """Extract layout, images, and text blocks with PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    
    result = {
        "page_width": page.rect.width,
        "page_height": page.rect.height,
        "images": [],
        "text_blocks": []
    }
    
    # Extract images
    for img_idx, img_info in enumerate(page.get_images(full=True)):
        xref = img_info[0]
        try:
            base_image = doc.extract_image(xref)
            img_rects = page.get_image_rects(xref)
            if img_rects:
                rect = img_rects[0]
                result["images"].append({
                    "data": base_image["image"],
                    "ext": base_image["ext"],
                    "x": rect.x0,
                    "y": rect.y0,
                    "width": rect.width,
                    "height": rect.height
                })
        except Exception as e:
            logger.warning(f"Could not extract image {img_idx}: {e}")
    
    # Extract text blocks with positions
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    for block in blocks:
        if block["type"] == 0:  # Text block
            text = ""
            font_size = 12
            is_bold = False
            
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    # NORMALIZE each span immediately on extraction
                    span_text = normalize_text(span.get("text", ""))
                    text += span_text
                    font_size = span.get("size", 12)
                    font_name = span.get("font", "").lower()
                    is_bold = "bold" in font_name
                text += "\n"
            
            # Normalize the complete block text
            normalized_text = normalize_text(text.strip())
            
            if normalized_text:
                result["text_blocks"].append({
                    "text": normalized_text,
                    "x": block["bbox"][0],
                    "y": block["bbox"][1],
                    "width": block["bbox"][2] - block["bbox"][0],
                    "height": block["bbox"][3] - block["bbox"][1],
                    "font_size": font_size,
                    "is_bold": is_bold
                })
    
    # Sort blocks by reading order
    result["text_blocks"] = sort_blocks_by_position(result["text_blocks"])
    
    doc.close()
    return result


def sort_blocks_by_position(blocks: List[Dict]) -> List[Dict]:
    """Sort text blocks in reading order."""
    if not blocks:
        return blocks
    
    avg_height = sum(b["height"] for b in blocks) / len(blocks) if blocks else 20
    y_threshold = avg_height * 0.5
    
    sorted_blocks = sorted(blocks, key=lambda b: (b["y"], b["x"]))
    
    lines = []
    current_line = []
    current_y = None
    
    for block in sorted_blocks:
        if current_y is None:
            current_y = block["y"]
            current_line = [block]
        elif abs(block["y"] - current_y) <= y_threshold:
            current_line.append(block)
        else:
            current_line.sort(key=lambda b: b["x"])
            lines.append(current_line)
            current_line = [block]
            current_y = block["y"]
    
    if current_line:
        current_line.sort(key=lambda b: b["x"])
        lines.append(current_line)
    
    result = []
    for line in lines:
        result.extend(line)
    
    return result


def merge_extractions(marker_text: Optional[str], pymupdf_data: Dict) -> Dict:
    """Merge Marker formulas with PyMuPDF layout."""
    result = pymupdf_data.copy()
    
    if not marker_text:
        return result
    
    # Extract formulas from Marker output
    formulas = extract_formulas_from_markdown(marker_text)
    result["marker_formulas"] = formulas
    result["marker_text"] = marker_text
    
    # Try to match formulas to positions
    if formulas:
        for block in result["text_blocks"]:
            for formula in formulas:
                if is_formula_placeholder_match(block["text"], formula):
                    block["has_formula"] = True
                    block["formula_latex"] = formula
    
    return result


def extract_formulas_from_markdown(markdown: str) -> List[str]:
    """Extract LaTeX formulas from Markdown text."""
    formulas = []
    
    # Display math: $$ ... $$
    display = re.findall(r'\$\$(.*?)\$\$', markdown, re.DOTALL)
    formulas.extend(display)
    
    # Display math: \[ ... \]
    bracket = re.findall(r'\\\[(.*?)\\\]', markdown, re.DOTALL)
    formulas.extend(bracket)
    
    # Inline math: $ ... $ (but not $$)
    inline = re.findall(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', markdown)
    formulas.extend(inline)
    
    # equation environments
    envs = re.findall(
        r'\\begin\{(equation|align|gather)\*?\}(.*?)\\end\{\1\*?\}',
        markdown, re.DOTALL
    )
    formulas.extend([e[1] for e in envs])
    
    return [f.strip() for f in formulas if f.strip()]


def is_formula_placeholder_match(text: str, formula: str) -> bool:
    """Check if text might contain this formula."""
    # Simple heuristic: if formula variables appear in text
    import re
    formula_vars = set(re.findall(r'[a-zA-Z]', formula))
    text_vars = set(re.findall(r'[a-zA-Z]', text))
    
    if not formula_vars:
        return False
    
    overlap = len(formula_vars & text_vars) / len(formula_vars)
    return overlap > 0.5


# =============================================================================
# TRANSLATION FUNCTIONS
# =============================================================================

def translate_block_with_context(
    text: str,
    model: str,
    target_language: str,
    context,
    element_type: str = "text",
    use_openai: bool = False,
    openai_api_key: str = None
) -> str:
    """Translate a single block with context for consistency."""
    if not text.strip():
        return text
    
    # Apply glossary protection
    try:
        from glossary import apply_glossary
        protected_text, restore_glossary = apply_glossary(text, target_language)
    except ImportError:
        protected_text = text
        restore_glossary = lambda x: x
    
    if use_openai and openai_api_key:
        translated = translate_with_openai(protected_text, openai_api_key, target_language)
    else:
        # Use context-aware translation if available
        if context:
            try:
                from ollama_backend import translate_with_context
                translated = translate_with_context(
                    protected_text, model, target_language, context, element_type
                )
            except ImportError:
                from ollama_backend import translate_with_ollama
                translated = translate_with_ollama(
                    protected_text, model, None, target_language, element_type
                )
        else:
            from ollama_backend import translate_with_ollama
            translated = translate_with_ollama(
                protected_text, model, None, target_language, element_type
            )
    
    # Restore glossary placeholders
    result = restore_glossary(translated)
    
    return result


def translate_with_openai(text: str, api_key: str, target_language: str) -> str:
    """Translate using OpenAI API."""
    try:
        import openai
        
        # Get glossary context
        glossary_context = ""
        try:
            from glossary import get_glossary_context
            glossary_context = get_glossary_context(target_language)
        except ImportError:
            pass
        
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a scientific translator.

{glossary_context}

Translate to {target_language}. Output ONLY the translation.
Keep math formulas, author names, and abbreviations unchanged."""
                },
                {"role": "user", "content": text}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.warning(f"OpenAI translation failed: {e}")
        return text


def create_translated_page(
    merged_data: Dict,
    images_dir: Path,
    page_num: int,
    output_path: Path,
    target_language: str,
    model: str,
    use_openai: bool,
    openai_api_key: str,
    progress_callback,
    translation_context=None
) -> bool:
    """Create a translated PDF page with preserved layout."""
    try:
        doc = fitz.open()
        page = doc.new_page(
            width=merged_data["page_width"],
            height=merged_data["page_height"]
        )
        
        # Insert images first (background layer)
        for img_idx, img in enumerate(merged_data.get("images", [])):
            try:
                img_path = images_dir / f"page{page_num}_img{img_idx}.{img['ext']}"
                img_path.write_bytes(img["data"])
                
                rect = fitz.Rect(
                    img["x"], img["y"],
                    img["x"] + img["width"],
                    img["y"] + img["height"]
                )
                page.insert_image(rect, filename=str(img_path))
            except Exception as e:
                logger.warning(f"Could not insert image: {e}")
        
        # Translate and insert text blocks
        blocks = merged_data.get("text_blocks", [])
        total_blocks = len(blocks)
        
        for idx, block in enumerate(blocks):
            if progress_callback and total_blocks > 0:
                progress_callback(idx + 1, total_blocks, f"Block {idx + 1}/{total_blocks}")
            
            text = block["text"]
            
            # Determine element type for specialized translation
            element_type = "text"
            text_lower = text.lower()
            if text_lower.startswith(("figure", "fig.", "abbildung", "abb.")):
                element_type = "figure_caption"
            elif text_lower.startswith(("table", "tab.", "tabelle")):
                element_type = "table_content"
            
            # Translate with context
            translated = translate_block_with_context(
                text, model, target_language, translation_context,
                element_type, use_openai, openai_api_key
            )
            
            # Reflow translated text
            translated = normalize_and_reflow(translated)
            
            # Insert translated text
            try:
                rect = fitz.Rect(
                    block["x"], block["y"],
                    block["x"] + block["width"],
                    block["y"] + block["height"]
                )
                
                font_size = min(block.get("font_size", 10), 14)
                
                page.insert_textbox(
                    rect,
                    translated,
                    fontsize=font_size,
                    fontname="helv",
                    align=fitz.TEXT_ALIGN_LEFT
                )
            except Exception as e:
                logger.warning(f"Could not insert text block: {e}")
        
        doc.save(str(output_path))
        doc.close()
        return True
        
    except Exception as e:
        logger.exception(f"Failed to create translated page: {e}")
        return False


def merge_pdfs(page_paths: List[Path], output_path: Path) -> bool:
    """Merge multiple PDF pages into one document."""
    try:
        output_doc = fitz.open()
        
        for page_path in page_paths:
            if page_path.exists():
                page_doc = fitz.open(str(page_path))
                output_doc.insert_pdf(page_doc)
                page_doc.close()
        
        output_doc.save(str(output_path))
        output_doc.close()
        
        logger.info(f"Merged {len(page_paths)} pages into {output_path}")
        return True
    except Exception as e:
        logger.exception(f"Failed to merge PDFs: {e}")
        return False


def post_build_sanity_check(pdf_path: Path, source_page_count: int) -> Dict:
    """Post-build sanity check for the generated PDF."""
    report = {
        "page_count": 0,
        "blank_pages_removed": 0,
        "garbage_chars_found": 0,
        "warnings": []
    }
    
    if not pdf_path.exists():
        report["warnings"].append("Output PDF does not exist")
        return report
    
    doc = fitz.open(str(pdf_path))
    report["page_count"] = len(doc)
    
    # Check for garbage characters
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    
    report["garbage_chars_found"] = count_garbage_chars(full_text)
    
    if report["garbage_chars_found"] > 0:
        report["warnings"].append(f"Found {report['garbage_chars_found']} garbage characters")
    
    if report["page_count"] > source_page_count + 2:
        report["warnings"].append(f"Page count increased: {source_page_count} → {report['page_count']}")
    
    doc.close()
    return report


# =============================================================================
# MAIN TRANSLATION PIPELINE
# =============================================================================

def translate_pdf_unified(
    input_pdf: str,
    output_dir: str,
    model: str,
    target_language: str,
    progress_callback=None,
    use_openai: bool = False,
    openai_api_key: str = None
) -> Tuple[Optional[str], str]:
    """
    Unified translation pipeline - combines all methods for best results.
    
    Features:
    - Smart extraction (Marker → Nougat fallback)
    - Glossary for consistent terminology
    - Context window for translation consistency
    - VRAM-efficient model management
    
    Returns (output_path, status_message)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    marker_dir = output_dir / "marker"
    marker_dir.mkdir(exist_ok=True)
    
    translated_dir = output_dir / "translated_pages"
    translated_dir.mkdir(exist_ok=True)
    
    # Initialize translation context for consistency
    translation_context = get_translation_context(target_language)
    
    extraction_methods_used = set()
    
    try:
        # Step 1: Split PDF
        if progress_callback:
            progress_callback(2, 100, "Splitting PDF into pages...")
        
        page_paths = split_pdf_into_pages(input_pdf, pages_dir)
        total_pages = len(page_paths)
        
        if total_pages == 0:
            return None, "❌ Could not extract pages from PDF"
        
        translated_page_paths = []
        
        # Step 2: Process each page
        for page_num, page_path in enumerate(page_paths, 1):
            base_progress = 5 + int(90 * (page_num - 1) / total_pages)
            page_progress_range = 90 / total_pages
            
            if progress_callback:
                progress_callback(base_progress, 100, f"Page {page_num}/{total_pages}: Extracting...")
            
            # 2a: Prepare for extraction (unload Ollama)
            model_manager.prepare_for_marker()
            
            # 2b: Smart extraction (Marker → Nougat fallback)
            page_marker_dir = marker_dir / f"page_{page_num:03d}"
            page_marker_dir.mkdir(exist_ok=True)
            
            marker_text, method = smart_extract_text(str(page_path), page_marker_dir, timeout=120)
            extraction_methods_used.add(method)
            
            if marker_text:
                logger.info(f"Page {page_num}: {method} extracted {len(marker_text)} chars")
            else:
                logger.warning(f"Page {page_num}: No text extracted")
            
            # 2c: Extract with PyMuPDF (layout/images)
            pymupdf_data = extract_with_pymupdf(page_path)
            logger.info(f"Page {page_num}: PyMuPDF found {len(pymupdf_data['images'])} images, {len(pymupdf_data['text_blocks'])} blocks")
            
            # 2d: Merge extractions
            merged_data = merge_extractions(marker_text, pymupdf_data)
            
            # 2e: Prepare for translation (load Ollama)
            if not use_openai:
                model_manager.prepare_for_translation(model)
            
            if progress_callback:
                progress_callback(
                    base_progress + int(page_progress_range * 0.3), 100,
                    f"Page {page_num}/{total_pages}: Translating..."
                )
            
            # 2f: Create translated page with context
            translated_page_path = translated_dir / f"translated_{page_num:03d}.pdf"
            
            def page_block_progress(current, total, msg):
                if progress_callback:
                    sub_progress = base_progress + int(page_progress_range * (0.3 + 0.7 * current / max(total, 1)))
                    progress_callback(sub_progress, 100, f"Page {page_num}: {msg}")
            
            success = create_translated_page(
                merged_data,
                images_dir,
                page_num,
                translated_page_path,
                target_language,
                model,
                use_openai,
                openai_api_key,
                page_block_progress,
                translation_context
            )
            
            if success and translated_page_path.exists():
                translated_page_paths.append(translated_page_path)
                logger.info(f"Page {page_num}: Translated successfully")
            else:
                logger.warning(f"Page {page_num}: Translation failed")
        
        # Step 3: Merge all pages
        if progress_callback:
            progress_callback(96, 100, "Merging pages...")
        
        output_pdf = output_dir / "translated.pdf"
        merge_success = merge_pdfs(translated_page_paths, output_pdf)
        
        if merge_success and output_pdf.exists():
            # Step 4: Post-build sanity check
            if progress_callback:
                progress_callback(98, 100, "Running sanity check...")
            
            sanity_report = post_build_sanity_check(output_pdf, total_pages)
            
            if progress_callback:
                progress_callback(100, 100, "Complete!")
            
            # Copy to stable location
            stable_output = Path(__file__).parent / "output" / "translated_unified.pdf"
            stable_output.parent.mkdir(exist_ok=True)
            shutil.copy2(output_pdf, stable_output)
            
            # Build status message
            methods_str = ", ".join(sorted(extraction_methods_used - {"none"})) or "PyMuPDF only"
            warnings_text = ""
            if sanity_report["warnings"]:
                warnings_text = "\n⚠️ " + "\n⚠️ ".join(sanity_report["warnings"])
            
            return str(output_pdf), f"""✅ Unified Translation Complete!

📄 **{sanity_report['page_count']} pages** (source: {total_pages})
🔬 Extraction: {methods_str}
📐 PyMuPDF for layout preservation
🖼️ Images extracted and preserved
🌐 Translated to {target_language}
📚 Glossary: Consistent terminology
🔗 Context: Translation consistency
🔍 Garbage chars: {sanity_report['garbage_chars_found']}{warnings_text}

📁 Also saved to: {stable_output}"""
        
        return None, "❌ Failed to merge translated pages"
        
    except Exception as e:
        logger.exception(f"Unified translation failed: {e}")
        return None, f"❌ Translation failed: {str(e)}"


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python unified_translator.py input.pdf [output_dir] [target_language]")
        sys.exit(1)
    
    input_pdf = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output_unified"
    target_lang = sys.argv[3] if len(sys.argv) > 3 else "German"
    
    result, status = translate_pdf_unified(
        input_pdf,
        output_dir,
        model="qwen2.5:7b",
        target_language=target_lang,
        progress_callback=lambda c, t, m: print(f"[{c:3d}%] {m}")
    )
    
    print(f"\n{'='*50}")
    print(status)
    if result:
        print(f"\nOutput: {result}")
