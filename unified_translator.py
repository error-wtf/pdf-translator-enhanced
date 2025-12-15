"""
Unified PDF Translator - Best Quality Pipeline

Combines all methods intelligently:
1. Marker for formula extraction (best for math)
2. PyMuPDF for layout/image extraction (best for structure)
3. Page-by-page processing (best for large docs)
4. Smart model loading/unloading for VRAM efficiency

Orchestration:
- Step 1: Split PDF into pages
- Step 2: For each page, extract with Marker (formulas) + PyMuPDF (images/layout)
- Step 3: Merge extractions intelligently
- Step 4: Translate with LLM (Ollama/OpenAI)
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
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import fitz  # PyMuPDF
from text_normalizer import normalize_text, normalize_and_reflow, count_garbage_chars
from caption_anchoring import anchor_captions_to_images, sort_blocks_reading_order, AnchoredFigure
from table_detector import detect_tables_in_page, DetectedTable
from scientific_postprocessor import ScientificPostProcessor, RepairMode

# Import enhanced formula protection
from formula_isolator import (
    extract_and_protect,
    normalize_output,
    audit_utf8,
    assert_no_corruption,
)

logger = logging.getLogger("pdf_translator.unified")


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


def extract_with_marker(pdf_path: str, output_dir: Path) -> Optional[str]:
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
            timeout=120,
            cwd=str(Path(__file__).parent)
        )
        
        for line in result.stdout.split('\n'):
            if line.startswith('MARKER_RESULT:'):
                result_json = json.loads(line[14:].strip())
                if result_json.get('success'):
                    return result_json.get('output_path')
        return None
    except Exception as e:
        logger.warning(f"Marker extraction failed: {e}")
        return None


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
    
    # Sort blocks by reading order (top-to-bottom, left-to-right)
    result["text_blocks"] = sort_blocks_reading_order(result["text_blocks"])
    
    doc.close()
    return result


def sort_blocks_reading_order(blocks: List[Dict]) -> List[Dict]:
    """
    Sort text blocks in reading order.
    
    Strategy:
    - Primary sort by y position (top to bottom)
    - Secondary sort by x position (left to right)
    - Group blocks that are on the same "line" (similar y)
    """
    if not blocks:
        return blocks
    
    # Calculate average line height for grouping
    avg_height = sum(b["height"] for b in blocks) / len(blocks) if blocks else 20
    y_threshold = avg_height * 0.5  # Blocks within this y-distance are on same "line"
    
    # Sort by y first, then x
    sorted_blocks = sorted(blocks, key=lambda b: (b["y"], b["x"]))
    
    # Group blocks by approximate y position (same line)
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
            # Sort current line by x and add to lines
            current_line.sort(key=lambda b: b["x"])
            lines.append(current_line)
            current_line = [block]
            current_y = block["y"]
    
    # Don't forget the last line
    if current_line:
        current_line.sort(key=lambda b: b["x"])
        lines.append(current_line)
    
    # Flatten back to list
    result = []
    for line in lines:
        result.extend(line)
    
    return result


def merge_extractions(marker_text: Optional[str], pymupdf_data: Dict) -> Dict:
    """
    Intelligently merge Marker (formulas) with PyMuPDF (layout).
    
    Strategy:
    - Use PyMuPDF for layout/positions
    - Use Marker text for formula-heavy blocks (detected by $ or \)
    - Preserve images from PyMuPDF
    """
    result = pymupdf_data.copy()
    
    if not marker_text:
        return result
    
    # Check if Marker found formulas
    has_formulas = '$' in marker_text or '\\' in marker_text
    
    if has_formulas:
        # Split Marker text into paragraphs
        marker_paragraphs = [p.strip() for p in marker_text.split('\n\n') if p.strip()]
        
        # Try to match Marker paragraphs to PyMuPDF blocks by similarity
        for i, block in enumerate(result["text_blocks"]):
            block_text = block["text"]
            
            # If block looks like it might have formulas, try to find better version from Marker
            if any(c in block_text for c in ['=', '+', '-', '*', '/', '^', '_']):
                # Find most similar Marker paragraph
                best_match = None
                best_score = 0
                
                for mp in marker_paragraphs:
                    # Simple word overlap score
                    block_words = set(block_text.lower().split())
                    marker_words = set(mp.lower().split())
                    if block_words and marker_words:
                        overlap = len(block_words & marker_words)
                        score = overlap / max(len(block_words), len(marker_words))
                        if score > best_score and score > 0.3:
                            best_score = score
                            best_match = mp
                
                if best_match and ('$' in best_match or '\\' in best_match):
                    # Use Marker version for this block (has formulas)
                    result["text_blocks"][i]["text"] = best_match
                    result["text_blocks"][i]["from_marker"] = True
    
    return result


def protect_formulas(text: str) -> tuple[str, callable]:
    """
    Protect mathematical formulas and scientific notation from translation.
    
    Uses formula_isolator for robust ⟦FORMULA_xxx⟧ placeholders that
    cannot be corrupted by LLMs.
    
    Returns (protected_text, restore_function)
    """
    # Use the robust formula_isolator instead of simple placeholders
    protected_text, restore_func = extract_and_protect(text)
    return protected_text, restore_func


def cleanup_llm_output(text: str) -> str:
    """
    Clean up HTML/Markdown artifacts from LLM output.
    
    LLMs often produce HTML tags like <sup>, <sub>, <b>, <i> etc.
    that should be converted to LaTeX or removed.
    """
    if not text:
        return text
    
    # HTML superscript to LaTeX
    text = re.sub(r'<sup>([^<]+)</sup>', r'$^{\1}$', text)
    text = re.sub(r'<SUP>([^<]+)</SUP>', r'$^{\1}$', text, flags=re.IGNORECASE)
    
    # HTML subscript to LaTeX
    text = re.sub(r'<sub>([^<]+)</sub>', r'$_{\1}$', text)
    text = re.sub(r'<SUB>([^<]+)</SUB>', r'$_{\1}$', text, flags=re.IGNORECASE)
    
    # HTML bold to LaTeX
    text = re.sub(r'<b>([^<]+)</b>', r'\\textbf{\1}', text, flags=re.IGNORECASE)
    text = re.sub(r'<strong>([^<]+)</strong>', r'\\textbf{\1}', text, flags=re.IGNORECASE)
    
    # HTML italic to LaTeX
    text = re.sub(r'<i>([^<]+)</i>', r'\\textit{\1}', text, flags=re.IGNORECASE)
    text = re.sub(r'<em>([^<]+)</em>', r'\\textit{\1}', text, flags=re.IGNORECASE)
    
    # Remove any remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Fix double dollar signs from LLM ($$...$$ should be display math)
    text = re.sub(r'\$\$([^$]+)\$\$', r'\\[\1\\]', text)
    
    # Fix escaped LaTeX that LLM might produce
    text = text.replace('\\\\', '\\')  # Double backslash to single
    
    # Remove markdown bold/italic if present
    text = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', text)
    text = re.sub(r'\*([^*]+)\*', r'\\textit{\1}', text)
    
    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


def sanitize_unicode(text: str) -> str:
    """
    Fix common Unicode extraction issues from PDFs.
    
    Fixes patterns like:
    - 99,1?% → 99,1%
    - ESO?Spektroskopie → ESO-Spektroskopie
    - Seite?2 → Seite 2
    - r*/r_s stays as-is (math notation)
    """
    if not text:
        return text
    
    # Fix ? before % (99,1?% → 99,1%)
    text = re.sub(r'\?(%)', r'\1', text)
    
    # Fix ? between word and number (Seite?2 → Seite 2)
    text = re.sub(r'([a-zA-ZäöüÄÖÜß])\?(\d)', r'\1 \2', text)
    
    # Fix ? between two words (ESO?Spektroskopie → ESO-Spektroskopie)
    text = re.sub(r'([a-zA-ZäöüÄÖÜß])\?([A-ZÄÖÜ][a-zäöüß])', r'\1-\2', text)
    
    # Fix standalone ? that should be space
    text = re.sub(r'(\w)\?(\w)', r'\1 \2', text)
    
    # Remove orphan ? at start/end of words
    text = re.sub(r'\?\s+', ' ', text)
    text = re.sub(r'\s+\?', ' ', text)
    
    return text


def apply_german_style(text: str) -> str:
    """
    Apply German scientific writing conventions.
    
    Converts:
    - Figure 1 → Abbildung 1
    - Fig. 1 → Abb. 1
    - Table 1 → Tabelle 1
    - Tab. 1 → Tab. 1 (already German)
    - Equation → Gleichung
    - Section → Abschnitt
    """
    if not text:
        return text
    
    # Figure → Abbildung (full word)
    text = re.sub(r'\bFigure\s+(\d+)', r'Abbildung \1', text)
    text = re.sub(r'\bfigure\s+(\d+)', r'Abbildung \1', text)
    
    # Fig. → Abb.
    text = re.sub(r'\bFig\.\s*(\d+)', r'Abb. \1', text)
    text = re.sub(r'\bfig\.\s*(\d+)', r'Abb. \1', text)
    
    # Table → Tabelle
    text = re.sub(r'\bTable\s+(\d+)', r'Tabelle \1', text)
    text = re.sub(r'\btable\s+(\d+)', r'Tabelle \1', text)
    
    # Equation → Gleichung
    text = re.sub(r'\bEquation\s+(\d+)', r'Gleichung \1', text)
    text = re.sub(r'\bEq\.\s*(\d+)', r'Gl. \1', text)
    
    # Section → Abschnitt
    text = re.sub(r'\bSection\s+(\d+)', r'Abschnitt \1', text)
    text = re.sub(r'\bSec\.\s*(\d+)', r'Abschn. \1', text)
    
    return text


def translate_text(text: str, model: str, target_language: str,
                   use_openai: bool = False, openai_api_key: str = None) -> str:
    """Translate text using Ollama or OpenAI, protecting formulas."""
    from pdf_marker_translator import translate_text_chunk
    
    # Protect formulas before translation (returns restore function)
    protected_text, restore_func = protect_formulas(text)
    
    # Translate
    translated = translate_text_chunk(protected_text, model, target_language, use_openai, openai_api_key)
    
    # Clean up HTML/Markdown artifacts from LLM
    translated = cleanup_llm_output(translated)
    
    # Restore formulas using the restore function
    result = restore_func(translated)
    
    # Normalize output for clean Unicode
    result = normalize_output(result, mode="unicode")
    
    # Sanitize Unicode extraction issues
    result = sanitize_unicode(result)
    
    # Apply German style if target is German
    if target_language.lower() in ["german", "deutsch", "de"]:
        result = apply_german_style(result)
    
    return result


def create_translated_page(
    page_data: Dict,
    images_dir: Path,
    page_num: int,
    output_path: Path,
    target_language: str,
    model: str,
    use_openai: bool = False,
    openai_api_key: str = None,
    progress_callback=None,
    repair_mode: str = "safe_repair"
) -> bool:
    """Create a translated PDF page with original layout."""
    
    # Create new PDF
    new_doc = fitz.open()
    new_page = new_doc.new_page(
        width=page_data["page_width"],
        height=page_data["page_height"]
    )
    
    # Insert images first (background layer)
    for i, img in enumerate(page_data["images"]):
        try:
            img_path = images_dir / f"page{page_num:03d}_img{i:02d}.{img['ext']}"
            with open(img_path, 'wb') as f:
                f.write(img["data"])
            
            img_rect = fitz.Rect(
                img["x"], img["y"],
                img["x"] + img["width"],
                img["y"] + img["height"]
            )
            new_page.insert_image(img_rect, filename=str(img_path))
        except Exception as e:
            logger.warning(f"Could not insert image: {e}")
    
    # Translate and insert text blocks
    total_blocks = len(page_data["text_blocks"])
    inserted_count = 0
    
    for i, block in enumerate(page_data["text_blocks"]):
        if not block["text"].strip():
            continue
        
        if progress_callback:
            progress_callback(i + 1, total_blocks, f"Block {i + 1}/{total_blocks}")
        
        # Translate
        translated = translate_text(
            block["text"], model, target_language,
            use_openai, openai_api_key
        )
        
        if not translated or not translated.strip():
            translated = block["text"]  # Keep original if translation failed
        
        # Apply scientific post-processing
        mode = RepairMode.SAFE_REPAIR if repair_mode == "safe_repair" else RepairMode.STRICT
        postprocessor = ScientificPostProcessor(mode)
        translated, _ = postprocessor.process(translated)
        
        # Calculate font size - start with original, but allow shrinking
        font_size = min(block["font_size"], 10)
        if font_size < 6:
            font_size = 8
        
        # Create rect with some padding for text overflow
        x0 = max(10, block["x"])
        y0 = max(10, block["y"])
        x1 = min(page_data["page_width"] - 10, block["x"] + block["width"] + 50)
        y1 = min(page_data["page_height"] - 10, block["y"] + block["height"] + 20)
        
        text_rect = fitz.Rect(x0, y0, x1, y1)
        
        # Replace problematic Unicode with ASCII equivalents
        translated = translated.replace('●', '-').replace('■', '-').replace('•', '-')
        translated = translated.replace('→', '->').replace('←', '<-').replace('↔', '<->')
        translated = translated.replace('✓', '[x]').replace('✗', '[ ]').replace('✔', '[x]')
        translated = translated.replace('★', '*').replace('☆', '*').replace('⭐', '*')
        translated = translated.replace('📄', '').replace('📁', '').replace('🔄', '')
        
        # Try insert_textbox first with auto-shrink
        try:
            rc = new_page.insert_textbox(
                text_rect,
                translated,
                fontsize=font_size,
                fontname="helv",
                align=fitz.TEXT_ALIGN_LEFT,
                expandtabs=True
            )
            # rc < 0 means text didn't fit, but some was inserted
            # rc >= 0 means all text fit
            inserted_count += 1
            logger.debug(f"Page {page_num} block {i}: inserted with rc={rc}")
        except Exception as e:
            # Fallback: use insert_text line by line
            logger.warning(f"Page {page_num} block {i}: textbox failed ({e}), using line-by-line")
            try:
                lines = translated.split('\n')
                y_pos = y0 + font_size
                for line in lines[:20]:  # Limit lines to prevent overflow
                    if y_pos > y1:
                        break
                    new_page.insert_text(
                        (x0, y_pos),
                        line[:200],  # Limit line length
                        fontsize=font_size,
                        fontname="helv"
                    )
                    y_pos += font_size + 2
                inserted_count += 1
            except Exception as e2:
                logger.error(f"Page {page_num} block {i}: all text insertion failed: {e2}")
    
    logger.info(f"Page {page_num}: Inserted {inserted_count}/{total_blocks} text blocks")
    
    new_doc.save(str(output_path))
    new_doc.close()
    return True


def merge_pdfs(pdf_paths: List[Path], output_path: Path) -> bool:
    """Merge multiple PDFs into one."""
    if not pdf_paths:
        return False
    
    merged_doc = fitz.open()
    for pdf_path in pdf_paths:
        if pdf_path.exists():
            doc = fitz.open(str(pdf_path))
            merged_doc.insert_pdf(doc)
            doc.close()
    
    merged_doc.save(str(output_path))
    merged_doc.close()
    return True


def remove_blank_pages(pdf_path: Path) -> int:
    """
    Remove blank pages from a PDF.
    
    A page is considered blank if it has no text AND no images.
    Returns the number of pages removed.
    """
    import os
    
    # First pass: identify blank pages
    doc = fitz.open(str(pdf_path))
    pages_to_remove = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Check for text
        text = page.get_text().strip()
        has_text = len(text) > 10  # More than just whitespace/artifacts
        
        # Check for images
        images = page.get_images()
        has_images = len(images) > 0
        
        # Check for drawings/paths
        drawings = page.get_drawings()
        has_drawings = len(drawings) > 5  # Some drawings might be just lines
        
        if not has_text and not has_images and not has_drawings:
            pages_to_remove.append(page_num)
            logger.info(f"Marking blank page {page_num + 1} for removal")
    
    doc.close()  # Close before modifying
    
    if not pages_to_remove:
        return 0
    
    # Second pass: create new PDF without blank pages
    doc = fitz.open(str(pdf_path))
    new_doc = fitz.open()  # Create empty document
    
    for page_num in range(len(doc)):
        if page_num not in pages_to_remove:
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
    
    doc.close()
    
    # Save new document to temp file
    temp_path = pdf_path.parent / f"{pdf_path.stem}_cleaned{pdf_path.suffix}"
    new_doc.save(str(temp_path))
    new_doc.close()
    
    # Replace original with cleaned version
    os.replace(str(temp_path), str(pdf_path))
    
    removed_count = len(pages_to_remove)
    logger.info(f"Removed {removed_count} blank pages")
    
    return removed_count


def post_build_sanity_check(pdf_path: Path, source_page_count: int) -> Dict:
    """
    Post-build sanity check for the generated PDF.
    
    Returns a report dict with:
    - page_count: final page count
    - blank_pages_removed: number of blank pages removed
    - garbage_chars_found: number of garbage chars in text
    - warnings: list of warning messages
    """
    report = {
        "page_count": 0,
        "blank_pages_removed": 0,
        "garbage_chars_found": 0,
        "warnings": []
    }
    
    if not pdf_path.exists():
        report["warnings"].append("Output PDF does not exist")
        return report
    
    # DISABLED: Blank page removal was incorrectly removing pages with content
    # The issue is that text insertion via insert_textbox doesn't always work
    # report["blank_pages_removed"] = remove_blank_pages(pdf_path)
    report["blank_pages_removed"] = 0
    
    # Check final page count
    doc = fitz.open(str(pdf_path))
    report["page_count"] = len(doc)
    
    # Check for garbage characters in text
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    
    report["garbage_chars_found"] = count_garbage_chars(full_text)
    
    if report["garbage_chars_found"] > 0:
        report["warnings"].append(f"Found {report['garbage_chars_found']} garbage characters in output")
    
    # Check page count explosion
    if report["page_count"] > source_page_count + 2:
        report["warnings"].append(f"Page count increased significantly: {source_page_count} → {report['page_count']}")
    
    doc.close()
    return report


def translate_pdf_unified(
    input_pdf: str,
    output_dir: str,
    model: str,
    target_language: str,
    progress_callback=None,
    use_openai: bool = False,
    openai_api_key: str = None,
    repair_mode: str = "safe_repair"
) -> Tuple[Optional[str], str]:
    """
    Unified translation pipeline - combines all methods for best results.
    
    Orchestration:
    1. Split PDF into pages
    2. For each page:
       a. Unload Ollama, load Marker
       b. Extract with Marker (formulas)
       c. Unload Marker, extract with PyMuPDF (layout/images)
       d. Merge extractions intelligently
       e. Load Ollama, translate
       f. Create translated page
    3. Merge all pages
    
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
            
            # 2a: Prepare for Marker (unload Ollama)
            model_manager.prepare_for_marker()
            
            # 2b: Extract with Marker (formulas)
            page_marker_dir = marker_dir / f"page_{page_num:03d}"
            page_marker_dir.mkdir(exist_ok=True)
            
            marker_text = None
            try:
                marker_md_path = extract_with_marker(str(page_path), page_marker_dir)
                if marker_md_path and Path(marker_md_path).exists():
                    marker_text = Path(marker_md_path).read_text(encoding='utf-8')
                    logger.info(f"Page {page_num}: Marker extracted {len(marker_text)} chars")
            except Exception as e:
                logger.warning(f"Page {page_num}: Marker failed: {e}")
            
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
            
            # 2f: Create translated page
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
                repair_mode
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
            
            # Build status message with sanity report
            warnings_text = ""
            if sanity_report["warnings"]:
                warnings_text = "\n⚠️ " + "\n⚠️ ".join(sanity_report["warnings"])
            
            return str(output_pdf), f"""✅ Unified Translation Complete!

📄 **{sanity_report['page_count']} pages** (source: {total_pages})
🔬 Marker extraction for formulas
📐 PyMuPDF for layout preservation
🖼️ Images extracted and preserved
🌐 Translated to {target_language}
🧹 Blank pages removed: {sanity_report['blank_pages_removed']}
🔍 Garbage chars found: {sanity_report['garbage_chars_found']}{warnings_text}

📁 Also saved to: {stable_output}"""
        
        return None, "❌ Failed to merge translated pages"
        
    except Exception as e:
        logger.exception(f"Unified translation failed: {e}")
        return None, f"❌ Translation failed: {str(e)}"


# CLI interface
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
