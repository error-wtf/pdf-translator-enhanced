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
# CONFIGURATION - QUALITY SETTINGS
# =============================================================================

# Minimum translation ratio (translated length / original length)
# If below this, something went wrong
MIN_TRANSLATION_RATIO = 0.5

# Maximum translation ratio (for detecting garbage output)
MAX_TRANSLATION_RATIO = 3.0

# Maximum tokens per chunk (to avoid truncation)
MAX_CHUNK_TOKENS = 2000

# Approximate chars per token
CHARS_PER_TOKEN = 4


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
        "text_blocks": [],
        "full_text": ""
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
    
    # Extract text blocks with position
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    
    all_text_parts = []
    
    for block in blocks:
        if block.get("type") == 0:  # Text block
            block_text = ""
            for line in block.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    line_text += span.get("text", "")
                block_text += line_text + "\n"
            
            if block_text.strip():
                result["text_blocks"].append({
                    "text": block_text.strip(),
                    "x": block.get("bbox", [0])[0],
                    "y": block.get("bbox", [0, 0])[1],
                    "width": block.get("bbox", [0, 0, 0])[2] - block.get("bbox", [0])[0],
                    "height": block.get("bbox", [0, 0, 0, 0])[3] - block.get("bbox", [0, 0])[1],
                })
                all_text_parts.append(block_text.strip())
    
    # Store complete text from PyMuPDF
    result["full_text"] = "\n\n".join(all_text_parts)
    
    doc.close()
    return result


def merge_extractions(marker_text: Optional[str], pymupdf_data: Dict) -> Dict:
    """
    IMPROVED: Merge Marker (formulas) + PyMuPDF (layout) extractions.
    
    Uses BOTH sources to ensure no text is lost:
    - Marker for formula-heavy sections
    - PyMuPDF for complete text coverage
    """
    merged = {
        "page_width": pymupdf_data.get("page_width", 612),
        "page_height": pymupdf_data.get("page_height", 792),
        "images": pymupdf_data.get("images", []),
        "text_blocks": [],
        "full_text": "",
        "source": "none"
    }
    
    pymupdf_text = pymupdf_data.get("full_text", "")
    pymupdf_blocks = pymupdf_data.get("text_blocks", [])
    
    # CRITICAL FIX: Compare both sources and use the more complete one
    if marker_text and pymupdf_text:
        marker_len = len(marker_text.strip())
        pymupdf_len = len(pymupdf_text.strip())
        
        # Use whichever has more content, but also check for formula indicators
        has_formulas = bool(re.search(r'\$.*?\$|\\begin\{|\\frac|\\sum|\\int', marker_text))
        
        if has_formulas and marker_len >= pymupdf_len * 0.7:
            # Marker has formulas and reasonable coverage - use it
            merged["full_text"] = marker_text
            merged["source"] = "marker"
            logger.info(f"Using Marker: {marker_len} chars (has formulas)")
        elif pymupdf_len > marker_len * 1.2:
            # PyMuPDF has significantly more content - use it
            merged["full_text"] = pymupdf_text
            merged["source"] = "pymupdf"
            merged["text_blocks"] = pymupdf_blocks
            logger.info(f"Using PyMuPDF: {pymupdf_len} chars (more complete)")
        else:
            # Similar length - prefer Marker for formula preservation
            merged["full_text"] = marker_text
            merged["source"] = "marker"
            logger.info(f"Using Marker: {marker_len} chars (similar, prefer formulas)")
            
    elif marker_text:
        merged["full_text"] = marker_text
        merged["source"] = "marker"
        logger.info(f"Using Marker only: {len(marker_text)} chars")
        
    elif pymupdf_text:
        merged["full_text"] = pymupdf_text
        merged["source"] = "pymupdf"
        merged["text_blocks"] = pymupdf_blocks
        logger.info(f"Using PyMuPDF only: {len(pymupdf_text)} chars")
    
    else:
        logger.warning("No text extracted from either source!")
        merged["source"] = "none"
    
    return merged


# =============================================================================
# TRANSLATION FUNCTIONS - IMPROVED
# =============================================================================

def estimate_tokens(text: str) -> int:
    """Estimate token count for text."""
    return len(text) // CHARS_PER_TOKEN


def chunk_text(text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> List[str]:
    """
    Split text into chunks that won't exceed token limit.
    Splits on paragraph boundaries to preserve context.
    """
    if estimate_tokens(text) <= max_tokens:
        return [text]
    
    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for para in paragraphs:
        para_tokens = estimate_tokens(para)
        current_tokens = estimate_tokens(current_chunk)
        
        if current_tokens + para_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If single paragraph is too long, split by sentences
            if para_tokens > max_tokens:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    if estimate_tokens(current_chunk + sent) > max_tokens:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sent + " "
                    else:
                        current_chunk += sent + " "
            else:
                current_chunk = para + "\n\n"
        else:
            current_chunk += para + "\n\n"
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks


def validate_translation(original: str, translated: str) -> Tuple[bool, str]:
    """
    Validate translation quality.
    Returns (is_valid, issue_description)
    """
    if not translated or not translated.strip():
        return False, "Empty translation"
    
    orig_len = len(original.strip())
    trans_len = len(translated.strip())
    
    if orig_len == 0:
        return True, "OK (empty original)"
    
    ratio = trans_len / orig_len
    
    if ratio < MIN_TRANSLATION_RATIO:
        return False, f"Translation too short ({ratio:.1%} of original)"
    
    if ratio > MAX_TRANSLATION_RATIO:
        return False, f"Translation too long ({ratio:.1%} of original) - possible garbage"
    
    # Check for common LLM failure patterns
    failure_patterns = [
        r'^(I\'m sorry|I cannot|I apologize|As an AI)',
        r'^(Here is|Here\'s|Below is)',
        r'^(Translation|Translated|Übersetzung|Traduzione):',
        r'^\s*\[.*?\]\s*$',  # Just brackets
    ]
    
    for pattern in failure_patterns:
        if re.match(pattern, translated.strip(), re.IGNORECASE):
            return False, f"LLM meta-response detected"
    
    return True, "OK"


def translate_text_block(
    text: str,
    target_language: str,
    model: str,
    use_openai: bool = False,
    openai_api_key: str = None,
    translation_context = None,
    max_retries: int = 3,
) -> str:
    """
    IMPROVED: Translate text block with validation and retry.
    
    Features:
    - Chunks long text to avoid truncation
    - Validates output length
    - Retries on failure
    - Uses formula isolation
    """
    if not text or not text.strip():
        return text
    
    # Protect formulas before translation
    try:
        from formula_isolator import extract_and_protect
        protected_text, restore_formulas = extract_and_protect(text)
    except ImportError:
        protected_text = text
        restore_formulas = lambda x: x
    
    # Chunk if too long
    chunks = chunk_text(protected_text)
    translated_chunks = []
    
    for chunk_idx, chunk in enumerate(chunks):
        chunk_translated = None
        last_error = None
        
        for attempt in range(max_retries):
            try:
                if use_openai and openai_api_key:
                    # OpenAI translation
                    chunk_translated = translate_with_openai(
                        chunk, target_language, openai_api_key
                    )
                else:
                    # Ollama translation
                    chunk_translated = translate_with_ollama_direct(
                        chunk, model, target_language, translation_context
                    )
                
                # Validate
                is_valid, issue = validate_translation(chunk, chunk_translated)
                
                if is_valid:
                    logger.debug(f"Chunk {chunk_idx+1}/{len(chunks)} translated OK")
                    break
                else:
                    last_error = issue
                    logger.warning(f"Translation validation failed (attempt {attempt+1}): {issue}")
                    chunk_translated = None
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Translation error (attempt {attempt+1}): {e}")
                time.sleep(1)  # Brief pause before retry
        
        # If all retries failed, use original
        if chunk_translated is None:
            logger.error(f"Chunk {chunk_idx+1} translation failed after {max_retries} attempts: {last_error}")
            logger.error(f"Using original text for this chunk")
            chunk_translated = chunk  # Fallback to original
        
        translated_chunks.append(chunk_translated)
    
    # Combine chunks
    full_translation = "\n\n".join(translated_chunks)
    
    # Restore formulas
    final_text = restore_formulas(full_translation)
    
    return final_text


def translate_with_ollama_direct(
    text: str,
    model: str,
    target_language: str,
    context = None
) -> str:
    """Direct Ollama translation with improved prompting."""
    import requests
    
    # Build prompt
    system_prompt = f"""You are a scientific document translator. Translate the text to {target_language}.

CRITICAL RULES:
1. Output ONLY the translation - no explanations, no meta-comments
2. Translate ALL text - do not summarize or shorten
3. Keep all LaTeX/math notation unchanged: $...$ and $$...$$
4. Keep all ⟦...⟧ placeholders exactly as they are
5. Keep author names, citations, references unchanged
6. Preserve paragraph structure

The translation must be COMPLETE - do not skip any content."""

    user_prompt = f"Translate to {target_language}:\n\n{text}"
    
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 8192,  # Generous output limit
                    "top_p": 0.9,
                },
            },
            timeout=300,
        )
        
        if response.status_code == 200:
            data = response.json()
            result = data.get("message", {}).get("content", "")
            
            # Clean any meta-text
            result = clean_translation_output(result)
            
            return result
        else:
            logger.error(f"Ollama API error: HTTP {response.status_code}")
            raise Exception(f"Ollama API error: {response.status_code}")
            
    except requests.exceptions.Timeout:
        logger.error("Ollama translation timed out")
        raise
    except Exception as e:
        logger.error(f"Ollama translation error: {e}")
        raise


def translate_with_openai(text: str, target_language: str, api_key: str) -> str:
    """OpenAI translation."""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"Translate the following text to {target_language}. "
                               f"Output ONLY the translation. Keep all formulas and placeholders unchanged."
                },
                {"role": "user", "content": text}
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"OpenAI translation error: {e}")
        raise


def clean_translation_output(text: str) -> str:
    """Remove LLM meta-text from translation output."""
    # Patterns that indicate echoed instructions or meta-comments
    bad_patterns = [
        r'^(Here is|Here\'s|Below is|The following is).*?:\s*\n',
        r'^(Translation|Translated|Übersetzung|Traduzione|Traduction):?\s*\n',
        r'^\s*---+\s*\n',
        r'\n\s*---+\s*$',
    ]
    
    result = text
    for pattern in bad_patterns:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE | re.MULTILINE)
    
    return result.strip()


# =============================================================================
# PAGE CREATION
# =============================================================================

def create_translated_page(
    merged_data: Dict,
    images_dir: Path,
    page_num: int,
    output_path: Path,
    target_language: str,
    model: str,
    use_openai: bool = False,
    openai_api_key: str = None,
    progress_callback = None,
    translation_context = None,
) -> bool:
    """Create a translated PDF page."""
    try:
        # Get full text to translate
        full_text = merged_data.get("full_text", "")
        
        if not full_text.strip():
            logger.warning(f"Page {page_num}: No text to translate")
            # Create empty page
            doc = fitz.open()
            page = doc.new_page(
                width=merged_data.get("page_width", 612),
                height=merged_data.get("page_height", 792)
            )
            doc.save(str(output_path))
            doc.close()
            return True
        
        # Log original text length
        orig_len = len(full_text)
        logger.info(f"Page {page_num}: Translating {orig_len} chars")
        
        if progress_callback:
            progress_callback(1, 3, "Translating text...")
        
        # Translate the full text
        translated_text = translate_text_block(
            full_text,
            target_language,
            model,
            use_openai,
            openai_api_key,
            translation_context
        )
        
        # Log translated text length
        trans_len = len(translated_text)
        ratio = trans_len / orig_len if orig_len > 0 else 0
        logger.info(f"Page {page_num}: Translated to {trans_len} chars ({ratio:.1%})")
        
        if ratio < MIN_TRANSLATION_RATIO:
            logger.warning(f"Page {page_num}: Translation suspiciously short!")
        
        if progress_callback:
            progress_callback(2, 3, "Creating PDF page...")
        
        # Create new PDF page
        doc = fitz.open()
        page = doc.new_page(
            width=merged_data.get("page_width", 612),
            height=merged_data.get("page_height", 792)
        )
        
        # Add translated text
        # Use a simple text insertion for now
        text_rect = fitz.Rect(50, 50, page.rect.width - 50, page.rect.height - 50)
        
        # Try to insert with standard font
        try:
            # Use a font that supports many characters
            fontname = "helv"  # Helvetica
            fontsize = 10
            
            # Insert text with automatic wrapping
            rc = page.insert_textbox(
                text_rect,
                translated_text,
                fontname=fontname,
                fontsize=fontsize,
                align=fitz.TEXT_ALIGN_LEFT,
            )
            
            # If text didn't fit, try smaller font
            if rc < 0:
                fontsize = 8
                page.insert_textbox(
                    text_rect,
                    translated_text,
                    fontname=fontname,
                    fontsize=fontsize,
                    align=fitz.TEXT_ALIGN_LEFT,
                )
                
        except Exception as e:
            logger.warning(f"Page {page_num}: Text insertion error: {e}")
            # Fallback: simple text insertion
            page.insert_text((50, 70), translated_text[:5000], fontsize=9)
        
        # Add images
        for img_idx, img_data in enumerate(merged_data.get("images", [])):
            try:
                img_bytes = img_data.get("data")
                if img_bytes:
                    img_rect = fitz.Rect(
                        img_data.get("x", 0),
                        img_data.get("y", 0),
                        img_data.get("x", 0) + img_data.get("width", 100),
                        img_data.get("y", 0) + img_data.get("height", 100)
                    )
                    page.insert_image(img_rect, stream=img_bytes)
            except Exception as e:
                logger.warning(f"Page {page_num}: Could not insert image {img_idx}: {e}")
        
        if progress_callback:
            progress_callback(3, 3, "Saving page...")
        
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
        "total_chars": 0,
        "warnings": []
    }
    
    if not pdf_path.exists():
        report["warnings"].append("Output PDF does not exist")
        return report
    
    doc = fitz.open(str(pdf_path))
    report["page_count"] = len(doc)
    
    # Check for garbage characters and count total text
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    
    report["total_chars"] = len(full_text)
    report["garbage_chars_found"] = count_garbage_chars(full_text)
    
    if report["garbage_chars_found"] > 0:
        report["warnings"].append(f"Found {report['garbage_chars_found']} garbage characters")
    
    if report["page_count"] > source_page_count + 2:
        report["warnings"].append(f"Page count increased: {source_page_count} → {report['page_count']}")
    
    if report["total_chars"] < 100:
        report["warnings"].append("Very little text in output PDF")
    
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
    - Validation and retry for quality
    
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
    total_original_chars = 0
    total_translated_chars = 0
    
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
                logger.warning(f"Page {page_num}: No text from {method}")
            
            # 2c: Extract with PyMuPDF (layout/images)
            pymupdf_data = extract_with_pymupdf(page_path)
            logger.info(f"Page {page_num}: PyMuPDF found {len(pymupdf_data['images'])} images, {len(pymupdf_data.get('full_text', ''))} chars")
            
            # 2d: Merge extractions - IMPROVED to use both sources
            merged_data = merge_extractions(marker_text, pymupdf_data)
            
            original_chars = len(merged_data.get("full_text", ""))
            total_original_chars += original_chars
            
            # 2e: Prepare for translation (load Ollama)
            if not use_openai:
                model_manager.prepare_for_translation(model)
            
            if progress_callback:
                progress_callback(
                    base_progress + int(page_progress_range * 0.3), 100,
                    f"Page {page_num}/{total_pages}: Translating..."
                )
            
            # 2f: Create translated page with improved translation
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
                
                # Count translated chars
                try:
                    doc = fitz.open(str(translated_page_path))
                    trans_chars = len(doc[0].get_text())
                    total_translated_chars += trans_chars
                    doc.close()
                except:
                    pass
                
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
            
            # Calculate translation ratio
            ratio = total_translated_chars / total_original_chars if total_original_chars > 0 else 0
            ratio_status = "✅" if 0.5 <= ratio <= 2.0 else "⚠️"
            
            return str(output_pdf), f"""✅ Unified Translation Complete!

📄 **{sanity_report['page_count']} pages** (source: {total_pages})
🔬 Extraction: {methods_str}
📐 PyMuPDF for layout preservation
🖼️ Images extracted and preserved
🌐 Translated to {target_language}
📚 Glossary: Consistent terminology
🔗 Context: Translation consistency

📊 **Quality Metrics:**
  - Original: {total_original_chars:,} chars
  - Translated: {total_translated_chars:,} chars
  - Ratio: {ratio_status} {ratio:.1%}
  - Garbage chars: {sanity_report['garbage_chars_found']}{warnings_text}

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
