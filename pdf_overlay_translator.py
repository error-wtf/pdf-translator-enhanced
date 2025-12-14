"""
PDF Overlay Translator - 100% Formula Preservation

Uses Cambria font which supports Unicode math symbols.
This enables translating text while preserving formulas like ΔΦ, ω, 10⁻¹⁶.

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import logging
import re
import time
import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable
from dataclasses import dataclass, field
import fitz  # PyMuPDF

# Import formula protection and validation
from formula_isolator import (
    extract_and_protect,
    normalize_output,
    audit_utf8,
    assert_no_corruption,
    regression_check,
)

logger = logging.getLogger("pdf_translator.overlay")


# =============================================================================
# CONFIGURATION
# =============================================================================

MIN_TEXT_LENGTH = 3  # Reduced from 15 to translate short titles
MAX_RETRIES = 5
RETRY_DELAY = 1.5

# Font that supports Unicode math symbols
UNICODE_FONT_PATHS = [
    "C:/Windows/Fonts/cambria.ttc",      # Windows - best for math
    "C:/Windows/Fonts/DejaVuSans.ttf",   # Windows fallback
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    "/System/Library/Fonts/STIXGeneral.otf",  # macOS
]

# Scientific glossary for German
GLOSSARY_DE = {
    "entanglement": "Verschränkung",
    "entangled": "verschränkt",
    "coherence": "Kohärenz",
    "decoherence": "Dekohärenz",
    "superposition": "Superposition",
    "qubit": "Qubit",
    "fidelity": "Fidelität",
    "spacetime": "Raumzeit",
    "phase": "Phase",
    "quantum": "Quanten",
    "abstract": "Zusammenfassung",
    "introduction": "Einleitung",
    "methods": "Methoden",
    "results": "Ergebnisse",
    "conclusions": "Schlussfolgerungen",
    "references": "Literaturverzeichnis",
    "acknowledgments": "Danksagung",
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TextBlock:
    """A text block with position and content."""
    text: str
    rect: fitz.Rect
    font_size: float
    font_name: str
    translated: str = ""


@dataclass
class TranslationResult:
    """Result of overlay translation."""
    success: bool
    output_path: Optional[str]
    pages_processed: int
    blocks_translated: int
    blocks_skipped: int
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# FONT HANDLING
# =============================================================================

def get_unicode_font() -> Optional[fitz.Font]:
    """Get a font that supports Unicode math symbols."""
    for font_path in UNICODE_FONT_PATHS:
        if os.path.exists(font_path):
            try:
                font = fitz.Font(fontfile=font_path)
                logger.info(f"Using Unicode font: {font.name}")
                return font
            except Exception as e:
                logger.warning(f"Could not load {font_path}: {e}")
    
    logger.warning("No Unicode font found, using default (symbols may be lost)")
    return None


# =============================================================================
# TEXT BLOCK HANDLING
# =============================================================================

def is_translatable_text(text: str) -> bool:
    """Check if text should be translated."""
    text = text.strip()
    
    if len(text) < MIN_TEXT_LENGTH:
        return False
    
    # Must have at least one letter
    if not re.search(r'[a-zA-Z]', text):
        return False
    
    # Skip URLs/emails
    if re.search(r'https?://|@.*\.[a-z]', text):
        return False
    
    # Skip pure numbers/dates
    if re.match(r'^[\d\s\-/\.,:]+$', text):
        return False
    
    return True


def extract_text_blocks(page: fitz.Page) -> List[TextBlock]:
    """Extract text blocks with their positions."""
    blocks = []
    
    text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        
        block_text = ""
        font_sizes = []
        font_names = []
        
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                span_text = span.get("text", "")
                block_text += span_text
                font_sizes.append(span.get("size", 10))
                font_names.append(span.get("font", ""))
            block_text += "\n"
        
        block_text = block_text.strip()
        if not block_text:
            continue
        
        bbox = block.get("bbox", [0, 0, 0, 0])
        rect = fitz.Rect(bbox)
        
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 10
        primary_font = font_names[0] if font_names else ""
        
        blocks.append(TextBlock(
            text=block_text,
            rect=rect,
            font_size=avg_font_size,
            font_name=primary_font
        ))
    
    return blocks


def merge_adjacent_blocks(blocks: List[TextBlock], tolerance: float = 5.0) -> List[TextBlock]:
    """Merge blocks that are on the same line (for split titles)."""
    if not blocks:
        return blocks
    
    # Sort by y position, then x position
    sorted_blocks = sorted(blocks, key=lambda b: (b.rect.y0, b.rect.x0))
    
    merged = []
    current = None
    
    for block in sorted_blocks:
        if current is None:
            current = block
            continue
        
        # Check if blocks are on same line (similar y position)
        same_line = abs(block.rect.y0 - current.rect.y0) < tolerance
        # Check if blocks are close horizontally
        close = block.rect.x0 - current.rect.x1 < 50
        # Check if similar font size
        similar_size = abs(block.font_size - current.font_size) < 2
        
        if same_line and close and similar_size:
            # Merge blocks
            new_rect = fitz.Rect(
                min(current.rect.x0, block.rect.x0),
                min(current.rect.y0, block.rect.y0),
                max(current.rect.x1, block.rect.x1),
                max(current.rect.y1, block.rect.y1)
            )
            current = TextBlock(
                text=current.text + " " + block.text,
                rect=new_rect,
                font_size=current.font_size,
                font_name=current.font_name
            )
        else:
            merged.append(current)
            current = block
    
    if current:
        merged.append(current)
    
    return merged


# =============================================================================
# TRANSLATION
# =============================================================================

def translate_text_ollama(
    text: str,
    model: str,
    target_language: str,
    glossary: Dict[str, str] = None
) -> str:
    """Translate text using Ollama."""
    import requests
    
    if not text.strip():
        return text
    
    # UTF-8 audit before translation
    utf8_warnings = audit_utf8(text, "input")
    if utf8_warnings:
        logger.warning(f"UTF-8 issues in input: {utf8_warnings}")
    
    # Quick glossary lookup for single words
    text_lower = text.strip().lower()
    if glossary and text_lower in glossary:
        return glossary[text_lower].title() if text[0].isupper() else glossary[text_lower]
    
    glossary_text = ""
    if glossary:
        glossary_text = "MANDATORY TERMINOLOGY:\n"
        for en, trans in glossary.items():
            glossary_text += f"  - {en} → {trans}\n"
    
    system_prompt = f"""You are a scientific translator. Translate to {target_language}.

{glossary_text}
CRITICAL RULES:
- Output ONLY the translation, nothing else
- Keep ALL ⟦...⟧ placeholders EXACTLY unchanged (these protect formulas)
- Keep ALL mathematical symbols unchanged: Δ, Φ, ω, ×, ⁻, ¹, ⁶, —, ħ, ∇, Ψ, etc.
- Keep numbers and units unchanged
- Keep author names unchanged
- Use correct scientific terminology from glossary
- Do NOT output HTML tags like <sup> or <sub>
- For short words like "Abstract", translate directly"""

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Translate to {target_language}:\n{text}"}
                    ],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 2048}
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json().get("message", {}).get("content", "").strip()
                result = re.sub(r'^(Here|Translation|Übersetzung).*?:\s*', '', result, flags=re.I)
                result = re.sub(r'^```\w*\n?', '', result)
                result = re.sub(r'\n?```$', '', result)
                result = result.strip('"\'')
                
                if result and len(result) >= len(text) * 0.3:
                    return result
            
            time.sleep(RETRY_DELAY * (attempt + 1))
            
        except Exception as e:
            logger.warning(f"Translation attempt {attempt + 1} failed: {e}")
            time.sleep(RETRY_DELAY * (attempt + 1))
    
    return text


# =============================================================================
# PDF TRANSLATION WITH UNICODE FONT
# =============================================================================

def translate_pdf_overlay(
    input_pdf: str,
    output_dir: str,
    model: str,
    target_language: str,
    progress_callback: Optional[Callable] = None
) -> TranslationResult:
    """
    Translate PDF using Unicode-capable font.
    Preserves mathematical symbols like ΔΦ, ω, 10⁻¹⁶.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = str(output_dir / "translated_overlay.pdf")
    
    result = TranslationResult(
        success=False,
        output_path=None,
        pages_processed=0,
        blocks_translated=0,
        blocks_skipped=0,
        warnings=[]
    )
    
    glossary = GLOSSARY_DE if target_language.lower() in ["german", "deutsch", "de"] else {}
    
    # Get Unicode font
    unicode_font = get_unicode_font()
    if unicode_font is None:
        result.warnings.append("No Unicode font found - symbols may be lost")
    
    try:
        doc = fitz.open(input_pdf)
        total_pages = len(doc)
        
        if progress_callback:
            progress_callback(0, 100, f"Processing {total_pages} pages...")
        
        for page_num in range(total_pages):
            page = doc[page_num]
            progress = int(5 + 90 * page_num / total_pages)
            
            if progress_callback:
                progress_callback(progress, 100, f"Page {page_num + 1}/{total_pages}")
            
            # Extract and merge blocks
            blocks = extract_text_blocks(page)
            blocks = merge_adjacent_blocks(blocks)
            
            blocks_to_replace = []
            
            for block in blocks:
                if not is_translatable_text(block.text):
                    result.blocks_skipped += 1
                    continue
                
                # === MATH PROTECTION: Extract and mask formulas ===
                protected_text, restore_func = extract_and_protect(block.text)
                
                # Translate (formulas are now placeholders)
                translated_protected = translate_text_ollama(
                    protected_text, model, target_language, glossary
                )
                
                # Restore formulas
                translated = restore_func(translated_protected)
                
                # Normalize output (remove HTML artifacts, ensure consistent format)
                translated = normalize_output(translated, mode="unicode")
                
                # Validate no corruption
                if not assert_no_corruption(translated):
                    result.warnings.append(f"Corruption detected in block: {block.text[:30]}...")
                    translated = block.text  # Fallback to original
                
                if translated == block.text:
                    result.blocks_skipped += 1
                    continue
                
                block.translated = translated
                blocks_to_replace.append(block)
                result.blocks_translated += 1
            
            # Redact original text
            for block in blocks_to_replace:
                page.add_redact_annot(block.rect, fill=(1, 1, 1))
            
            page.apply_redactions()
            
            # Insert translated text with Unicode font
            for block in blocks_to_replace:
                try:
                    if unicode_font:
                        # Use TextWriter for proper Unicode support
                        tw = fitz.TextWriter(page.rect)
                        
                        # Calculate position
                        x = block.rect.x0 + 2
                        y = block.rect.y0 + block.font_size
                        
                        # Wrap text to fit rectangle
                        max_width = block.rect.width - 4
                        lines = wrap_text(block.translated, unicode_font, block.font_size, max_width)
                        
                        for line in lines:
                            if y > block.rect.y1:
                                break
                            tw.append((x, y), line, font=unicode_font, fontsize=block.font_size)
                            y += block.font_size * 1.2
                        
                        tw.write_text(page)
                    else:
                        # Fallback to standard font
                        page.insert_textbox(
                            block.rect,
                            block.translated,
                            fontsize=block.font_size,
                            fontname="helv",
                            align=fitz.TEXT_ALIGN_LEFT
                        )
                except Exception as e:
                    logger.warning(f"Text insertion failed: {e}")
                    # Fallback
                    try:
                        page.insert_textbox(
                            block.rect,
                            block.translated,
                            fontsize=block.font_size * 0.9,
                            fontname="helv",
                            align=fitz.TEXT_ALIGN_LEFT
                        )
                    except:
                        pass
            
            result.pages_processed += 1
        
        if progress_callback:
            progress_callback(96, 100, "Saving PDF...")
        
        doc.save(output_pdf)
        doc.close()
        
        result.success = True
        result.output_path = output_pdf
        
        if progress_callback:
            progress_callback(100, 100, "Complete!")
        
        return result
        
    except Exception as e:
        logger.exception(f"PDF overlay translation failed: {e}")
        result.warnings.append(str(e))
        return result


def wrap_text(text: str, font: fitz.Font, fontsize: float, max_width: float) -> List[str]:
    """Wrap text to fit within max_width."""
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = f"{current_line} {word}".strip()
        width = font.text_length(test_line, fontsize=fontsize)
        
        if width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return lines


# =============================================================================
# IDENTITY TEST
# =============================================================================

def identity_test(input_pdf: str, output_pdf: str) -> Dict:
    """Test: Copy PDF without modification."""
    doc = fitz.open(input_pdf)
    out_doc = fitz.open()
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        new_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
        new_page.show_pdf_page(new_page.rect, doc, page_num)
    
    out_doc.save(output_pdf)
    out_doc.close()
    doc.close()
    
    doc1 = fitz.open(input_pdf)
    doc2 = fitz.open(output_pdf)
    
    differences = []
    for i in range(len(doc1)):
        t1 = doc1[i].get_text()
        t2 = doc2[i].get_text()
        if t1 != t2:
            differences.append({"page": i + 1})
    
    doc1.close()
    doc2.close()
    
    return {"identical": len(differences) == 0, "differences": differences}


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_overlay_translator.py input.pdf [output_dir] [language] [model]")
        print("       python pdf_overlay_translator.py --test input.pdf")
        sys.exit(1)
    
    if sys.argv[1] == "--test":
        if len(sys.argv) < 3:
            print("Usage: python pdf_overlay_translator.py --test input.pdf")
            sys.exit(1)
        
        print("Running identity test...")
        result = identity_test(sys.argv[2], "identity_test_output.pdf")
        
        if result["identical"]:
            print("SUCCESS: PDF copy is identical!")
        else:
            print(f"DIFFERENCES: {result['differences']}")
        sys.exit(0)
    
    input_pdf = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output_overlay"
    language = sys.argv[3] if len(sys.argv) > 3 else "German"
    model = sys.argv[4] if len(sys.argv) > 4 else "qwen2.5:7b"
    
    print(f"Translating {input_pdf} to {language}...")
    print("Using Cambria font for Unicode symbol support.")
    
    result = translate_pdf_overlay(
        input_pdf, output_dir, model, language,
        progress_callback=lambda c, t, m: print(f"[{c:3d}%] {m}")
    )
    
    print(f"\n{'='*60}")
    print(f"SUCCESS: {result.success}")
    print(f"Pages: {result.pages_processed}")
    print(f"Blocks translated: {result.blocks_translated}")
    print(f"Blocks skipped: {result.blocks_skipped}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")
    if result.output_path:
        print(f"\nOutput: {result.output_path}")
