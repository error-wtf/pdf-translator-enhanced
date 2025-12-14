"""
PDF Overlay Translator - 100% Formula Preservation

CRITICAL: Formulas must remain 100% identical!
Strategy:
1. Copy original page structure
2. Identify text blocks
3. Skip ANY block containing Unicode math symbols
4. Only translate pure ASCII/Latin text blocks
5. Formulas stay as original vectors/images

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable
from dataclasses import dataclass, field
import fitz  # PyMuPDF

logger = logging.getLogger("pdf_translator.overlay")


# =============================================================================
# CONFIGURATION
# =============================================================================

MIN_TEXT_LENGTH = 20  # Minimum characters to translate
MAX_RETRIES = 5
RETRY_DELAY = 1.5

# Unicode ranges that indicate formulas - NEVER translate these blocks
FORMULA_UNICODE_RANGES = [
    (0x0370, 0x03FF),   # Greek letters
    (0x2070, 0x209F),   # Superscripts/subscripts
    (0x2100, 0x214F),   # Letterlike symbols
    (0x2150, 0x218F),   # Number forms
    (0x2190, 0x21FF),   # Arrows
    (0x2200, 0x22FF),   # Mathematical operators
    (0x2300, 0x23FF),   # Misc technical
    (0x27C0, 0x27EF),   # Misc math symbols A
    (0x2980, 0x29FF),   # Misc math symbols B
    (0x2A00, 0x2AFF),   # Supplemental math operators
    (0x1D400, 0x1D7FF), # Math alphanumeric symbols
]

# Individual formula characters
FORMULA_CHARS = set('∫∑∏∂∇√∞±×÷≈≠≤≥∈∉⊂⊃∪∩∧∨¬∀∃∅∆∇αβγδεζηθικλμνξπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΠΡΣΤΥΦΧΨΩ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎')

# Scientific glossary
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
    has_formula: bool = False
    translated: str = ""


@dataclass
class TranslationResult:
    """Result of overlay translation."""
    success: bool
    output_path: Optional[str]
    pages_processed: int
    blocks_translated: int
    blocks_skipped_formula: int
    blocks_skipped_other: int
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# FORMULA DETECTION - STRICT
# =============================================================================

def contains_formula_unicode(text: str) -> bool:
    """
    Check if text contains ANY Unicode character that indicates a formula.
    If yes, the ENTIRE block is skipped to preserve formulas 100%.
    """
    for char in text:
        # Check individual formula characters
        if char in FORMULA_CHARS:
            return True
        
        # Check Unicode ranges
        code = ord(char)
        for start, end in FORMULA_UNICODE_RANGES:
            if start <= code <= end:
                return True
    
    return False


def contains_latex_patterns(text: str) -> bool:
    """Check for LaTeX-style patterns."""
    patterns = [
        r'\$.*?\$',           # Inline math
        r'\\[a-zA-Z]+',       # LaTeX commands
        r'\^{',               # Superscript
        r'_{',                # Subscript
        r'\\frac',            # Fractions
        r'\\sqrt',            # Square root
    ]
    
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    
    return False


def is_pure_text_block(text: str) -> bool:
    """
    Check if block contains ONLY translatable text.
    Returns False if ANY formula indicator is found.
    """
    text = text.strip()
    
    # Too short
    if len(text) < MIN_TEXT_LENGTH:
        return False
    
    # Contains formula Unicode
    if contains_formula_unicode(text):
        return False
    
    # Contains LaTeX patterns
    if contains_latex_patterns(text):
        return False
    
    # Only numbers/symbols (no words)
    if not re.search(r'[a-zA-Z]{4,}', text):
        return False
    
    # URL or email
    if re.search(r'https?://|@.*\.[a-z]', text):
        return False
    
    # Em-dash (often near formulas)
    if '—' in text or '–' in text:
        return False
    
    return True


# =============================================================================
# TEXT EXTRACTION WITH POSITIONS
# =============================================================================

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
        
        # Check for formula content
        has_formula = contains_formula_unicode(block_text) or contains_latex_patterns(block_text)
        
        blocks.append(TextBlock(
            text=block_text,
            rect=rect,
            font_size=avg_font_size,
            font_name=primary_font,
            has_formula=has_formula
        ))
    
    return blocks


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
    
    glossary_text = ""
    if glossary:
        glossary_text = "MANDATORY TERMINOLOGY:\n"
        for en, trans in glossary.items():
            glossary_text += f"  - {en} → {trans}\n"
    
    system_prompt = f"""You are a scientific translator. Translate to {target_language}.

{glossary_text}
RULES:
- Output ONLY the translation
- Keep numbers unchanged
- Keep author names unchanged
- Use correct scientific terminology"""

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Translate:\n{text}"}
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
                
                if result and len(result) >= len(text) * 0.5:
                    return result
            
            time.sleep(RETRY_DELAY * (attempt + 1))
            
        except Exception as e:
            logger.warning(f"Translation attempt {attempt + 1} failed: {e}")
            time.sleep(RETRY_DELAY * (attempt + 1))
    
    return text


# =============================================================================
# PDF TRANSLATION - FORMULA-SAFE
# =============================================================================

def translate_pdf_overlay(
    input_pdf: str,
    output_dir: str,
    model: str,
    target_language: str,
    progress_callback: Optional[Callable] = None
) -> TranslationResult:
    """
    Translate PDF while preserving formulas 100%.
    
    Any block containing Unicode math symbols is left UNTOUCHED.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = str(output_dir / "translated_overlay.pdf")
    
    result = TranslationResult(
        success=False,
        output_path=None,
        pages_processed=0,
        blocks_translated=0,
        blocks_skipped_formula=0,
        blocks_skipped_other=0,
        warnings=[]
    )
    
    glossary = GLOSSARY_DE if target_language.lower() in ["german", "deutsch"] else {}
    
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
            
            blocks = extract_text_blocks(page)
            
            for block in blocks:
                # NEVER touch formula blocks
                if block.has_formula:
                    result.blocks_skipped_formula += 1
                    logger.debug(f"Skipped formula block: {block.text[:50]}...")
                    continue
                
                # Skip non-pure text blocks
                if not is_pure_text_block(block.text):
                    result.blocks_skipped_other += 1
                    continue
                
                # Translate
                translated = translate_text_ollama(
                    block.text, model, target_language, glossary
                )
                
                if translated == block.text:
                    result.blocks_skipped_other += 1
                    continue
                
                # Redact and replace
                page.add_redact_annot(block.rect, fill=(1, 1, 1))
                block.translated = translated
                result.blocks_translated += 1
            
            page.apply_redactions()
            
            # Insert translated text
            for block in blocks:
                if block.translated:
                    try:
                        font_size = block.font_size
                        rc = page.insert_textbox(
                            block.rect,
                            block.translated,
                            fontsize=font_size,
                            fontname="helv",
                            align=fitz.TEXT_ALIGN_LEFT
                        )
                        
                        if rc < 0:
                            font_size = max(6, font_size * 0.8)
                            page.insert_textbox(
                                block.rect,
                                block.translated,
                                fontsize=font_size,
                                fontname="helv",
                                align=fitz.TEXT_ALIGN_LEFT
                            )
                    except Exception as e:
                        logger.warning(f"Text insertion failed: {e}")
            
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


# =============================================================================
# IDENTITY TEST
# =============================================================================

def identity_test(input_pdf: str, output_pdf: str) -> Dict:
    """Test: Copy PDF without modification. Should be identical."""
    doc = fitz.open(input_pdf)
    out_doc = fitz.open()
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        new_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
        new_page.show_pdf_page(new_page.rect, doc, page_num)
    
    out_doc.save(output_pdf)
    out_doc.close()
    doc.close()
    
    # Compare
    doc1 = fitz.open(input_pdf)
    doc2 = fitz.open(output_pdf)
    
    differences = []
    for i in range(len(doc1)):
        t1 = doc1[i].get_text()
        t2 = doc2[i].get_text()
        if t1 != t2:
            differences.append({"page": i + 1, "diff": len(t1) - len(t2)})
    
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
    print("NOTE: Blocks with formula symbols will be preserved unchanged.")
    
    result = translate_pdf_overlay(
        input_pdf, output_dir, model, language,
        progress_callback=lambda c, t, m: print(f"[{c:3d}%] {m}")
    )
    
    print(f"\n{'='*60}")
    print(f"SUCCESS: {result.success}")
    print(f"Pages: {result.pages_processed}")
    print(f"Blocks translated: {result.blocks_translated}")
    print(f"Blocks skipped (formulas): {result.blocks_skipped_formula}")
    print(f"Blocks skipped (other): {result.blocks_skipped_other}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")
    if result.output_path:
        print(f"\nOutput: {result.output_path}")
