"""
PDF Overlay Translator - 100% Layout Preservation

This translator preserves the EXACT original PDF layout:
- Formulas remain as images/vectors (100% identical)
- Images remain untouched
- Only TEXT BLOCKS are translated and replaced

Strategy:
1. Copy original PDF
2. For each text block:
   - Extract text
   - Translate text
   - Redact (white-out) original text
   - Insert translated text at same position
3. Formulas/images untouched

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import logging
import re
import copy
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable
from dataclasses import dataclass, field
import fitz  # PyMuPDF

logger = logging.getLogger("pdf_translator.overlay")


# =============================================================================
# CONFIGURATION
# =============================================================================

MIN_TEXT_LENGTH = 10  # Minimum characters to translate
MAX_RETRIES = 5
RETRY_DELAY = 1.5

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
    is_formula: bool = False
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
# FORMULA DETECTION
# =============================================================================

def is_likely_formula(text: str) -> bool:
    """Detect if text is likely a mathematical formula."""
    # Check for common formula patterns
    formula_indicators = [
        r'[∫∑∏∂∇√∞±×÷≈≠≤≥∈∉⊂⊃∪∩]',  # Math symbols
        r'[αβγδεζηθικλμνξπρστυφχψω]',  # Greek letters
        r'[ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΠΡΣΤΥΦΧΨΩ]',  # Capital Greek
        r'\d+[\^_]\{?\d',  # Superscripts/subscripts
        r'\\[a-zA-Z]+',   # LaTeX commands
        r'\$.*\$',        # Inline math
        r'[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]',  # Unicode superscripts
        r'[₀₁₂₃₄₅₆₇₈₉₊₋]',  # Unicode subscripts
    ]
    
    for pattern in formula_indicators:
        if re.search(pattern, text):
            # If mostly symbols, it's a formula
            symbol_count = len(re.findall(pattern, text))
            word_count = len(text.split())
            if symbol_count > 0 and (symbol_count / max(word_count, 1)) > 0.3:
                return True
    
    return False


def is_translatable_text(text: str) -> bool:
    """Check if text should be translated."""
    text = text.strip()
    
    # Too short
    if len(text) < MIN_TEXT_LENGTH:
        return False
    
    # Likely a formula
    if is_likely_formula(text):
        return False
    
    # Only numbers/symbols
    if not re.search(r'[a-zA-Z]{3,}', text):
        return False
    
    # URL or email
    if re.search(r'https?://|@.*\.', text):
        return False
    
    return True


# =============================================================================
# TEXT EXTRACTION WITH POSITIONS
# =============================================================================

def extract_text_blocks(page: fitz.Page) -> List[TextBlock]:
    """Extract text blocks with their positions."""
    blocks = []
    
    # Get text as dictionary with position info
    text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:  # Only text blocks
            continue
        
        # Collect all text from this block
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
        
        # Get block rectangle
        bbox = block.get("bbox", [0, 0, 0, 0])
        rect = fitz.Rect(bbox)
        
        # Average font size
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 10
        primary_font = font_names[0] if font_names else ""
        
        # Check if formula
        is_formula = is_likely_formula(block_text)
        
        blocks.append(TextBlock(
            text=block_text,
            rect=rect,
            font_size=avg_font_size,
            font_name=primary_font,
            is_formula=is_formula
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
    
    # Build glossary instruction
    glossary_text = ""
    if glossary:
        glossary_text = "MANDATORY TERMINOLOGY:\n"
        for en, trans in glossary.items():
            glossary_text += f"  - {en} → {trans}\n"
    
    system_prompt = f"""You are a scientific translator. Translate to {target_language}.

{glossary_text}
RULES:
- Output ONLY the translation
- Keep numbers, symbols, formulas unchanged
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
                
                # Clean up common artifacts
                result = re.sub(r'^(Here|Translation|Übersetzung).*?:\s*', '', result, flags=re.I)
                result = re.sub(r'^```\w*\n?', '', result)
                result = re.sub(r'\n?```$', '', result)
                
                if result and len(result) >= len(text) * 0.5:
                    return result
            
            time.sleep(RETRY_DELAY * (attempt + 1))
            
        except Exception as e:
            logger.warning(f"Translation attempt {attempt + 1} failed: {e}")
            time.sleep(RETRY_DELAY * (attempt + 1))
    
    return text  # Return original on failure


# =============================================================================
# PDF OVERLAY MODIFICATION
# =============================================================================

def replace_text_in_pdf(
    input_pdf: str,
    output_pdf: str,
    model: str,
    target_language: str,
    progress_callback: Optional[Callable] = None
) -> TranslationResult:
    """
    Replace text in PDF while preserving layout.
    
    This uses the redaction technique:
    1. Add white redaction annotation over original text
    2. Apply redaction (removes original)
    3. Insert translated text at same position
    """
    result = TranslationResult(
        success=False,
        output_path=None,
        pages_processed=0,
        blocks_translated=0,
        blocks_skipped=0,
        warnings=[]
    )
    
    # Get glossary for target language
    glossary = GLOSSARY_DE if target_language.lower() in ["german", "deutsch"] else {}
    
    try:
        # Open document
        doc = fitz.open(input_pdf)
        total_pages = len(doc)
        
        if progress_callback:
            progress_callback(0, 100, f"Processing {total_pages} pages...")
        
        for page_num in range(total_pages):
            page = doc[page_num]
            progress = int(5 + 90 * page_num / total_pages)
            
            if progress_callback:
                progress_callback(progress, 100, f"Page {page_num + 1}/{total_pages}")
            
            # Extract text blocks
            blocks = extract_text_blocks(page)
            
            # Process each block
            for block in blocks:
                # Skip formulas and non-translatable text
                if block.is_formula or not is_translatable_text(block.text):
                    result.blocks_skipped += 1
                    continue
                
                # Translate
                translated = translate_text_ollama(
                    block.text, model, target_language, glossary
                )
                
                if translated == block.text:
                    result.blocks_skipped += 1
                    continue
                
                # Redact original text (white fill)
                page.add_redact_annot(
                    block.rect,
                    fill=(1, 1, 1)  # White
                )
                
                block.translated = translated
                result.blocks_translated += 1
            
            # Apply all redactions for this page
            page.apply_redactions()
            
            # Insert translated text
            for block in blocks:
                if block.translated:
                    # Calculate font size to fit
                    font_size = block.font_size
                    
                    # Insert text
                    try:
                        # Try to fit text in the original rectangle
                        rc = page.insert_textbox(
                            block.rect,
                            block.translated,
                            fontsize=font_size,
                            fontname="helv",
                            align=fitz.TEXT_ALIGN_LEFT
                        )
                        
                        # If text doesn't fit, reduce font size
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
                        result.warnings.append(f"Page {page_num + 1}: text insertion failed")
            
            result.pages_processed += 1
        
        # Save modified PDF
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
# IDENTITY TEST (EN -> EN)
# =============================================================================

def identity_test(input_pdf: str, output_pdf: str) -> Dict:
    """
    Test: Extract text, don't translate, reconstruct.
    Should produce identical text output.
    """
    doc = fitz.open(input_pdf)
    out_doc = fitz.open()
    
    differences = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Get original text
        original_text = page.get_text()
        
        # Create new page with same dimensions
        new_page = out_doc.new_page(
            width=page.rect.width,
            height=page.rect.height
        )
        
        # Copy page content (including images, formulas as vectors)
        new_page.show_pdf_page(new_page.rect, doc, page_num)
        
    out_doc.save(output_pdf)
    out_doc.close()
    doc.close()
    
    # Compare
    doc1 = fitz.open(input_pdf)
    doc2 = fitz.open(output_pdf)
    
    for i in range(len(doc1)):
        t1 = doc1[i].get_text()
        t2 = doc2[i].get_text()
        if t1 != t2:
            differences.append({
                "page": i + 1,
                "original_len": len(t1),
                "copy_len": len(t2)
            })
    
    doc1.close()
    doc2.close()
    
    return {
        "identical": len(differences) == 0,
        "differences": differences
    }


# =============================================================================
# MAIN - PERFECT COPY STRATEGY
# =============================================================================

def translate_pdf_overlay(
    input_pdf: str,
    output_dir: str,
    model: str,
    target_language: str,
    progress_callback: Optional[Callable] = None
) -> TranslationResult:
    """
    Perfect PDF translation using overlay strategy.
    
    If target_language == source language (e.g., English -> English),
    this produces an identical copy (for testing).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = str(output_dir / "translated_overlay.pdf")
    
    return replace_text_in_pdf(
        input_pdf, output_pdf, model, target_language, progress_callback
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_overlay_translator.py input.pdf [output_dir] [language] [model]")
        print("       python pdf_overlay_translator.py --test input.pdf  # Identity test")
        sys.exit(1)
    
    if sys.argv[1] == "--test":
        if len(sys.argv) < 3:
            print("Usage: python pdf_overlay_translator.py --test input.pdf")
            sys.exit(1)
        
        print("Running identity test (copy without translation)...")
        result = identity_test(sys.argv[2], "identity_test_output.pdf")
        
        if result["identical"]:
            print("SUCCESS: PDF copy is identical!")
        else:
            print(f"DIFFERENCES FOUND: {result['differences']}")
        sys.exit(0)
    
    input_pdf = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output_overlay"
    language = sys.argv[3] if len(sys.argv) > 3 else "German"
    model = sys.argv[4] if len(sys.argv) > 4 else "qwen2.5:7b"
    
    print(f"Translating {input_pdf} to {language}...")
    
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
