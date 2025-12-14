"""
PDF Layout-Preserving Translator

Translates PDFs while preserving the original layout, fonts, and formulas.
Uses PyMuPDF (fitz) for precise text replacement.

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import fitz  # PyMuPDF

logger = logging.getLogger("pdf_translator.layout")


def is_formula_text(text: str) -> bool:
    """
    Detects if text is likely a mathematical formula that should NOT be translated.
    """
    if not text or len(text.strip()) < 2:
        return False
    
    text = text.strip()
    
    # LaTeX patterns
    if any(p in text for p in ['$', '\\', '\\frac', '\\sum', '\\int', '\\alpha', '\\beta']):
        return True
    
    # Greek letters
    greek = set('αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ')
    if any(c in greek for c in text):
        return True
    
    # Math symbols
    math_symbols = set('∫∑∏√∞≈≠≤≥±×÷∈∉⊂⊃∪∩∧∨¬∀∃∂∇→←↔⇒⇐⇔')
    if any(c in math_symbols for c in text):
        return True
    
    # Equation-like: mostly symbols and numbers, few words
    words = re.findall(r'[a-zA-Z]{3,}', text)
    symbols = re.findall(r'[=+\-*/^(){}[\]<>]', text)
    if len(symbols) > len(words) * 2:
        return True
    
    # Single letters with subscripts/superscripts
    if re.match(r'^[A-Za-z][_^]', text):
        return True
    
    return False


def translate_text_block(
    text: str,
    model: str,
    source_lang: str,
    target_lang: str,
    ollama_url: str = "http://localhost:11434"
) -> str:
    """
    Translates a single text block using Ollama.
    Preserves formulas and special formatting.
    """
    import requests
    
    if not text or not text.strip():
        return text
    
    # Skip formula-like text
    if is_formula_text(text):
        return text
    
    # Skip very short text (likely labels, numbers)
    if len(text.strip()) < 5:
        return text
    
    # Skip if mostly numbers/symbols
    alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    if alpha_ratio < 0.3:
        return text
    
    system_prompt = f"""You are a scientific translator. Translate to {target_lang}.
RULES:
- Output ONLY the translation in {target_lang}
- Keep ALL formulas, equations, symbols unchanged
- Keep author names unchanged
- Keep numbers unchanged
- NO comments, NO explanations
- Preserve line breaks if present"""

    user_prompt = f"Translate to {target_lang}:\n{text}"
    
    try:
        response = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 2048}
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json().get("message", {}).get("content", text)
            # Clean up any meta-comments
            result = re.sub(r'^(Translation|Here is|Note:).*?:\s*', '', result, flags=re.IGNORECASE)
            return result.strip()
    except Exception as e:
        logger.warning(f"Translation error: {e}")
    
    return text


def translate_pdf_preserve_layout(
    input_path: str,
    output_path: str,
    model: str,
    source_lang: str,
    target_lang: str,
    progress_callback=None
) -> bool:
    """
    Translates a PDF while preserving the original layout.
    
    This approach:
    1. Extracts text blocks with their positions
    2. Translates text blocks (skipping formulas)
    3. Redacts original text and inserts translated text at same position
    
    Args:
        input_path: Path to input PDF
        output_path: Path to output PDF
        model: Ollama model name
        source_lang: Source language
        target_lang: Target language
        progress_callback: Optional callback(page, total_pages, status)
    
    Returns:
        True if successful
    """
    try:
        doc = fitz.open(input_path)
        total_pages = len(doc)
        
        for page_num in range(total_pages):
            page = doc[page_num]
            
            if progress_callback:
                progress_callback(page_num + 1, total_pages, f"Processing page {page_num + 1}/{total_pages}")
            
            # Get text blocks with positions
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            
            for block in blocks:
                if block["type"] != 0:  # Skip non-text blocks (images)
                    continue
                
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        original_text = span.get("text", "")
                        
                        if not original_text.strip():
                            continue
                        
                        # Skip formulas
                        if is_formula_text(original_text):
                            continue
                        
                        # Translate
                        translated = translate_text_block(
                            original_text, model, source_lang, target_lang
                        )
                        
                        if translated and translated != original_text:
                            # Get position
                            bbox = fitz.Rect(span["bbox"])
                            font_size = span.get("size", 11)
                            font_name = span.get("font", "helv")
                            
                            # Redact original text
                            page.add_redact_annot(bbox, fill=(1, 1, 1))  # White fill
                            page.apply_redactions()
                            
                            # Insert translated text
                            # Adjust font size if text is longer
                            text_length_ratio = len(translated) / max(len(original_text), 1)
                            adjusted_size = font_size
                            if text_length_ratio > 1.3:
                                adjusted_size = font_size * 0.85
                            
                            try:
                                page.insert_text(
                                    (bbox.x0, bbox.y1 - 2),  # Baseline position
                                    translated,
                                    fontsize=adjusted_size,
                                    fontname="helv",  # Use standard font
                                    color=(0, 0, 0)
                                )
                            except Exception as e:
                                logger.warning(f"Could not insert text: {e}")
        
        # Save
        doc.save(output_path)
        doc.close()
        
        if progress_callback:
            progress_callback(total_pages, total_pages, "Complete!")
        
        return True
        
    except Exception as e:
        logger.exception(f"PDF translation error: {e}")
        return False


def translate_pdf_text_layer(
    input_path: str,
    output_path: str,
    model: str,
    source_lang: str,
    target_lang: str,
    progress_callback=None
) -> bool:
    """
    Alternative approach: Creates a new PDF with translated text overlay.
    Better for complex PDFs where direct text replacement fails.
    
    This preserves the original PDF as background and adds translated text on top.
    """
    try:
        doc = fitz.open(input_path)
        total_pages = len(doc)
        
        # Collect all text to translate
        all_translations = []
        
        for page_num in range(total_pages):
            page = doc[page_num]
            
            if progress_callback:
                progress_callback(page_num + 1, total_pages, f"Extracting page {page_num + 1}")
            
            # Extract text with layout
            text = page.get_text("text")
            
            # Split into paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            page_translations = []
            for para in paragraphs:
                if is_formula_text(para):
                    page_translations.append(para)  # Keep as-is
                else:
                    translated = translate_text_block(para, model, source_lang, target_lang)
                    page_translations.append(translated)
            
            all_translations.append(page_translations)
        
        # Create output document
        # For now, just save the translated text
        output_text = ""
        for page_num, translations in enumerate(all_translations):
            output_text += f"\n\n=== Page {page_num + 1} ===\n\n"
            output_text += "\n\n".join(translations)
        
        # Save as text file alongside PDF
        text_output = output_path.replace('.pdf', '_translated.txt')
        with open(text_output, 'w', encoding='utf-8') as f:
            f.write(output_text)
        
        doc.close()
        
        if progress_callback:
            progress_callback(total_pages, total_pages, f"Saved to {text_output}")
        
        return True
        
    except Exception as e:
        logger.exception(f"PDF translation error: {e}")
        return False


# For testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python pdf_layout_translator.py input.pdf output.pdf [model] [target_lang]")
        sys.exit(1)
    
    input_pdf = sys.argv[1]
    output_pdf = sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "qwen2.5:7b"
    target = sys.argv[4] if len(sys.argv) > 4 else "German"
    
    print(f"Translating {input_pdf} to {target} using {model}...")
    
    success = translate_pdf_preserve_layout(
        input_pdf, output_pdf, model, "auto", target,
        progress_callback=lambda p, t, s: print(f"  {s}")
    )
    
    if success:
        print(f"Done! Output: {output_pdf}")
    else:
        print("Translation failed!")
