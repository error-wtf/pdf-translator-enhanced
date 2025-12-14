"""
PDF Processing for PDF-Translator

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from models import Block

logger = logging.getLogger("pdf_translator.pdf_processing")


# Regex patterns to detect and protect mathematical content
MATH_PATTERNS = [
    # LaTeX inline math
    r'\$[^$]+\$',
    # LaTeX display math
    r'\\\[[^\]]+\\\]',
    r'\\\([^)]+\\\)',
    # Common math symbols that indicate formulas
    r'[α-ωΑ-Ω∫∑∏√∞≈≠≤≥±×÷∈∉⊂⊃∪∩∧∨¬∀∃∂∇]+',
    # Subscripts/superscripts patterns
    r'\w+[_^]\{[^}]+\}',
    r'\w+[_^]\w',
    # Fractions and common LaTeX commands
    r'\\frac\{[^}]+\}\{[^}]+\}',
    r'\\sqrt\{[^}]+\}',
    r'\\[a-zA-Z]+\{[^}]*\}',
    # Equation-like patterns (e.g., "E = mc²", "F = ma")
    r'[A-Z]\s*=\s*[a-zA-Z0-9\s\+\-\*\/\^]+',
]

# Compiled pattern for formula detection
FORMULA_PATTERN = re.compile('|'.join(MATH_PATTERNS), re.UNICODE)


def protect_formulas(text: str) -> Tuple[str, dict]:
    """
    Replaces formulas with placeholders and returns mapping for restoration.
    """
    placeholders = {}
    counter = [0]
    
    def replace_formula(match):
        placeholder = f"__FORMULA_{counter[0]}__"
        placeholders[placeholder] = match.group(0)
        counter[0] += 1
        return placeholder
    
    protected_text = FORMULA_PATTERN.sub(replace_formula, text)
    return protected_text, placeholders


def restore_formulas(text: str, placeholders: dict) -> str:
    """
    Restores formulas from placeholders.
    """
    for placeholder, formula in placeholders.items():
        text = text.replace(placeholder, formula)
    return text


def analyze_pdf(pdf_path: Path) -> Tuple[List[Block], str]:
    """
    Analyzes PDF using pdfplumber (better extraction) with PyPDF2 fallback.
    """
    # Try pdfplumber first (better quality)
    try:
        import pdfplumber
        return _analyze_with_pdfplumber(pdf_path)
    except ImportError:
        logger.warning("pdfplumber not available, falling back to PyPDF2")
    except Exception as e:
        logger.warning("pdfplumber failed: %s, falling back to PyPDF2", e)
    
    # Fallback to PyPDF2
    return _analyze_with_pypdf2(pdf_path)


def _analyze_with_pdfplumber(pdf_path: Path) -> Tuple[List[Block], str]:
    """
    Extract text using pdfplumber - better handling of layout and special characters.
    """
    import pdfplumber
    
    logger.info("analyze_pdf: reading %s with pdfplumber", pdf_path)
    blocks: List[Block] = []
    
    PLACEHOLDER_TOKEN = "___INLINE_GRAPHIC_PLACEHOLDER___"
    reference_patterns = [
        r"(siehe|see|refer\s+to|in)\s+(Fig\.|Figure|Abb\.|Abbildung|Plot|Diagramm|Graph)\s+\d+",
        r"(\[graphics\]|\[image\]|\(plot\))",
    ]
    LATEX_PLACEHOLDER = (
        r"\centering\framebox[0.9\linewidth][c]{\huge[GRAPHIC / PLOT POSITION]}\par\vspace{1em}"
    )
    
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            # Extract text with better layout preservation
            text = page.extract_text(
                x_tolerance=3,
                y_tolerance=3,
                layout=True,  # Preserve layout
                x_density=7.25,
                y_density=13,
            ) or ""
            
            # Process text
            processed_text = text
            for pattern in reference_patterns:
                processed_text = re.sub(
                    pattern,
                    lambda m: f"\n\n{PLACEHOLDER_TOKEN}\n\n{m.group(0)}",
                    processed_text,
                    flags=re.IGNORECASE,
                )
            
            # Split into blocks
            page_blocks = processed_text.split("\n\n")
            
            for content in page_blocks:
                content = content.strip()
                content = content.replace("\u2500", "")
                
                if not content:
                    continue
                
                element_type = "text"
                
                if content == PLACEHOLDER_TOKEN:
                    element_type = "image_placeholder"
                    content = LATEX_PLACEHOLDER
                elif content.lower().startswith("figure") or content.lower().startswith("fig."):
                    element_type = "figure_caption"
                elif content.lower().startswith("table") or content.count("\t") > 2:
                    element_type = "table_content"
                # Detect formula blocks
                elif _is_formula_block(content):
                    element_type = "formula"
                
                if PLACEHOLDER_TOKEN in content:
                    content = content.replace(PLACEHOLDER_TOKEN, "").strip()
                    element_type = "text"
                
                blocks.append(
                    Block(
                        page=i + 1,
                        content=content,
                        element_type=element_type,
                    )
                )
    
    # Detect language
    sample_text = "\n".join(b.content or "" for b in blocks[:3])
    detected = _detect_language(sample_text)
    
    logger.info(
        "analyze_pdf: finished with pdfplumber (pages=%d, blocks=%d, detected_language=%s)",
        len(pdf.pages) if 'pdf' in dir() else 0,
        len(blocks),
        detected,
    )
    
    return blocks, detected


def _is_formula_block(content: str) -> bool:
    """
    Detects if a block is primarily a mathematical formula.
    """
    # High density of math symbols
    math_chars = set('αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ∫∑∏√∞≈≠≤≥±×÷∈∉⊂⊃∪∩∧∨¬∀∃∂∇')
    math_count = sum(1 for c in content if c in math_chars)
    
    # Contains LaTeX-like patterns
    has_latex = bool(re.search(r'\\[a-zA-Z]+|[_^]\{|\$', content))
    
    # Equation pattern (X = ...)
    has_equation = bool(re.search(r'^[A-Z]\s*=', content.strip()))
    
    # Short content with high math density
    if len(content) < 100 and (math_count > 3 or has_latex or has_equation):
        return True
    
    return False


def _detect_language(text: str) -> str:
    """
    Detects the language of the given text.
    """
    try:
        from langdetect import detect
        if text.strip():
            return detect(text)
    except Exception:
        pass
    return "en"


def _analyze_with_pypdf2(pdf_path: Path) -> Tuple[List[Block], str]:
    """
    Fallback extraction using PyPDF2.
    """
    try:
        from PyPDF2 import PdfReader
    except Exception as exc:
        raise RuntimeError(
            "PyPDF2 is required for PDF parsing. Install with `pip install PyPDF2`."
        ) from exc

    logger.info("analyze_pdf: reading %s with PyPDF2", pdf_path)
    reader = PdfReader(str(pdf_path))
    blocks: List[Block] = []

    PLACEHOLDER_TOKEN = "___INLINE_GRAPHIC_PLACEHOLDER___"
    reference_patterns = [
        r"(siehe|see|refer\s+to|in)\s+(Fig\.|Figure|Abb\.|Abbildung|Plot|Diagramm|Graph)\s+\d+",
        r"(\[graphics\]|\[image\]|\(plot\))",
    ]
    LATEX_PLACEHOLDER = (
        r"\centering\framebox[0.9\linewidth][c]{\huge[GRAPHIC / PLOT POSITION]}\par\vspace{1em}"
    )

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        processed_text = text
        for pattern in reference_patterns:
            processed_text = re.sub(
                pattern,
                lambda m: f"\n\n{PLACEHOLDER_TOKEN}\n\n{m.group(0)}",
                processed_text,
                flags=re.IGNORECASE,
            )

        page_blocks = processed_text.split("\n\n")

        for content in page_blocks:
            content = content.strip()
            content = content.replace("\u2500", "")

            if not content:
                continue

            element_type = "text"

            if content == PLACEHOLDER_TOKEN:
                element_type = "image_placeholder"
                content = LATEX_PLACEHOLDER
            elif content.lower().startswith("figure") or content.lower().startswith(
                "fig."
            ):
                element_type = "figure_caption"
            elif content.lower().startswith("table") or content.count("\t") > 2:
                element_type = "table_content"
            # Detect formula blocks
            elif _is_formula_block(content):
                element_type = "formula"

            if PLACEHOLDER_TOKEN in content:
                content = content.replace(PLACEHOLDER_TOKEN, "").strip()
                element_type = "text"

            blocks.append(
                Block(
                    page=i + 1,
                    content=content,
                    element_type=element_type,
                )
            )

    sample_text = "\n".join(b.content or "" for b in blocks[:3])
    detected = _detect_language(sample_text)

    logger.info(
        "analyze_pdf: finished with PyPDF2 (pages=%d, blocks=%d, detected_language=%s)",
        len(reader.pages),
        len(blocks),
        detected,
    )

    return blocks, detected


def translate_blocks(
    blocks: List[Block],
    *,
    source_language: Optional[str],
    target_language: str,
    use_openai: bool,
    openai_api_key: Optional[str],
    use_ollama: bool = False,
    ollama_model: Optional[str] = None,
) -> List[Block]:
    logger.info(
        "translate_blocks: called (blocks=%d, source_language=%s, target_language=%s, use_openai=%s, api_key_present=%s, use_ollama=%s, ollama_model=%s)",
        len(blocks),
        source_language,
        target_language,
        use_openai,
        bool(openai_api_key),
        use_ollama,
        ollama_model,
    )

    # OLLAMA BACKEND (Fallback/Alternative zu OpenAI)
    if use_ollama and ollama_model:
        logger.info("translate_blocks: Using Ollama backend with model %s", ollama_model)
        try:
            from ollama_backend import translate_with_ollama
        except ImportError:
            from ollama_backend import translate_with_ollama
        
        translated: List[Block] = []
        for idx, block in enumerate(blocks):
            original_text = block.content or ""
            
            if not original_text.strip():
                translated.append(block)
                continue
            
            # Skip translation for image placeholders and formula blocks
            if block.element_type in ("image_placeholder", "formula"):
                translated.append(block.model_copy(update={"translated_latex": original_text}))
                continue
            
            out_text = translate_with_ollama(
                text=original_text,
                model=ollama_model,
                source_language=source_language,
                target_language=target_language,
                element_type=block.element_type,
            )
            translated.append(block.model_copy(update={"translated_latex": out_text}))
            logger.debug("translate_blocks: Ollama translated block %d (len_in=%d, len_out=%d)", idx, len(original_text), len(out_text))
        
        logger.info("translate_blocks: Ollama finished (input_blocks=%d, output_blocks=%d)", len(blocks), len(translated))
        return translated

    if not use_openai or not openai_api_key:
        logger.warning(
            "translate_blocks: OpenAI translation disabled (use_openai=%s, api_key_present=%s) – returning original blocks",
            use_openai,
            bool(openai_api_key),
        )
        return blocks

    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        logger.exception("translate_blocks: failed to import OpenAI client: %s", exc)
        return blocks

    client = OpenAI(api_key=openai_api_key)
    translated: List[Block] = []

    for idx, block in enumerate(blocks):
        original_text = block.content or ""

        if not original_text.strip():
            translated.append(block)
            continue

        # Skip translation for image placeholders and formula blocks
        if block.element_type in ("image_placeholder", "formula"):
            translated_block = block.model_copy(update={"translated_latex": original_text})
            translated.append(translated_block)
            continue

        if block.element_type == "figure_caption":
            task_description = f"""
- This is a **Figure Caption**. Translate it from {source_language or "the source language"} to {target_language}.
- **Preserve the leading text** like "Figure 1:" or "Fig. 3.2." exactly as it is, only translating the descriptive text that follows.
- Preserve *all* original LaTeX math segments as they are.
"""
        elif block.element_type == "table_content":
            task_description = f"""
- This is **Table Content (potentially with a title/caption)**. Translate it from {source_language or "the source language"} to {target_language}.
- **Strictly maintain the tabular structure and alignment** using Markdown, LaTeX, or other clear text formatting. Preserve row/column delineation.
- Only translate the text within the table (headers, cells, notes).
- Preserve *all* original LaTeX math segments as they are.
"""
        else:
            task_description = f"""
- Most publications have two text columns per page. read from left to right, put placeholders for image and graphics data.
- Translate the text from {source_language or "the source language"} to {target_language}.
- Translate Umlaute
- Translate it in blocks, while every Abstract, Summary, Chapter and Subchapter is recognized and separated into Blocks.
- Gebe dir verdammt noch mal Mühe, die Struktur und Ordnung des Originals aufrecht zu erhalten.
- Write every mathematical equation or Physical equation or Expression in a new, seperate row.
- Recognize it as equation for LaTeX, do *not* make changes in the equation or expression.
- Texts got side margins. Respect those. Split up overfull columns into readable ones.
- Sätze werden so getrennt, dass sie in die Formatierung des (Ursprungs-)Textes passen und nicht über Ränder hinaus schauen.
- Es gilt die Rechtschreibung! Dort wo stumme Leerzeichen eingefügt sind, sind diese durch normale Leerzeichen zu ersetzen.
- Die ursprüngliche formatierung und Textausrichtung ist Maßgebend für die Qualität der Ausgabe. Wenn du alles in einem Block runter schreibst sieht das fürchterlich aus.
- Strictly preserve *all* formatting, paragraphs, headings, subheadings, titles, subtitles, sections, subsections.
- Preserve *all* LaTeX math segments exactly as they are.
- Do *not* comment at the beginning or end of a block.
- DO NOT insert graphic or image placeholders; this is handled in the pre-processing stage.
"""

        prompt = f"""
You are a professional scientific translator.

Task:
{task_description}
- Do NOT change anything inside these math segments.
- Outside of math, write fluent, natural {target_language} scientific language.

Text to translate:
\"\"\"
{original_text}
\"\"\"
"""

        try:
            completion = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
            )
            out_text = completion.choices[0].message.content or ""
            logger.debug(
                "translate_blocks: block %d on page %s translated (len_in=%d, len_out=%d)",
                idx,
                block.page,
                len(original_text),
                len(out_text),
            )
        except Exception as exc:
            logger.exception(
                "translate_blocks: OpenAI error while translating block %d on page %s: %s",
                idx,
                block.page,
                exc,
            )
            msg = str(exc)
            if "insufficient_quota" in msg or "You exceeded your current quota" in msg:
                raise RuntimeError(
                    "OpenAI API: insufficient quota – please check your plan and billing details."
                ) from exc
            out_text = original_text

        translated.append(block.model_copy(update={"translated_latex": out_text}))

    logger.info(
        "translate_blocks: finished (input_blocks=%d, output_blocks=%d)",
        len(blocks),
        len(translated),
    )
    return translated
