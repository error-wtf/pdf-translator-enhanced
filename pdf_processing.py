from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from .models import Block

logger = logging.getLogger("pdf_translator.pdf_processing")


def analyze_pdf(pdf_path: Path) -> Tuple[List[Block], str]:
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyPDF2 is required for PDF parsing. Install with `pip install PyPDF2`."
        ) from exc

    logger.info("analyze_pdf: reading %s", pdf_path)
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
    detected = "en"

    try:
        from langdetect import detect  # type: ignore

        if sample_text.strip():
            detected = detect(sample_text)
    except Exception:
        detected = "en"

    logger.info(
        "analyze_pdf: finished (pages=%d, blocks=%d, detected_language=%s)",
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
) -> List[Block]:
    logger.info(
        "translate_blocks: called (blocks=%d, source_language=%s, target_language=%s, use_openai=%s, api_key_present=%s)",
        len(blocks),
        source_language,
        target_language,
        use_openai,
        bool(openai_api_key),
    )

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

        if block.element_type == "image_placeholder":
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
