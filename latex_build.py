from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import List

from .models import Block

logger = logging.getLogger("pdf_translator.latex_build")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_for_latex(text: str) -> str:
    if not text:
        return ""

    replacements = {
        "−": "-",
        "–": "-",
        "—": "-",
        "…": "...",
        "Σ": r"\Sigma",
        "\u202f": " ",
        "\xa0": " ",
        "\u2009": " ",      # thin space
        "\u2003": " ",      # em space
        "₀": r"$_0$",       # subscript zero
        "≤": r"\leq ",
    }

    for src, dst in replacements.items():
        text = text.replace(src, dst)

    return text


def _babel_for_language(target_language: str) -> str:
    if not target_language:
        return "english"

    lang = target_language.lower()

    if lang.startswith("de"):
        return "ngerman"
    if lang.startswith("en"):
        return "english"
    if lang.startswith("fr"):
        return "french"
    if lang.startswith("es"):
        return "spanish"

    if lang.startswith("ja") or lang.startswith("jp"):
        return "english"
    if lang.startswith("zh") or lang.startswith("cn") or lang.startswith("ch"):
        return "english"

    return "english"


def _needs_cjk(target_language: str) -> tuple[bool, str]:
    lang = (target_language or "").lower()

    if lang.startswith("ja") or lang.startswith("jp"):
        return True, "min"

    if lang.startswith("zh") or lang.startswith("cn") or lang.startswith("ch"):
        return True, "gbsn"

    return False, ""


def render_latex(
    job_id: str,
    blocks: List[Block],
    target_language: str = "de",
    source_language: str | None = None,
) -> str:
    babel_lang = _babel_for_language(target_language)
    use_cjk, cjk_font = _needs_cjk(target_language)

    preamble = rf"""\documentclass[11pt]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage{{amsmath,amssymb}}
\usepackage{{geometry}}
\geometry{{margin=2cm}}
\usepackage[{babel_lang}]{{babel}}
\usepackage{{CJKutf8}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{caption}}

"""

    if use_cjk:
        document_begin = rf"""\begin{{document}}
\begin{{CJK}}{{UTF8}}{{{cjk_font}}}
"""
        document_end = r"""\end{CJK}
\end{document}
"""
    else:
        document_begin = r"""\begin{document}
"""
        document_end = r"""\end{document}
"""

    body_parts: list[str] = []

    for idx, block in enumerate(blocks):
        raw_text = block.translated_latex or block.content or ""
        safe_text = sanitize_for_latex(raw_text)

        used_translation = (
            bool(block.translated_latex)
            and (block.translated_latex != (block.content or ""))
        )

        block_header = (
            f"% Block from page {block.page}, job {job_id}, index {idx}, "
            f"used_translation={used_translation}, element_type={block.element_type}\n"
        )

        wrapped_text = safe_text

        if block.element_type == "figure_caption":
            wrapped_text = (
                f"\\begin{{figure}}[h!]\n\\centering\n"
                f"\\caption{{{safe_text}}}\n\\end{{figure}}\n"
            )
        elif block.element_type == "table_content":
            wrapped_text = (
                f"\\begin{{table}}[h!]\n\\centering\n{safe_text}\n\\end{{table}}\n"
            )
        elif block.element_type == "image_placeholder":
            wrapped_text = safe_text

        body_parts.append(block_header + wrapped_text + "\n")

    body = "\n".join(body_parts)

    logger.info(
        "render_latex: built LaTeX for job %s (blocks=%d, target_language=%s)",
        job_id,
        len(blocks),
        target_language,
    )

    return preamble + document_begin + body + document_end


def build_and_compile(
    job_id: str,
    blocks: List[Block],
    target_language: str = "de",
    source_language: str | None = None,
) -> None:
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    latex_source = render_latex(
        job_id=job_id,
        blocks=blocks,
        target_language=target_language,
        source_language=source_language,
    )

    tex_path = job_dir / "main.tex"
    with tex_path.open("w", encoding="utf-8") as f:
        f.write(latex_source)

    logger.info("[latex_build] main.tex written to %s", tex_path)

    cmd = ["pdflatex", "-interaction=nonstopmode", "main.tex"]
    try:
        subprocess.run(cmd, cwd=job_dir, check=True)
        logger.info("[latex_build] pdflatex finished successfully for job %s", job_id)
    except subprocess.CalledProcessError as exc:
        logger.error(
            "[latex_build] ERROR: pdflatex returned non-zero exit status for job %s: %s",
            job_id,
            exc,
        )
        pdf_path = job_dir / "main.pdf"
        if pdf_path.exists():
            logger.warning(
                "[latex_build] pdflatex reported errors but main.pdf exists for job %s, continuing",
                job_id,
            )
        else:
            raise
