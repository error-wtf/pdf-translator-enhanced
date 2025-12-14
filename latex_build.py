"""
LaTeX Build for PDF-Translator

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import List

from models import Block
import tempfile

logger = logging.getLogger("pdf_translator.latex_build")

# Use system temp directory for Gradio compatibility
JOBS_DIR = Path(tempfile.gettempdir()) / "pdf_translator_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


def cleanup_old_jobs(max_age_hours: int = 24):
    """
    Remove job directories older than max_age_hours.
    Called periodically to prevent temp folder from growing indefinitely.
    """
    import time
    from datetime import datetime
    
    if not JOBS_DIR.exists():
        return
    
    now = time.time()
    max_age_seconds = max_age_hours * 3600
    deleted_count = 0
    
    try:
        for job_dir in JOBS_DIR.iterdir():
            if job_dir.is_dir():
                # Check modification time
                mtime = job_dir.stat().st_mtime
                age = now - mtime
                
                if age > max_age_seconds:
                    try:
                        import shutil
                        shutil.rmtree(job_dir)
                        deleted_count += 1
                        logger.debug(f"Deleted old job: {job_dir.name}")
                    except Exception as e:
                        logger.warning(f"Could not delete {job_dir}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old job directories")
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")


def sanitize_for_latex(text: str) -> str:
    """
    Convert Unicode math symbols to LaTeX commands.
    This is critical for scientific PDFs with formulas.
    """
    if not text:
        return ""

    # === GREEK LETTERS ===
    greek_upper = {
        "Α": r"A", "Β": r"B", "Γ": r"$\Gamma$", "Δ": r"$\Delta$",
        "Ε": r"E", "Ζ": r"Z", "Η": r"H", "Θ": r"$\Theta$",
        "Ι": r"I", "Κ": r"K", "Λ": r"$\Lambda$", "Μ": r"M",
        "Ν": r"N", "Ξ": r"$\Xi$", "Ο": r"O", "Π": r"$\Pi$",
        "Ρ": r"P", "Σ": r"$\Sigma$", "Τ": r"T", "Υ": r"$\Upsilon$",
        "Φ": r"$\Phi$", "Χ": r"X", "Ψ": r"$\Psi$", "Ω": r"$\Omega$",
    }
    greek_lower = {
        "α": r"$\alpha$", "β": r"$\beta$", "γ": r"$\gamma$", "δ": r"$\delta$",
        "ε": r"$\varepsilon$", "ζ": r"$\zeta$", "η": r"$\eta$", "θ": r"$\theta$",
        "ι": r"$\iota$", "κ": r"$\kappa$", "λ": r"$\lambda$", "μ": r"$\mu$",
        "ν": r"$\nu$", "ξ": r"$\xi$", "ο": r"o", "π": r"$\pi$",
        "ρ": r"$\rho$", "σ": r"$\sigma$", "τ": r"$\tau$", "υ": r"$\upsilon$",
        "φ": r"$\varphi$", "χ": r"$\chi$", "ψ": r"$\psi$", "ω": r"$\omega$",
        "ϕ": r"$\phi$", "ϵ": r"$\epsilon$", "ϑ": r"$\vartheta$", "ϱ": r"$\varrho$",
    }
    
    # === MATH OPERATORS & SYMBOLS ===
    math_symbols = {
        # Basic operators
        "−": "-", "–": "-", "—": "-",
        "×": r"$\times$", "÷": r"$\div$", "±": r"$\pm$", "∓": r"$\mp$",
        "·": r"$\cdot$", "∗": r"$\ast$",
        # Relations
        "≤": r"$\leq$", "≥": r"$\geq$", "≠": r"$\neq$", "≈": r"$\approx$",
        "≡": r"$\equiv$", "∝": r"$\propto$", "≪": r"$\ll$", "≫": r"$\gg$",
        "∼": r"$\sim$", "≃": r"$\simeq$",
        # Arrows
        "→": r"$\rightarrow$", "←": r"$\leftarrow$", "↔": r"$\leftrightarrow$",
        "⇒": r"$\Rightarrow$", "⇐": r"$\Leftarrow$", "⇔": r"$\Leftrightarrow$",
        "↑": r"$\uparrow$", "↓": r"$\downarrow$",
        # Calculus & Analysis
        "∂": r"$\partial$", "∇": r"$\nabla$", "∆": r"$\Delta$",
        "∫": r"$\int$", "∮": r"$\oint$", "∑": r"$\sum$", "∏": r"$\prod$",
        "√": r"$\sqrt{}$", "∞": r"$\infty$", "∅": r"$\emptyset$",
        # Set theory
        "∈": r"$\in$", "∉": r"$\notin$", "⊂": r"$\subset$", "⊃": r"$\supset$",
        "⊆": r"$\subseteq$", "⊇": r"$\supseteq$", "∪": r"$\cup$", "∩": r"$\cap$",
        "∧": r"$\wedge$", "∨": r"$\vee$", "¬": r"$\neg$",
        # Special symbols
        "ℏ": r"$\hbar$", "ħ": r"$\hbar$",  # h-bar (Planck constant)
        "ℓ": r"$\ell$",  # script l
        "℘": r"$\wp$",  # Weierstrass p
        "ℜ": r"$\Re$", "ℑ": r"$\Im$",  # Real/Imaginary
        "⊕": r"$\oplus$", "⊗": r"$\otimes$", "⊥": r"$\perp$", "∥": r"$\parallel$",
        "†": r"$\dagger$", "‡": r"$\ddagger$",
        "°": r"$^\circ$",  # degree
        "′": r"$'$", "″": r"$''$",  # prime
        "…": "...",
    }
    
    # === SUBSCRIPTS ===
    subscripts = {
        "₀": r"$_0$", "₁": r"$_1$", "₂": r"$_2$", "₃": r"$_3$", "₄": r"$_4$",
        "₅": r"$_5$", "₆": r"$_6$", "₇": r"$_7$", "₈": r"$_8$", "₉": r"$_9$",
        "₊": r"$_+$", "₋": r"$_-$", "₌": r"$_=$",
        "ₐ": r"$_a$", "ₑ": r"$_e$", "ₒ": r"$_o$", "ₓ": r"$_x$",
        "ₕ": r"$_h$", "ₖ": r"$_k$", "ₗ": r"$_l$", "ₘ": r"$_m$",
        "ₙ": r"$_n$", "ₚ": r"$_p$", "ₛ": r"$_s$", "ₜ": r"$_t$",
    }
    
    # === SUPERSCRIPTS ===
    superscripts = {
        "⁰": r"$^0$", "¹": r"$^1$", "²": r"$^2$", "³": r"$^3$", "⁴": r"$^4$",
        "⁵": r"$^5$", "⁶": r"$^6$", "⁷": r"$^7$", "⁸": r"$^8$", "⁹": r"$^9$",
        "⁺": r"$^+$", "⁻": r"$^-$", "⁼": r"$^=$",
        "ⁿ": r"$^n$", "ⁱ": r"$^i$",
    }
    
    # === SPACES ===
    spaces = {
        "\u202f": " ", "\xa0": " ", "\u2009": " ", "\u2003": " ",
        "\u2002": " ", "\u2004": " ", "\u2005": " ", "\u2006": " ",
    }
    
    # Apply all replacements
    all_replacements = {}
    all_replacements.update(greek_upper)
    all_replacements.update(greek_lower)
    all_replacements.update(math_symbols)
    all_replacements.update(subscripts)
    all_replacements.update(superscripts)
    all_replacements.update(spaces)

    for src, dst in all_replacements.items():
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

    # Try to find pdflatex - check common MiKTeX paths on Windows
    pdflatex_cmd = "pdflatex"
    if os.name == 'nt':  # Windows
        miktex_paths = [
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"),
            os.path.expandvars(r"%PROGRAMFILES%\MiKTeX\miktex\bin\x64\pdflatex.exe"),
            r"C:\MiKTeX\miktex\bin\x64\pdflatex.exe",
        ]
        for path in miktex_paths:
            if os.path.exists(path):
                pdflatex_cmd = path
                logger.info("[latex_build] Found pdflatex at: %s", path)
                break
    
    cmd = [pdflatex_cmd, "-interaction=nonstopmode", "main.tex"]
    try:
        subprocess.run(cmd, cwd=job_dir, check=True, timeout=300)  # SECURITY: 5 min timeout
        logger.info("[latex_build] pdflatex finished successfully for job %s", job_id)
    except FileNotFoundError:
        logger.error("[latex_build] ERROR: pdflatex not found! LaTeX is not installed.")
        raise RuntimeError(
            "pdflatex not found!\n\n"
            "Please install LaTeX:\n"
            "  Windows: winget install MiKTeX.MiKTeX\n"
            "  Linux: sudo apt install texlive-latex-base\n"
            "  macOS: brew install --cask basictex\n\n"
            "After installation, restart the terminal/app!"
        )
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
