"""
Nougat OCR Extractor - Scientific PDF to LaTeX/Markdown

Meta's Nougat (Neural Optical Understanding for Academic Documents) is
specifically trained on scientific papers and excels at:
- Formula extraction as proper LaTeX
- Table structure preservation
- Multi-column layout handling
- Reference parsing

This module provides Nougat as a fallback/alternative to Marker,
especially useful when Marker hangs or produces poor formula output.

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4

Installation:
    pip install nougat-ocr
    
Or for GPU acceleration:
    pip install nougat-ocr[gpu]
"""
from __future__ import annotations

import logging
import os
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, List, Callable

logger = logging.getLogger("pdf_translator.nougat")


# =============================================================================
# NOUGAT AVAILABILITY CHECK
# =============================================================================

_NOUGAT_AVAILABLE: Optional[bool] = None


def is_nougat_available() -> bool:
    """
    Check if Nougat OCR is installed and available.
    
    Returns:
        True if Nougat can be used
    """
    global _NOUGAT_AVAILABLE
    
    if _NOUGAT_AVAILABLE is not None:
        return _NOUGAT_AVAILABLE
    
    # Try importing nougat
    try:
        import nougat
        _NOUGAT_AVAILABLE = True
        logger.info("Nougat OCR is available")
        return True
    except ImportError:
        pass
    
    # Try command line
    try:
        result = subprocess.run(
            ["nougat", "--help"],
            capture_output=True,
            timeout=10
        )
        _NOUGAT_AVAILABLE = result.returncode == 0
        if _NOUGAT_AVAILABLE:
            logger.info("Nougat CLI is available")
        return _NOUGAT_AVAILABLE
    except Exception:
        pass
    
    _NOUGAT_AVAILABLE = False
    logger.warning("Nougat OCR is not available - install with: pip install nougat-ocr")
    return False


def get_nougat_install_instructions() -> str:
    """Returns instructions for installing Nougat."""
    return """
## Installing Nougat OCR

Nougat is Meta's Neural Optical Understanding for Academic Documents.
It's excellent for extracting formulas and tables from scientific PDFs.

### Basic Installation (CPU)
```bash
pip install nougat-ocr
```

### GPU Installation (Faster)
```bash
pip install nougat-ocr[gpu]
```

### First Run
On first use, Nougat will download the model (~1.5 GB).
This only happens once.

### Requirements
- Python 3.8+
- ~4 GB RAM (CPU) or ~6 GB VRAM (GPU)
- ~2 GB disk space for model
"""


# =============================================================================
# NOUGAT EXTRACTION
# =============================================================================

def extract_with_nougat(
    pdf_path: str,
    output_dir: Optional[str] = None,
    model: str = "0.1.0-small",
    batch_size: int = 1,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> str:
    """
    Extract text and formulas from PDF using Nougat OCR.
    
    Nougat outputs Markdown with LaTeX math notation, which is
    ideal for scientific document translation.
    
    Args:
        pdf_path: Path to input PDF file
        output_dir: Directory for output files (temp dir if None)
        model: Nougat model to use:
            - "0.1.0-small": Faster, less accurate
            - "0.1.0-base": Balanced (default)
        batch_size: Pages to process in parallel (higher = more VRAM)
        progress_callback: Optional callback(current_page, total_pages, status)
    
    Returns:
        Extracted text as Markdown with LaTeX formulas
    
    Raises:
        RuntimeError: If Nougat is not available
    """
    if not is_nougat_available():
        raise RuntimeError(
            "Nougat OCR is not available. Install with: pip install nougat-ocr"
        )
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Create output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="nougat_")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Extracting {pdf_path} with Nougat...")
    
    if progress_callback:
        progress_callback(0, 100, "Starting Nougat extraction...")
    
    try:
        # Try Python API first
        return _extract_with_nougat_api(
            pdf_path, output_dir, model, batch_size, progress_callback
        )
    except ImportError:
        # Fall back to CLI
        return _extract_with_nougat_cli(
            pdf_path, output_dir, model, batch_size, progress_callback
        )


def _extract_with_nougat_api(
    pdf_path: Path,
    output_dir: Path,
    model: str,
    batch_size: int,
    progress_callback: Optional[Callable],
) -> str:
    """Extract using Nougat Python API."""
    from nougat import NougatModel
    from nougat.utils.dataset import LazyDataset
    from nougat.utils.checkpoint import get_checkpoint
    import torch
    from PIL import Image
    import fitz  # PyMuPDF
    
    # Load model
    if progress_callback:
        progress_callback(0, 100, "Loading Nougat model...")
    
    checkpoint = get_checkpoint(model_tag=model)
    nougat_model = NougatModel.from_pretrained(checkpoint)
    
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nougat_model = nougat_model.to(device)
    nougat_model.eval()
    
    # Open PDF and convert pages to images
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    if progress_callback:
        progress_callback(0, total_pages, f"Processing {total_pages} pages...")
    
    results = []
    
    for page_num in range(total_pages):
        if progress_callback:
            progress_callback(page_num + 1, total_pages, f"Page {page_num + 1}/{total_pages}")
        
        # Render page to image
        page = doc[page_num]
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Process with Nougat
        with torch.no_grad():
            output = nougat_model.inference(image=img)
        
        if output:
            results.append(output)
    
    doc.close()
    
    # Combine results
    markdown = "\n\n".join(results)
    
    # Save to file
    output_file = output_dir / f"{pdf_path.stem}.mmd"
    output_file.write_text(markdown, encoding="utf-8")
    
    if progress_callback:
        progress_callback(total_pages, total_pages, "Extraction complete!")
    
    logger.info(f"Extracted {len(results)} pages to {output_file}")
    return markdown


def _extract_with_nougat_cli(
    pdf_path: Path,
    output_dir: Path,
    model: str,
    batch_size: int,
    progress_callback: Optional[Callable],
) -> str:
    """Extract using Nougat CLI (fallback)."""
    if progress_callback:
        progress_callback(0, 100, "Running Nougat CLI...")
    
    # Build command
    cmd = [
        "nougat",
        str(pdf_path),
        "-o", str(output_dir),
        "--model", model,
        "--batchsize", str(batch_size),
        "--no-skipping",  # Don't skip failed pages
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes max
        )
        
        if result.returncode != 0:
            logger.error(f"Nougat CLI error: {result.stderr}")
            raise RuntimeError(f"Nougat failed: {result.stderr}")
        
        # Find output file
        output_file = output_dir / f"{pdf_path.stem}.mmd"
        if not output_file.exists():
            # Try .md extension
            output_file = output_dir / f"{pdf_path.stem}.md"
        
        if output_file.exists():
            markdown = output_file.read_text(encoding="utf-8")
            if progress_callback:
                progress_callback(100, 100, "Extraction complete!")
            return markdown
        else:
            raise RuntimeError(f"Nougat output not found in {output_dir}")
            
    except subprocess.TimeoutExpired:
        raise RuntimeError("Nougat timed out after 10 minutes")


# =============================================================================
# FORMULA EXTRACTION HELPERS
# =============================================================================

def extract_formulas_only(pdf_path: str) -> List[str]:
    """
    Extract only the mathematical formulas from a PDF.
    
    Returns list of LaTeX formula strings.
    """
    import re
    
    markdown = extract_with_nougat(pdf_path)
    
    formulas = []
    
    # Display math: $$ ... $$ or \[ ... \]
    display_math = re.findall(r'\$\$(.*?)\$\$', markdown, re.DOTALL)
    formulas.extend(display_math)
    
    bracket_math = re.findall(r'\\\[(.*?)\\\]', markdown, re.DOTALL)
    formulas.extend(bracket_math)
    
    # Inline math: $ ... $
    inline_math = re.findall(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', markdown)
    formulas.extend(inline_math)
    
    # equation environments
    env_math = re.findall(
        r'\\begin\{(equation|align|gather|multline)\*?\}(.*?)\\end\{\1\*?\}',
        markdown, re.DOTALL
    )
    formulas.extend([m[1] for m in env_math])
    
    # Clean up
    formulas = [f.strip() for f in formulas if f.strip()]
    
    logger.info(f"Extracted {len(formulas)} formulas from {pdf_path}")
    return formulas


def has_complex_formulas(pdf_path: str, sample_pages: int = 3) -> bool:
    """
    Quick check if a PDF contains complex mathematical formulas.
    
    Useful for deciding whether to use Nougat (better for formulas)
    or Marker (faster for simple text).
    
    Args:
        pdf_path: Path to PDF
        sample_pages: Number of pages to sample
    
    Returns:
        True if complex formulas detected
    """
    try:
        import fitz
        
        doc = fitz.open(pdf_path)
        text = ""
        
        # Sample first N pages
        for i in range(min(sample_pages, len(doc))):
            text += doc[i].get_text()
        
        doc.close()
        
        # Look for formula indicators
        formula_indicators = [
            r'\$',           # LaTeX math
            r'\\frac',       # Fractions
            r'\\int',        # Integrals
            r'\\sum',        # Summations
            r'\\sqrt',       # Square roots
            r'\\partial',    # Partial derivatives
            r'\\nabla',      # Nabla operator
            r'\\vec',        # Vectors
            r'\\mathbf',     # Bold math
            r'\\alpha',      # Greek letters
            r'\\beta',
            r'\\gamma',
            r'→',            # Arrow
            r'≈',            # Approximately
            r'≤',            # Less than or equal
            r'≥',            # Greater than or equal
            r'∫',            # Integral symbol
            r'∑',            # Sum symbol
            r'∏',            # Product symbol
        ]
        
        import re
        count = sum(len(re.findall(p, text)) for p in formula_indicators)
        
        # If more than 10 formula indicators, consider it complex
        return count > 10
        
    except Exception as e:
        logger.warning(f"Could not analyze PDF for formulas: {e}")
        return False


# =============================================================================
# SMART EXTRACTOR - CHOOSES BEST METHOD
# =============================================================================

def smart_extract(
    pdf_path: str,
    prefer_nougat: bool = False,
    marker_timeout: int = 120,
    progress_callback: Optional[Callable] = None,
) -> str:
    """
    Intelligently choose the best extraction method.
    
    Decision logic:
    1. If prefer_nougat=True and Nougat available → use Nougat
    2. If PDF has complex formulas → try Nougat first
    3. Otherwise → try Marker first, fallback to Nougat
    
    Args:
        pdf_path: Path to PDF
        prefer_nougat: Force Nougat if available
        marker_timeout: Timeout for Marker in seconds
        progress_callback: Optional progress callback
    
    Returns:
        Extracted Markdown text
    """
    nougat_available = is_nougat_available()
    
    # Check for complex formulas
    has_formulas = has_complex_formulas(pdf_path)
    
    if has_formulas:
        logger.info("PDF contains complex formulas")
    
    # Decision: Nougat first if formulas detected or preferred
    if (prefer_nougat or has_formulas) and nougat_available:
        logger.info("Using Nougat (formula-optimized extraction)")
        try:
            return extract_with_nougat(pdf_path, progress_callback=progress_callback)
        except Exception as e:
            logger.warning(f"Nougat failed: {e}, falling back to Marker")
    
    # Try Marker
    try:
        from pdf_marker_translator import pdf_to_markdown_with_marker
        
        if progress_callback:
            progress_callback(0, 100, "Using Marker extraction...")
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            markdown = pdf_to_markdown_with_marker(
                pdf_path, tmpdir, timeout_seconds=marker_timeout
            )
            if markdown and len(markdown) > 100:
                return markdown
    except Exception as e:
        logger.warning(f"Marker failed: {e}")
    
    # Fallback to Nougat if available
    if nougat_available:
        logger.info("Falling back to Nougat")
        return extract_with_nougat(pdf_path, progress_callback=progress_callback)
    
    # Last resort: basic PyMuPDF extraction
    logger.warning("Using basic PyMuPDF extraction (formulas may be lost)")
    import fitz
    doc = fitz.open(pdf_path)
    text = "\n\n".join(page.get_text() for page in doc)
    doc.close()
    return text


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=== Nougat Extractor Test ===\n")
    
    # Check availability
    if is_nougat_available():
        print("✅ Nougat is available")
    else:
        print("❌ Nougat is not available")
        print(get_nougat_install_instructions())
        sys.exit(1)
    
    # Test with a PDF if provided
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"\nExtracting: {pdf_path}")
        
        def progress(current, total, status):
            print(f"  [{current}/{total}] {status}")
        
        try:
            result = extract_with_nougat(pdf_path, progress_callback=progress)
            print(f"\n=== Result ({len(result)} chars) ===")
            print(result[:2000])
            if len(result) > 2000:
                print(f"... ({len(result) - 2000} more chars)")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("\nUsage: python nougat_extractor.py <pdf_file>")
