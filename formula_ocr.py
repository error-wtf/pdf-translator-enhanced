"""
Formula OCR - Recognize LaTeX formulas in images

Many PDFs embed formulas as images rather than text. This module
uses pix2tex (LaTeX-OCR) to recognize these formulas and convert
them back to LaTeX notation.

Use cases:
- Older PDFs with embedded formula images
- Scanned documents
- PDFs where formula extraction fails

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4

Installation:
    pip install pix2tex
    
Or with all dependencies:
    pip install "pix2tex[gui]"
"""
from __future__ import annotations

import logging
import io
from pathlib import Path
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass

logger = logging.getLogger("pdf_translator.formula_ocr")


# =============================================================================
# AVAILABILITY CHECK
# =============================================================================

_PIX2TEX_AVAILABLE: Optional[bool] = None


def is_pix2tex_available() -> bool:
    """Check if pix2tex is installed and available."""
    global _PIX2TEX_AVAILABLE
    
    if _PIX2TEX_AVAILABLE is not None:
        return _PIX2TEX_AVAILABLE
    
    try:
        from pix2tex.cli import LatexOCR
        _PIX2TEX_AVAILABLE = True
        logger.info("pix2tex (LaTeX-OCR) is available")
        return True
    except ImportError:
        _PIX2TEX_AVAILABLE = False
        logger.warning("pix2tex not available - install with: pip install pix2tex")
        return False


def get_install_instructions() -> str:
    """Returns installation instructions for pix2tex."""
    return """
## Installing pix2tex (LaTeX-OCR)

pix2tex recognizes mathematical formulas in images and converts
them to LaTeX notation.

### Installation
```bash
pip install pix2tex
```

### With GUI (optional)
```bash
pip install "pix2tex[gui]"
```

### First Run
On first use, pix2tex will download the model (~500 MB).

### Requirements
- Python 3.7+
- ~2 GB RAM
- ~1 GB disk space for model
"""


# =============================================================================
# FORMULA DETECTION
# =============================================================================

@dataclass
class FormulaImage:
    """An image that potentially contains a formula."""
    data: bytes
    x: float
    y: float
    width: float
    height: float
    is_formula: bool = False
    latex: str = ""
    confidence: float = 0.0


def is_likely_formula_image(
    image_data: bytes,
    width: float,
    height: float,
    min_width: float = 20,
    max_aspect_ratio: float = 10.0,
) -> bool:
    """
    Heuristic check if an image is likely a formula.
    
    Formulas typically:
    - Are wider than tall (equations) or roughly square (single symbols)
    - Have mostly white/light background
    - Are relatively small (not full-page figures)
    - Have high contrast (black text on white)
    
    Args:
        image_data: Raw image bytes
        width: Image width in PDF units
        height: Image height in PDF units
        min_width: Minimum width to consider
        max_aspect_ratio: Maximum width/height ratio
    
    Returns:
        True if image is likely a formula
    """
    if width < min_width:
        return False
    
    # Aspect ratio check
    aspect = width / max(height, 1)
    if aspect > max_aspect_ratio:
        return False  # Too wide, probably a diagram
    
    try:
        from PIL import Image
        import numpy as np
        
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        
        arr = np.array(img)
        
        # Check if mostly white background (formulas typically are)
        white_ratio = np.sum(arr > 240) / arr.size
        if white_ratio < 0.5:
            return False  # Too much dark area, probably a figure
        
        # Check for high contrast (text)
        has_dark = np.sum(arr < 50) > 0.01 * arr.size
        if not has_dark:
            return False  # No dark pixels, probably blank
        
        # Size check: formulas are usually small
        if img.width > 1000 and img.height > 500:
            return False  # Too large, probably a figure
        
        return True
        
    except Exception as e:
        logger.debug(f"Could not analyze image: {e}")
        # If we can't analyze, assume it might be a formula if small
        return width < 400 and height < 200


def detect_formula_images(
    images: List[dict],
    strict: bool = False
) -> List[FormulaImage]:
    """
    Detect which images in a list are likely formulas.
    
    Args:
        images: List of image dicts with 'data', 'x', 'y', 'width', 'height'
        strict: If True, be more conservative in detection
    
    Returns:
        List of FormulaImage objects with is_formula flag set
    """
    results = []
    
    for img in images:
        formula_img = FormulaImage(
            data=img.get("data", b""),
            x=img.get("x", 0),
            y=img.get("y", 0),
            width=img.get("width", 0),
            height=img.get("height", 0),
        )
        
        # Check if likely a formula
        formula_img.is_formula = is_likely_formula_image(
            formula_img.data,
            formula_img.width,
            formula_img.height,
            min_width=30 if strict else 20,
            max_aspect_ratio=5.0 if strict else 10.0,
        )
        
        results.append(formula_img)
    
    formula_count = sum(1 for f in results if f.is_formula)
    logger.info(f"Detected {formula_count}/{len(images)} images as potential formulas")
    
    return results


# =============================================================================
# FORMULA RECOGNITION
# =============================================================================

_OCR_MODEL = None


def get_ocr_model():
    """Get or create the pix2tex OCR model (singleton)."""
    global _OCR_MODEL
    
    if _OCR_MODEL is not None:
        return _OCR_MODEL
    
    if not is_pix2tex_available():
        return None
    
    try:
        from pix2tex.cli import LatexOCR
        logger.info("Loading pix2tex model...")
        _OCR_MODEL = LatexOCR()
        logger.info("pix2tex model loaded")
        return _OCR_MODEL
    except Exception as e:
        logger.error(f"Failed to load pix2tex model: {e}")
        return None


def recognize_formula(image_data: bytes) -> Tuple[str, float]:
    """
    Recognize a formula in an image using pix2tex.
    
    Args:
        image_data: Raw image bytes (PNG, JPEG, etc.)
    
    Returns:
        Tuple of (latex_string, confidence)
        Returns ("", 0.0) if recognition fails
    """
    model = get_ocr_model()
    if model is None:
        return "", 0.0
    
    try:
        from PIL import Image
        
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Run OCR
        latex = model(img)
        
        # pix2tex doesn't provide confidence, estimate from output
        confidence = 0.8 if latex and len(latex) > 2 else 0.3
        
        # Clean up the result
        latex = latex.strip()
        
        # Remove outer $ if present (we'll add them ourselves)
        if latex.startswith('$') and latex.endswith('$'):
            latex = latex[1:-1]
        
        logger.debug(f"Recognized formula: {latex[:50]}...")
        return latex, confidence
        
    except Exception as e:
        logger.warning(f"Formula recognition failed: {e}")
        return "", 0.0


def recognize_formulas_batch(
    formula_images: List[FormulaImage],
    only_detected: bool = True,
) -> List[FormulaImage]:
    """
    Recognize formulas in multiple images.
    
    Args:
        formula_images: List of FormulaImage objects
        only_detected: If True, only process images flagged as formulas
    
    Returns:
        Updated list with latex and confidence filled in
    """
    model = get_ocr_model()
    if model is None:
        logger.warning("Formula OCR not available")
        return formula_images
    
    processed = 0
    for img in formula_images:
        if only_detected and not img.is_formula:
            continue
        
        latex, confidence = recognize_formula(img.data)
        img.latex = latex
        img.confidence = confidence
        processed += 1
    
    logger.info(f"Processed {processed} formula images")
    return formula_images


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def process_page_images(
    images: List[dict],
    recognize: bool = True,
) -> Tuple[List[dict], List[str]]:
    """
    Process images from a PDF page, detecting and recognizing formulas.
    
    Args:
        images: List of image dicts from PyMuPDF
        recognize: If True, run OCR on detected formulas
    
    Returns:
        Tuple of (non_formula_images, formula_latex_list)
    """
    # Detect formula images
    formula_images = detect_formula_images(images)
    
    # Recognize formulas if requested
    if recognize and is_pix2tex_available():
        formula_images = recognize_formulas_batch(formula_images)
    
    # Separate formulas from regular images
    non_formula_images = []
    formula_latex = []
    
    for i, fimg in enumerate(formula_images):
        if fimg.is_formula and fimg.latex:
            # Add position info for later placement
            formula_latex.append({
                "latex": f"${fimg.latex}$",
                "x": fimg.x,
                "y": fimg.y,
                "width": fimg.width,
                "height": fimg.height,
            })
        else:
            # Keep as regular image
            non_formula_images.append(images[i])
    
    logger.info(f"Extracted {len(formula_latex)} formulas, {len(non_formula_images)} regular images")
    return non_formula_images, formula_latex


def image_to_latex_or_placeholder(
    image_data: bytes,
    x: float,
    y: float,
    width: float,
    height: float,
    fallback_path: str = "image.png",
) -> str:
    """
    Convert an image to LaTeX: either recognized formula or includegraphics.
    
    Args:
        image_data: Raw image bytes
        x, y, width, height: Position info
        fallback_path: Path to use if image is not a formula
    
    Returns:
        LaTeX string (formula or \\includegraphics)
    """
    # Check if it's a formula
    if is_likely_formula_image(image_data, width, height):
        latex, confidence = recognize_formula(image_data)
        if latex and confidence > 0.5:
            return f"${latex}$"
    
    # Not a formula or recognition failed - use image
    return f"\\includegraphics[width={width}pt]{{{fallback_path}}}"


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=== Formula OCR Test ===\n")
    
    # Check availability
    if is_pix2tex_available():
        print("✅ pix2tex is available")
    else:
        print("❌ pix2tex is not available")
        print(get_install_instructions())
        sys.exit(1)
    
    # Test with an image file if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\nProcessing: {image_path}")
        
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Check if formula
        is_formula = is_likely_formula_image(image_data, 200, 50)
        print(f"Detected as formula: {is_formula}")
        
        # Recognize
        if is_formula or "--force" in sys.argv:
            latex, confidence = recognize_formula(image_data)
            print(f"\nRecognized LaTeX (confidence: {confidence:.2f}):")
            print(f"  {latex}")
    else:
        print("\nUsage: python formula_ocr.py <image_file> [--force]")
        print("\nTest with a formula image to see recognition results.")
