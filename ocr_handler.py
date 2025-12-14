"""
OCR Handler - Extract text from images in PDFs
© 2025 Sven Kalinowski - Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import fitz  # PyMuPDF

logger = logging.getLogger("pdf_translator.ocr")

# Try to import OCR libraries
OCR_AVAILABLE = False
try:
    import pytesseract
    from PIL import Image
    import io
    OCR_AVAILABLE = True
    logger.info("OCR available via pytesseract")
except ImportError:
    logger.warning("pytesseract not installed - OCR disabled")


@dataclass
class ImageText:
    """Text extracted from an image."""
    text: str
    rect: fitz.Rect
    confidence: float
    image_index: int


def check_ocr_available() -> bool:
    """Check if OCR is available."""
    return OCR_AVAILABLE


def extract_images_from_page(page: fitz.Page) -> List[Tuple[int, fitz.Rect, bytes]]:
    """Extract images from a PDF page."""
    images = []
    
    image_list = page.get_images(full=True)
    
    for img_index, img in enumerate(image_list):
        xref = img[0]
        
        try:
            # Get image data
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Find image rectangle on page
            for item in page.get_image_rects(xref):
                images.append((img_index, item, image_bytes))
                break
                
        except Exception as e:
            logger.warning(f"Could not extract image {img_index}: {e}")
    
    return images


def ocr_image(image_bytes: bytes, lang: str = "eng") -> Tuple[str, float]:
    """
    Perform OCR on an image.
    
    Args:
        image_bytes: Raw image data
        lang: Tesseract language code (eng, deu, fra, etc.)
        
    Returns:
        Tuple of (extracted_text, confidence)
    """
    if not OCR_AVAILABLE:
        return "", 0.0
    
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Perform OCR with confidence data
        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
        
        # Build text and calculate average confidence
        texts = []
        confidences = []
        
        for i, text in enumerate(data['text']):
            conf = int(data['conf'][i])
            if conf > 0 and text.strip():
                texts.append(text)
                confidences.append(conf)
        
        full_text = ' '.join(texts)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        return full_text, avg_conf / 100.0
        
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return "", 0.0


def extract_text_from_images(page: fitz.Page, lang: str = "eng", min_confidence: float = 0.5) -> List[ImageText]:
    """
    Extract text from all images on a page using OCR.
    
    Args:
        page: PyMuPDF page object
        lang: Tesseract language code
        min_confidence: Minimum confidence threshold
        
    Returns:
        List of ImageText objects with extracted text
    """
    if not OCR_AVAILABLE:
        logger.warning("OCR not available - skipping image text extraction")
        return []
    
    results = []
    images = extract_images_from_page(page)
    
    for img_index, rect, image_bytes in images:
        text, confidence = ocr_image(image_bytes, lang)
        
        if text.strip() and confidence >= min_confidence:
            results.append(ImageText(
                text=text,
                rect=rect,
                confidence=confidence,
                image_index=img_index
            ))
            logger.debug(f"OCR extracted from image {img_index}: {text[:50]}... (conf={confidence:.0%})")
    
    return results


def get_tesseract_lang(target_language: str) -> str:
    """Map target language to Tesseract language code."""
    lang_map = {
        "German": "deu",
        "English": "eng",
        "French": "fra",
        "Spanish": "spa",
        "Italian": "ita",
        "Portuguese": "por",
        "Russian": "rus",
        "Chinese": "chi_sim",
        "Japanese": "jpn",
        "Korean": "kor",
        "Arabic": "ara",
        "Dutch": "nld",
        "Polish": "pol",
        "Turkish": "tur",
    }
    return lang_map.get(target_language, "eng")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def should_ocr_image(rect: fitz.Rect, min_size: float = 50.0) -> bool:
    """Check if an image is large enough to potentially contain text."""
    return rect.width >= min_size and rect.height >= min_size


def has_significant_text(text: str, min_words: int = 3) -> bool:
    """Check if OCR result contains significant text content."""
    words = text.split()
    # Filter out single characters and noise
    meaningful_words = [w for w in words if len(w) > 1]
    return len(meaningful_words) >= min_words
