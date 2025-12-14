"""
Page-by-Page PDF Translator

Processes PDFs page by page to:
1. Preserve original layout (1:1)
2. Extract and include images
3. Handle large documents (>9 pages)
4. Merge pages at the end

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import logging
import re
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import fitz  # PyMuPDF
from PIL import Image
import io
from text_normalizer import normalize_text, count_garbage_chars

logger = logging.getLogger("pdf_translator.page_by_page")


def split_pdf_into_pages(pdf_path: str, output_dir: Path) -> List[Path]:
    """
    Split a PDF into individual page PDFs.
    Returns list of paths to single-page PDFs.
    """
    doc = fitz.open(pdf_path)
    page_paths = []
    
    for page_num in range(len(doc)):
        page_pdf = output_dir / f"page_{page_num + 1:03d}.pdf"
        
        # Create single-page PDF
        single_doc = fitz.open()
        single_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        single_doc.save(str(page_pdf))
        single_doc.close()
        
        page_paths.append(page_pdf)
        logger.info(f"Extracted page {page_num + 1}/{len(doc)}")
    
    doc.close()
    return page_paths


def extract_images_from_page(pdf_path: Path, output_dir: Path, page_num: int) -> List[Dict]:
    """
    Extract all images from a single PDF page.
    Returns list of image info dicts with path, position, size.
    """
    doc = fitz.open(str(pdf_path))
    page = doc[0]  # Single-page PDF
    images = []
    
    image_list = page.get_images(full=True)
    
    for img_idx, img_info in enumerate(image_list):
        xref = img_info[0]
        
        try:
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Save image
            img_filename = f"page{page_num:03d}_img{img_idx + 1:02d}.{image_ext}"
            img_path = output_dir / img_filename
            
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            
            # Get image position on page
            img_rects = page.get_image_rects(xref)
            if img_rects:
                rect = img_rects[0]
                images.append({
                    "path": img_path,
                    "filename": img_filename,
                    "x": rect.x0,
                    "y": rect.y0,
                    "width": rect.width,
                    "height": rect.height,
                    "page": page_num
                })
                logger.info(f"Extracted image: {img_filename} at ({rect.x0:.1f}, {rect.y0:.1f})")
            else:
                images.append({
                    "path": img_path,
                    "filename": img_filename,
                    "x": 0,
                    "y": 0,
                    "width": 100,
                    "height": 100,
                    "page": page_num
                })
        except Exception as e:
            logger.warning(f"Could not extract image {img_idx}: {e}")
    
    doc.close()
    return images


def extract_text_blocks_from_page(pdf_path: Path) -> List[Dict]:
    """
    Extract text blocks with their positions from a single PDF page.
    Returns list of text block dicts with text, position, font info.
    """
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    
    # Get text blocks with position info
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    
    text_blocks = []
    for block in blocks:
        if block["type"] == 0:  # Text block
            block_text = ""
            font_size = 12
            is_bold = False
            
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    # NORMALIZE each span immediately on extraction
                    span_text = normalize_text(span.get("text", ""))
                    block_text += span_text
                    font_size = span.get("size", 12)
                    font_name = span.get("font", "").lower()
                    is_bold = "bold" in font_name
                block_text += "\n"
            
            # Normalize the complete block text
            normalized_text = normalize_text(block_text.strip())
            
            if not normalized_text:
                continue
            
            text_blocks.append({
                "text": normalized_text,
                "x": block["bbox"][0],
                "y": block["bbox"][1],
                "width": block["bbox"][2] - block["bbox"][0],
                "height": block["bbox"][3] - block["bbox"][1],
                "font_size": font_size,
                "is_bold": is_bold
            })
    
    doc.close()
    return text_blocks


def translate_text_block(text: str, model: str, target_language: str, 
                         use_openai: bool = False, openai_api_key: str = None) -> str:
    """Translate a single text block."""
    from pdf_marker_translator import translate_text_chunk
    return translate_text_chunk(text, model, target_language, use_openai, openai_api_key)


def create_translated_page_pdf(
    original_page: Path,
    text_blocks: List[Dict],
    images: List[Dict],
    output_path: Path,
    target_language: str,
    model: str,
    use_openai: bool = False,
    openai_api_key: str = None,
    progress_callback=None
) -> bool:
    """
    Create a translated PDF page preserving original layout.
    """
    doc = fitz.open(str(original_page))
    page = doc[0]
    
    # Get page dimensions
    page_rect = page.rect
    
    # Create new PDF with same dimensions
    new_doc = fitz.open()
    new_page = new_doc.new_page(width=page_rect.width, height=page_rect.height)
    
    # First, copy images to new page
    for img_info in images:
        try:
            img_rect = fitz.Rect(
                img_info["x"],
                img_info["y"],
                img_info["x"] + img_info["width"],
                img_info["y"] + img_info["height"]
            )
            new_page.insert_image(img_rect, filename=str(img_info["path"]))
            logger.debug(f"Inserted image at {img_rect}")
        except Exception as e:
            logger.warning(f"Could not insert image: {e}")
    
    # Then, add translated text blocks
    for i, block in enumerate(text_blocks):
        if not block["text"].strip():
            continue
        
        # Translate the text
        translated = translate_text_block(
            block["text"], model, target_language,
            use_openai, openai_api_key
        )
        
        if progress_callback:
            progress_callback(i + 1, len(text_blocks), f"Block {i + 1}/{len(text_blocks)}")
        
        # Insert translated text at original position
        text_rect = fitz.Rect(
            block["x"],
            block["y"],
            block["x"] + block["width"],
            block["y"] + block["height"]
        )
        
        # Determine font size (scale down if needed to fit)
        font_size = min(block["font_size"], 11)
        
        try:
            # Use text writer for better control
            new_page.insert_textbox(
                text_rect,
                translated,
                fontsize=font_size,
                fontname="helv",  # Helvetica - widely available
                align=fitz.TEXT_ALIGN_LEFT
            )
        except Exception as e:
            logger.warning(f"Could not insert text block: {e}")
            # Fallback: insert as annotation
            new_page.insert_text(
                (block["x"], block["y"] + font_size),
                translated[:200],  # Truncate if too long
                fontsize=font_size
            )
    
    # Save the new page
    new_doc.save(str(output_path))
    new_doc.close()
    doc.close()
    
    return True


def merge_pdfs(pdf_paths: List[Path], output_path: Path) -> bool:
    """
    Merge multiple PDFs into one.
    """
    if not pdf_paths:
        return False
    
    merged_doc = fitz.open()
    
    for pdf_path in pdf_paths:
        if pdf_path.exists():
            doc = fitz.open(str(pdf_path))
            merged_doc.insert_pdf(doc)
            doc.close()
            logger.info(f"Merged: {pdf_path.name}")
    
    merged_doc.save(str(output_path))
    merged_doc.close()
    
    logger.info(f"Created merged PDF: {output_path}")
    return True


def translate_pdf_page_by_page(
    input_pdf: str,
    output_dir: str,
    model: str,
    target_language: str,
    progress_callback=None,
    use_openai: bool = False,
    openai_api_key: str = None
) -> Tuple[Optional[str], str]:
    """
    Main function: Translates PDF page by page.
    
    1. Split PDF into pages
    2. For each page:
       - Extract images
       - Extract text blocks with positions
       - Translate text blocks
       - Create new page with translated text + original images
    3. Merge all pages
    
    Returns (output_path, status_message)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    translated_pages_dir = output_dir / "translated_pages"
    translated_pages_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Split PDF into pages
        if progress_callback:
            progress_callback(5, 100, "Splitting PDF into pages...")
        
        page_paths = split_pdf_into_pages(input_pdf, pages_dir)
        total_pages = len(page_paths)
        
        if total_pages == 0:
            return None, "❌ Could not extract pages from PDF"
        
        logger.info(f"Split PDF into {total_pages} pages")
        
        # Step 2: Process each page
        translated_page_paths = []
        
        for page_num, page_path in enumerate(page_paths, 1):
            page_progress_base = 10 + int(80 * (page_num - 1) / total_pages)
            
            if progress_callback:
                progress_callback(page_progress_base, 100, f"Processing page {page_num}/{total_pages}...")
            
            # Extract images from this page
            images = extract_images_from_page(page_path, images_dir, page_num)
            logger.info(f"Page {page_num}: Found {len(images)} images")
            
            # Extract text blocks
            text_blocks = extract_text_blocks_from_page(page_path)
            logger.info(f"Page {page_num}: Found {len(text_blocks)} text blocks")
            
            # Create translated page
            translated_page_path = translated_pages_dir / f"translated_page_{page_num:03d}.pdf"
            
            def page_progress(current, total, msg):
                if progress_callback:
                    sub_progress = page_progress_base + int(80 / total_pages * current / max(total, 1))
                    progress_callback(sub_progress, 100, f"Page {page_num}: {msg}")
            
            success = create_translated_page_pdf(
                page_path,
                text_blocks,
                images,
                translated_page_path,
                target_language,
                model,
                use_openai,
                openai_api_key,
                page_progress
            )
            
            if success and translated_page_path.exists():
                translated_page_paths.append(translated_page_path)
            else:
                logger.warning(f"Failed to translate page {page_num}")
        
        # Step 3: Merge all translated pages
        if progress_callback:
            progress_callback(95, 100, "Merging pages...")
        
        output_pdf = output_dir / "translated.pdf"
        merge_success = merge_pdfs(translated_page_paths, output_pdf)
        
        if merge_success and output_pdf.exists():
            if progress_callback:
                progress_callback(100, 100, "Complete!")
            
            # Copy to stable location
            stable_output = Path(__file__).parent / "output" / "translated_paged.pdf"
            stable_output.parent.mkdir(exist_ok=True)
            shutil.copy2(output_pdf, stable_output)
            
            return str(output_pdf), f"✅ Translation complete!\n\n📄 {total_pages} pages processed\n🖼️ Images preserved\n📁 Also saved to: {stable_output}"
        
        return None, f"❌ Failed to merge translated pages"
        
    except Exception as e:
        logger.exception(f"Page-by-page translation failed: {e}")
        return None, f"❌ Translation failed: {str(e)}"


# Test function
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python page_by_page_translator.py input.pdf [output_dir]")
        sys.exit(1)
    
    input_pdf = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output_paged"
    
    logging.basicConfig(level=logging.INFO)
    
    result, status = translate_pdf_page_by_page(
        input_pdf,
        output_dir,
        model="qwen2.5:7b",
        target_language="German",
        progress_callback=lambda c, t, m: print(f"[{c}/{t}] {m}")
    )
    
    print(f"\nResult: {result}")
    print(f"Status: {status}")
