"""
Layout Engine - Precise PDF Layout Reconstruction

Ensures translated PDFs maintain the exact visual layout of the original:
- Font matching and fallback
- Text reflow with proper line breaks
- Column preservation
- Spacing and margins

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import logging
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("pdf_translator.layout")


# =============================================================================
# FONT CONFIGURATION
# =============================================================================

# Unicode-capable fonts for different scripts
FONT_FALLBACKS = {
    # Latin-based languages (German, French, Spanish, Italian, etc.)
    "latin": [
        "Helvetica",
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
        "Noto Sans",
    ],
    # Cyrillic (Russian, Ukrainian, etc.)
    "cyrillic": [
        "DejaVu Sans",
        "Liberation Sans",
        "Noto Sans",
        "Arial",
    ],
    # Greek
    "greek": [
        "DejaVu Sans",
        "Noto Sans",
        "Arial",
    ],
    # CJK (Chinese, Japanese, Korean)
    "cjk": [
        "Noto Sans CJK",
        "Source Han Sans",
        "MS Gothic",
        "SimSun",
    ],
    # Arabic/Hebrew
    "rtl": [
        "Noto Sans Arabic",
        "Noto Sans Hebrew",
        "Arial",
    ],
}

# Language to script mapping
LANGUAGE_SCRIPTS = {
    "German": "latin",
    "French": "latin",
    "Spanish": "latin",
    "Italian": "latin",
    "Portuguese": "latin",
    "Dutch": "latin",
    "Polish": "latin",
    "Czech": "latin",
    "Swedish": "latin",
    "Norwegian": "latin",
    "Danish": "latin",
    "Finnish": "latin",
    "Russian": "cyrillic",
    "Ukrainian": "cyrillic",
    "Bulgarian": "cyrillic",
    "Greek": "greek",
    "Chinese": "cjk",
    "Japanese": "cjk",
    "Korean": "cjk",
    "Arabic": "rtl",
    "Hebrew": "rtl",
}


def get_font_for_language(target_language: str, original_font: str = "") -> str:
    """Get appropriate font for target language."""
    script = LANGUAGE_SCRIPTS.get(target_language, "latin")
    fonts = FONT_FALLBACKS.get(script, FONT_FALLBACKS["latin"])
    
    # Try to match original font style
    if original_font:
        original_lower = original_font.lower()
        if "bold" in original_lower:
            return fonts[0]  # + bold flag separately
        if "italic" in original_lower:
            return fonts[0]  # + italic flag separately
    
    return fonts[0]


# =============================================================================
# TEXT BLOCK LAYOUT
# =============================================================================

@dataclass
class TextStyle:
    """Text styling information."""
    font_name: str = "Helvetica"
    font_size: float = 10.0
    is_bold: bool = False
    is_italic: bool = False
    color: Tuple[float, float, float] = (0, 0, 0)  # RGB
    line_height: float = 1.2  # multiplier


@dataclass
class LayoutBlock:
    """A positioned text block with layout info."""
    text: str
    x: float
    y: float
    width: float
    height: float
    style: TextStyle = field(default_factory=TextStyle)
    column: int = 0  # 0 = single column, 1/2 = left/right
    is_heading: bool = False
    is_caption: bool = False
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def center_x(self) -> float:
        return self.x + self.width / 2
    
    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


@dataclass
class PageLayout:
    """Complete page layout information."""
    width: float
    height: float
    blocks: List[LayoutBlock] = field(default_factory=list)
    columns: int = 1  # 1 or 2
    margins: Dict[str, float] = field(default_factory=lambda: {
        "top": 72, "bottom": 72, "left": 72, "right": 72
    })
    
    @property
    def content_width(self) -> float:
        return self.width - self.margins["left"] - self.margins["right"]
    
    @property
    def content_height(self) -> float:
        return self.height - self.margins["top"] - self.margins["bottom"]


# =============================================================================
# COLUMN DETECTION
# =============================================================================

def detect_columns(blocks: List[Dict], page_width: float) -> int:
    """
    Detect if page uses single or double column layout.
    
    Returns 1 or 2.
    """
    if len(blocks) < 4:
        return 1
    
    # Get x-positions of block centers
    centers = [b.get("x", 0) + b.get("width", 0) / 2 for b in blocks]
    
    # Check distribution
    mid_page = page_width / 2
    left_count = sum(1 for c in centers if c < mid_page * 0.8)
    right_count = sum(1 for c in centers if c > mid_page * 1.2)
    
    # If significant blocks on both sides, likely 2-column
    if left_count > 3 and right_count > 3:
        return 2
    
    return 1


def assign_columns(blocks: List[LayoutBlock], page_width: float, num_columns: int) -> List[LayoutBlock]:
    """Assign each block to a column."""
    if num_columns == 1:
        for block in blocks:
            block.column = 0
        return blocks
    
    mid_page = page_width / 2
    
    for block in blocks:
        if block.center_x < mid_page:
            block.column = 1  # Left
        else:
            block.column = 2  # Right
    
    return blocks


# =============================================================================
# TEXT REFLOW
# =============================================================================

def calculate_text_width(text: str, font_size: float, avg_char_width: float = 0.5) -> float:
    """Estimate text width in points."""
    # Approximate: average character is 0.5 * font_size wide
    return len(text) * font_size * avg_char_width


def reflow_text(
    text: str,
    max_width: float,
    font_size: float,
    hyphenate: bool = True
) -> List[str]:
    """
    Reflow text to fit within max_width.
    
    Returns list of lines.
    """
    if not text:
        return []
    
    words = text.split()
    lines = []
    current_line = []
    current_width = 0
    
    avg_char_width = font_size * 0.5
    space_width = avg_char_width
    
    for word in words:
        word_width = len(word) * avg_char_width
        
        if current_width + word_width + space_width <= max_width:
            current_line.append(word)
            current_width += word_width + space_width
        else:
            # Word doesn't fit
            if current_line:
                lines.append(" ".join(current_line))
            
            # Check if word itself is too long
            if word_width > max_width and hyphenate:
                # Hyphenate long word
                hyphenated = hyphenate_word(word, max_width, avg_char_width)
                for part in hyphenated[:-1]:
                    lines.append(part + "-")
                current_line = [hyphenated[-1]]
                current_width = len(hyphenated[-1]) * avg_char_width
            else:
                current_line = [word]
                current_width = word_width
    
    if current_line:
        lines.append(" ".join(current_line))
    
    return lines


def hyphenate_word(word: str, max_width: float, char_width: float) -> List[str]:
    """Simple word hyphenation for very long words."""
    max_chars = int(max_width / char_width) - 1  # -1 for hyphen
    
    if max_chars < 3:
        return [word]
    
    parts = []
    remaining = word
    
    while len(remaining) > max_chars:
        # Find a good break point
        break_point = max_chars
        
        # Prefer breaking at certain patterns
        for i in range(max_chars - 2, max_chars // 2, -1):
            if i < len(remaining):
                # Break after vowels or common suffixes
                if remaining[i] in "aeiouäöüAEIOUÄÖÜ":
                    break_point = i + 1
                    break
        
        parts.append(remaining[:break_point])
        remaining = remaining[break_point:]
    
    parts.append(remaining)
    return parts


# =============================================================================
# FONT SIZE SCALING
# =============================================================================

def calculate_scaled_font_size(
    original_text: str,
    translated_text: str,
    original_font_size: float,
    block_width: float,
    min_font_size: float = 6.0,
    max_reduction: float = 0.7
) -> float:
    """
    Calculate font size for translated text to fit in same space.
    
    German text is typically ~30% longer than English.
    """
    if not original_text or not translated_text:
        return original_font_size
    
    length_ratio = len(translated_text) / len(original_text)
    
    if length_ratio <= 1.0:
        # Translation is shorter or same - keep original size
        return original_font_size
    
    # Scale down proportionally
    scaled_size = original_font_size / (length_ratio ** 0.5)
    
    # Apply limits
    min_allowed = max(min_font_size, original_font_size * max_reduction)
    
    return max(scaled_size, min_allowed)


# =============================================================================
# PAGE RECONSTRUCTION
# =============================================================================

def reconstruct_page_layout(
    original_blocks: List[Dict],
    translated_blocks: List[str],
    page_width: float,
    page_height: float,
    target_language: str
) -> PageLayout:
    """
    Reconstruct page layout with translated text.
    
    Maintains original positions while adjusting for text length.
    """
    layout = PageLayout(width=page_width, height=page_height)
    
    # Detect columns
    layout.columns = detect_columns(original_blocks, page_width)
    
    for i, (orig_block, trans_text) in enumerate(zip(original_blocks, translated_blocks)):
        # Extract original style
        style = TextStyle(
            font_name=get_font_for_language(target_language, orig_block.get("font_name", "")),
            font_size=orig_block.get("font_size", 10),
            is_bold=orig_block.get("is_bold", False),
            is_italic=orig_block.get("is_italic", False),
        )
        
        # Scale font if needed
        original_text = orig_block.get("text", "")
        style.font_size = calculate_scaled_font_size(
            original_text,
            trans_text,
            style.font_size,
            orig_block.get("width", 200)
        )
        
        # Create layout block
        block = LayoutBlock(
            text=trans_text,
            x=orig_block.get("x", 0),
            y=orig_block.get("y", 0),
            width=orig_block.get("width", 200),
            height=orig_block.get("height", 20),
            style=style,
            is_heading=is_heading_block(orig_block),
            is_caption=is_caption_block(orig_block),
        )
        
        layout.blocks.append(block)
    
    # Assign columns
    layout.blocks = assign_columns(layout.blocks, page_width, layout.columns)
    
    return layout


def is_heading_block(block: Dict) -> bool:
    """Check if block is a heading."""
    font_size = block.get("font_size", 10)
    is_bold = block.get("is_bold", False)
    text = block.get("text", "").strip()
    
    # Headings are typically larger and/or bold
    if font_size > 12 and is_bold:
        return True
    
    # Or numbered sections
    if re.match(r'^\d+\.?\s+\w', text):
        return True
    
    return False


def is_caption_block(block: Dict) -> bool:
    """Check if block is a figure/table caption."""
    text = block.get("text", "").strip().lower()
    return bool(re.match(r'^(figure|fig\.?|table|tab\.?)\s*\d+', text))


# =============================================================================
# PDF RENDERING
# =============================================================================

def render_layout_to_pdf(layout: PageLayout, output_path: str) -> bool:
    """
    Render a PageLayout to a PDF file using PyMuPDF.
    
    Returns True on success.
    """
    try:
        import fitz
        
        doc = fitz.open()
        page = doc.new_page(width=layout.width, height=layout.height)
        
        for block in layout.blocks:
            # Create text rect
            rect = fitz.Rect(
                block.x,
                block.y,
                block.x + block.width,
                block.y + block.height
            )
            
            # Determine font
            fontname = "helv"  # Helvetica
            if block.style.is_bold:
                fontname = "hebo"  # Helvetica Bold
            
            # Reflow text to fit
            lines = reflow_text(
                block.text,
                block.width,
                block.style.font_size
            )
            text_to_insert = "\n".join(lines)
            
            # Insert text
            try:
                page.insert_textbox(
                    rect,
                    text_to_insert,
                    fontname=fontname,
                    fontsize=block.style.font_size,
                    align=fitz.TEXT_ALIGN_LEFT
                )
            except Exception as e:
                logger.warning(f"Text insertion failed: {e}")
                # Fallback: insert as simple text
                page.insert_text(
                    (block.x, block.y + block.style.font_size),
                    block.text[:100],
                    fontsize=block.style.font_size
                )
        
        doc.save(output_path)
        doc.close()
        
        logger.info(f"Rendered layout to {output_path}")
        return True
        
    except Exception as e:
        logger.exception(f"Failed to render layout: {e}")
        return False


# =============================================================================
# LAYOUT COMPARISON
# =============================================================================

def compare_layouts(original: PageLayout, translated: PageLayout) -> Dict:
    """
    Compare original and translated layouts.
    
    Returns metrics about layout preservation.
    """
    metrics = {
        "block_count_match": len(original.blocks) == len(translated.blocks),
        "column_match": original.columns == translated.columns,
        "position_drift": [],  # List of position differences
        "size_changes": [],    # List of size changes
    }
    
    for orig, trans in zip(original.blocks, translated.blocks):
        # Position drift
        x_drift = abs(orig.x - trans.x)
        y_drift = abs(orig.y - trans.y)
        metrics["position_drift"].append((x_drift, y_drift))
        
        # Size change
        width_change = trans.width / orig.width if orig.width > 0 else 1
        height_change = trans.height / orig.height if orig.height > 0 else 1
        metrics["size_changes"].append((width_change, height_change))
    
    # Calculate averages
    if metrics["position_drift"]:
        avg_x = sum(d[0] for d in metrics["position_drift"]) / len(metrics["position_drift"])
        avg_y = sum(d[1] for d in metrics["position_drift"]) / len(metrics["position_drift"])
        metrics["avg_position_drift"] = (avg_x, avg_y)
    
    return metrics


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Layout Engine Test ===\n")
    
    # Test font selection
    for lang in ["German", "Russian", "Chinese", "Arabic"]:
        font = get_font_for_language(lang)
        print(f"{lang}: {font}")
    
    # Test text reflow
    print("\n### Text Reflow Test")
    long_text = "Die Schwarzschild-Metrik beschreibt die Raumzeitgeometrie außerhalb einer kugelsymmetrischen nicht-rotierenden Masse."
    lines = reflow_text(long_text, 200, 10)
    print(f"Original: {long_text}")
    print(f"Reflowed ({len(lines)} lines):")
    for line in lines:
        print(f"  {line}")
    
    # Test font scaling
    print("\n### Font Scaling Test")
    orig = "The black hole has an event horizon."
    trans = "Das schwarze Loch hat einen Ereignishorizont."
    scaled = calculate_scaled_font_size(orig, trans, 12, 200)
    print(f"Original: {len(orig)} chars, 12pt")
    print(f"Translated: {len(trans)} chars, {scaled:.1f}pt")
    
    # Test column detection
    print("\n### Column Detection")
    single_col_blocks = [
        {"x": 50, "width": 500},
        {"x": 50, "width": 500},
    ]
    double_col_blocks = [
        {"x": 50, "width": 200},
        {"x": 350, "width": 200},
        {"x": 50, "width": 200},
        {"x": 350, "width": 200},
    ]
    print(f"Single-column test: {detect_columns(single_col_blocks, 600)} column(s)")
    print(f"Double-column test: {detect_columns(double_col_blocks, 600)} column(s)")
    
    print("\n✅ Layout Engine ready")
