"""
Caption Anchoring - Pair figure/table captions with their images

Ensures captions stay anchored to their corresponding figures/tables
during translation and layout reconstruction.

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("pdf_translator.caption_anchoring")


@dataclass
class ImageBlock:
    """An image with position info."""
    path: str
    x: float
    y: float
    width: float
    height: float
    page: int = 0
    
    @property
    def center_x(self) -> float:
        return self.x + self.width / 2
    
    @property
    def center_y(self) -> float:
        return self.y + self.height / 2
    
    @property
    def y1(self) -> float:
        return self.y + self.height


@dataclass
class CaptionBlock:
    """A figure/table caption."""
    text: str
    x: float
    y: float
    width: float
    height: float
    caption_type: str = "figure"  # "figure" or "table"
    number: int = 0
    
    @property
    def center_x(self) -> float:
        return self.x + self.width / 2


@dataclass
class AnchoredFigure:
    """A figure with its anchored caption."""
    image: ImageBlock
    caption: Optional[CaptionBlock] = None
    
    def to_latex(self, image_path: str) -> str:
        """Generate LaTeX for this figure."""
        latex = "\\begin{figure}[H]\n"
        latex += "\\centering\n"
        latex += f"\\includegraphics[width=0.8\\textwidth]{{{image_path}}}\n"
        
        if self.caption:
            # Escape special characters in caption
            caption_text = self._escape_latex(self.caption.text)
            # Remove "Figure X:" prefix if present (LaTeX adds it automatically)
            caption_text = re.sub(r'^(Figure|Fig\.?)\s*\d+[:\.]?\s*', '', caption_text, flags=re.IGNORECASE)
            latex += f"\\caption{{{caption_text}}}\n"
        
        latex += "\\end{figure}\n"
        return latex
    
    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters."""
        # Don't escape $ for math mode
        replacements = [
            ('\\', '\\textbackslash{}'),
            ('%', '\\%'),
            ('#', '\\#'),
            ('_', '\\_'),
            ('{', '\\{'),
            ('}', '\\}'),
            ('&', '\\&'),
            ('~', '\\textasciitilde{}'),
            ('^', '\\textasciicircum{}'),
        ]
        for old, new in replacements:
            if old != '\\':
                text = text.replace(old, new)
        return text


def is_figure_caption(text: str) -> Tuple[bool, str, int]:
    """
    Check if text is a figure caption.
    
    Returns (is_caption, type, number)
    """
    text = text.strip()
    
    # Figure patterns
    fig_match = re.match(r'^(Figure|Fig\.?)\s*(\d+)', text, re.IGNORECASE)
    if fig_match:
        return True, "figure", int(fig_match.group(2))
    
    # Table patterns
    tab_match = re.match(r'^(Table|Tab\.?)\s*(\d+)', text, re.IGNORECASE)
    if tab_match:
        return True, "table", int(tab_match.group(2))
    
    # Abbildung (German)
    abb_match = re.match(r'^(Abbildung|Abb\.?)\s*(\d+)', text, re.IGNORECASE)
    if abb_match:
        return True, "figure", int(abb_match.group(2))
    
    # Tabelle (German)
    tabelle_match = re.match(r'^(Tabelle|Tab\.?)\s*(\d+)', text, re.IGNORECASE)
    if tabelle_match:
        return True, "table", int(tabelle_match.group(2))
    
    return False, "", 0


def find_nearest_image(caption: CaptionBlock, 
                       images: List[ImageBlock],
                       max_distance: float = 100.0,
                       same_column_tolerance: float = 50.0) -> Optional[ImageBlock]:
    """
    Find the nearest image to a caption.
    
    Prioritizes:
    1. Images in the same column (similar x position)
    2. Images directly above or below the caption
    3. Closest by y-distance
    """
    if not images:
        return None
    
    best_image = None
    best_score = float('inf')
    
    for img in images:
        # Calculate horizontal alignment score (lower is better)
        x_diff = abs(img.center_x - caption.center_x)
        same_column = x_diff < same_column_tolerance
        
        # Calculate vertical distance
        if caption.y > img.y + img.height:
            # Caption is below image
            y_dist = caption.y - (img.y + img.height)
        elif img.y > caption.y + caption.height:
            # Caption is above image
            y_dist = img.y - (caption.y + caption.height)
        else:
            # Overlapping
            y_dist = 0
        
        if y_dist > max_distance:
            continue
        
        # Score: prioritize same column, then y distance
        score = y_dist + (0 if same_column else 1000) + x_diff * 0.1
        
        if score < best_score:
            best_score = score
            best_image = img
    
    return best_image


def anchor_captions_to_images(text_blocks: List[Dict], 
                               images: List[Dict],
                               max_distance: float = 100.0) -> Tuple[List[AnchoredFigure], List[Dict]]:
    """
    Anchor captions to their corresponding images.
    
    Args:
        text_blocks: List of text block dicts
        images: List of image dicts with 'path', 'x', 'y', 'width', 'height'
        max_distance: Maximum y-distance to consider for anchoring
    
    Returns:
        Tuple of (anchored_figures, remaining_text_blocks)
    """
    # Convert images to ImageBlock objects
    image_blocks = []
    for img in images:
        image_blocks.append(ImageBlock(
            path=img.get('path', ''),
            x=img.get('x', 0),
            y=img.get('y', 0),
            width=img.get('width', 100),
            height=img.get('height', 100),
            page=img.get('page', 0)
        ))
    
    # Find captions in text blocks
    captions = []
    caption_indices = set()
    
    for i, block in enumerate(text_blocks):
        text = block.get('text', '')
        is_cap, cap_type, cap_num = is_figure_caption(text)
        
        if is_cap:
            captions.append(CaptionBlock(
                text=text,
                x=block.get('x', 0),
                y=block.get('y', 0),
                width=block.get('width', 100),
                height=block.get('height', 20),
                caption_type=cap_type,
                number=cap_num
            ))
            caption_indices.add(i)
    
    # Anchor captions to images
    anchored = []
    used_images = set()
    used_captions = set()
    
    for cap_idx, caption in enumerate(captions):
        if caption.caption_type != "figure":
            continue  # Only anchor figure captions to images
        
        # Find available images (not yet used)
        available_images = [img for i, img in enumerate(image_blocks) if i not in used_images]
        
        nearest = find_nearest_image(caption, available_images, max_distance)
        
        if nearest:
            img_idx = image_blocks.index(nearest)
            used_images.add(img_idx)
            used_captions.add(cap_idx)
            
            anchored.append(AnchoredFigure(
                image=nearest,
                caption=caption
            ))
            logger.info(f"Anchored caption '{caption.text[:50]}...' to image at ({nearest.x}, {nearest.y})")
    
    # Add unanchored images as figures without captions
    for i, img in enumerate(image_blocks):
        if i not in used_images:
            anchored.append(AnchoredFigure(image=img, caption=None))
            logger.info(f"Image at ({img.x}, {img.y}) has no caption")
    
    # Build remaining text blocks (excluding used captions)
    remaining = []
    for i, block in enumerate(text_blocks):
        if i not in caption_indices:
            remaining.append(block)
        elif i in caption_indices:
            # Check if this caption was used
            cap_text = block.get('text', '')
            was_used = any(
                cap.text == cap_text 
                for cap_idx, cap in enumerate(captions) 
                if cap_idx in used_captions
            )
            if not was_used:
                remaining.append(block)  # Keep unused captions
    
    logger.info(f"Anchored {len([a for a in anchored if a.caption])} captions to images")
    
    return anchored, remaining


def sort_blocks_reading_order(blocks: List[Dict], 
                              line_tolerance: float = 15.0,
                              column_threshold: float = 300.0) -> List[Dict]:
    """
    Sort text blocks in reading order.
    
    Handles:
    - Single column: top to bottom
    - Multi-column: left column first, then right column
    - Same line: left to right
    
    Args:
        blocks: List of text block dicts
        line_tolerance: Y-distance to consider blocks on same line
        column_threshold: X-distance to detect multi-column layout
    """
    if not blocks:
        return []
    
    # Detect if multi-column
    x_positions = sorted(set(b.get('x', 0) for b in blocks))
    is_multi_column = len(x_positions) > 1 and (max(x_positions) - min(x_positions)) > column_threshold
    
    if is_multi_column:
        # Split into columns
        mid_x = (min(x_positions) + max(x_positions)) / 2
        left_col = [b for b in blocks if b.get('x', 0) < mid_x]
        right_col = [b for b in blocks if b.get('x', 0) >= mid_x]
        
        # Sort each column by y, then x
        left_col.sort(key=lambda b: (b.get('y', 0), b.get('x', 0)))
        right_col.sort(key=lambda b: (b.get('y', 0), b.get('x', 0)))
        
        # Combine: left column first, then right
        return left_col + right_col
    else:
        # Single column: group by lines, sort lines by y, within line by x
        sorted_blocks = sorted(blocks, key=lambda b: (b.get('y', 0), b.get('x', 0)))
        
        # Group into lines
        lines = []
        current_line = [sorted_blocks[0]]
        current_y = sorted_blocks[0].get('y', 0)
        
        for block in sorted_blocks[1:]:
            block_y = block.get('y', 0)
            if abs(block_y - current_y) <= line_tolerance:
                current_line.append(block)
            else:
                # Sort current line by x
                current_line.sort(key=lambda b: b.get('x', 0))
                lines.append(current_line)
                current_line = [block]
                current_y = block_y
        
        if current_line:
            current_line.sort(key=lambda b: b.get('x', 0))
            lines.append(current_line)
        
        # Flatten
        result = []
        for line in lines:
            result.extend(line)
        
        return result


# Test
if __name__ == "__main__":
    # Test caption detection
    test_captions = [
        "Figure 1: This is a test caption",
        "Fig. 2: Another caption",
        "Table 1: Test table",
        "Abbildung 3: German caption",
        "Regular text, not a caption",
    ]
    
    for text in test_captions:
        is_cap, cap_type, num = is_figure_caption(text)
        print(f"'{text[:30]}...' -> is_caption={is_cap}, type={cap_type}, num={num}")
    
    # Test anchoring
    test_blocks = [
        {'text': 'Some text', 'x': 50, 'y': 50, 'width': 200, 'height': 20},
        {'text': 'Figure 1: Test caption', 'x': 100, 'y': 250, 'width': 200, 'height': 20},
        {'text': 'More text', 'x': 50, 'y': 300, 'width': 200, 'height': 20},
    ]
    
    test_images = [
        {'path': 'img1.png', 'x': 100, 'y': 100, 'width': 200, 'height': 150},
    ]
    
    anchored, remaining = anchor_captions_to_images(test_blocks, test_images)
    
    print(f"\nAnchored figures: {len(anchored)}")
    for fig in anchored:
        print(f"  Image at ({fig.image.x}, {fig.image.y})")
        if fig.caption:
            print(f"    Caption: {fig.caption.text[:50]}")
    
    print(f"\nRemaining blocks: {len(remaining)}")
    for block in remaining:
        print(f"  {block['text'][:50]}")
