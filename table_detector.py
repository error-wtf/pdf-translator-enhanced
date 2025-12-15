"""
Table Detector - Detect and reconstruct tables from PDF text blocks

Detects table regions by analyzing aligned text boxes and reconstructs
them as LaTeX tabular environments.

Features:
- Heuristic-based detection (alignment analysis)
- Optional ML-based detection (Table Transformer)
- Header row detection
- Multi-language caption support
- Merged cell handling
- Tables-first hardening (grid detection, cell stitching, artifact repair)

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4

Optional ML Installation:
    pip install transformers torch torchvision
"""
from __future__ import annotations

import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger("pdf_translator.table_detector")

# Import table hardening module
try:
    from table_hardening import (
        harden_table, TableArtifactResolver, GridDetector, 
        CellStitcher, StitchedCell, CellToken
    )
    TABLE_HARDENING_AVAILABLE = True
    logger.info("Table hardening module loaded")
except ImportError:
    TABLE_HARDENING_AVAILABLE = False
    logger.debug("Table hardening module not available")


# =============================================================================
# ML AVAILABILITY CHECK
# =============================================================================

_TATR_AVAILABLE: Optional[bool] = None


def is_tatr_available() -> bool:
    """Check if Table Transformer is available."""
    global _TATR_AVAILABLE
    
    if _TATR_AVAILABLE is not None:
        return _TATR_AVAILABLE
    
    try:
        from transformers import TableTransformerForObjectDetection
        import torch
        _TATR_AVAILABLE = True
        logger.info("Table Transformer (TATR) is available")
        return True
    except ImportError:
        _TATR_AVAILABLE = False
        logger.debug("Table Transformer not available")
        return False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TextBlock:
    """A text block with position and content."""
    text: str
    x: float
    y: float
    width: float
    height: float
    font_size: float = 10.0
    is_bold: bool = False
    font_name: str = ""
    
    @property
    def x1(self) -> float:
        return self.x + self.width
    
    @property
    def y1(self) -> float:
        return self.y + self.height
    
    @property
    def center_x(self) -> float:
        return self.x + self.width / 2
    
    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


@dataclass
class TableCell:
    """A cell in a detected table."""
    text: str
    row: int
    col: int
    x: float
    y: float
    is_header: bool = False
    rowspan: int = 1
    colspan: int = 1


@dataclass
class DetectedTable:
    """A detected table with its cells and metadata."""
    cells: List[TableCell]
    rows: int
    cols: int
    x: float
    y: float
    width: float
    height: float
    caption: Optional[str] = None
    confidence: float = 0.0
    has_header: bool = False
    detection_method: str = "heuristic"
    
    def to_latex(self, translate_func=None) -> str:
        """
        Convert table to LaTeX tabular environment.
        
        Args:
            translate_func: Optional function to translate cell text
        """
        if not self.cells:
            return ""
        
        # Build grid
        grid = [['' for _ in range(self.cols)] for _ in range(self.rows)]
        header_rows = set()
        
        for cell in self.cells:
            if 0 <= cell.row < self.rows and 0 <= cell.col < self.cols:
                text = cell.text.strip()
                if translate_func:
                    text = translate_func(text)
                grid[cell.row][cell.col] = text
                if cell.is_header:
                    header_rows.add(cell.row)
        
        # Generate LaTeX
        col_spec = '|' + 'l|' * self.cols
        lines = [f"\\begin{{tabular}}{{{col_spec}}}"]
        lines.append("\\hline")
        
        for row_idx, row in enumerate(grid):
            escaped_row = [self._escape_latex(cell) for cell in row]
            
            # Bold header rows
            if row_idx in header_rows:
                escaped_row = [f"\\textbf{{{cell}}}" for cell in escaped_row]
            
            lines.append(' & '.join(escaped_row) + ' \\\\')
            lines.append("\\hline")
        
        lines.append("\\end{tabular}")
        
        # Wrap in table environment with caption
        if self.caption:
            caption_text = self._escape_latex(self.caption)
            if translate_func:
                # Don't translate "Table X:" prefix
                match = re.match(r'^(Table|Tab\.?|Tabelle|Tableau|Tabla|Tabella)\s*\d+[.:]?\s*', 
                                caption_text, re.IGNORECASE)
                if match:
                    prefix = match.group()
                    rest = caption_text[len(prefix):]
                    rest = translate_func(rest) if rest else ""
                    caption_text = prefix + rest
                else:
                    caption_text = translate_func(caption_text)
            
            latex = "\\begin{table}[H]\n\\centering\n"
            latex += '\n'.join(lines)
            latex += f"\n\\caption{{{caption_text}}}"
            latex += "\n\\end{table}"
            return latex
        
        return '\n'.join(lines)
    
    def to_markdown(self) -> str:
        """Convert table to Markdown format."""
        if not self.cells:
            return ""
        
        # Build grid
        grid = [['' for _ in range(self.cols)] for _ in range(self.rows)]
        for cell in self.cells:
            if 0 <= cell.row < self.rows and 0 <= cell.col < self.cols:
                grid[cell.row][cell.col] = cell.text.strip()
        
        lines = []
        
        # Add caption if present
        if self.caption:
            lines.append(f"**{self.caption}**\n")
        
        for row_idx, row in enumerate(grid):
            lines.append('| ' + ' | '.join(row) + ' |')
            
            # Add separator after first row (header)
            if row_idx == 0:
                lines.append('|' + '|'.join(['---'] * self.cols) + '|')
        
        return '\n'.join(lines)
    
    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters."""
        if not text:
            return ""
        
        # Don't escape if already contains LaTeX commands
        if '\\' in text and any(cmd in text for cmd in ['\\frac', '\\sqrt', '\\sum', '\\int']):
            return text
        
        replacements = [
            ('&', '\\&'),
            ('%', '\\%'),
            ('$', '\\$'),
            ('#', '\\#'),
            ('_', '\\_'),
            ('{', '\\{'),
            ('}', '\\}'),
            ('~', '\\textasciitilde{}'),
            ('^', '\\textasciicircum{}'),
        ]
        
        for old, new in replacements:
            text = text.replace(old, new)
        
        return text


# =============================================================================
# CAPTION DETECTION
# =============================================================================

# Multi-language table caption patterns
TABLE_CAPTION_PATTERNS = [
    r'^(Table|Tab\.?)\s*\d+',           # English
    r'^(Tabelle|Tab\.?)\s*\d+',         # German
    r'^(Tableau|Tab\.?)\s*\d+',         # French
    r'^(Tabla|Tab\.?)\s*\d+',           # Spanish
    r'^(Tabella|Tab\.?)\s*\d+',         # Italian
    r'^(Tabela|Tab\.?)\s*\d+',          # Portuguese
    r'^(Таблица)\s*\d+',                # Russian
    r'^(表)\s*\d+',                     # Chinese/Japanese
]


def is_table_caption(text: str) -> bool:
    """Check if text looks like a table caption."""
    text = text.strip()
    for pattern in TABLE_CAPTION_PATTERNS:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    return False


def find_table_caption(blocks: List[TextBlock], table: DetectedTable, 
                       max_distance: float = 50.0) -> Optional[str]:
    """Find caption for a detected table."""
    for block in blocks:
        if is_table_caption(block.text):
            # Check if above table
            if block.y < table.y and table.y - block.y1 < max_distance:
                return block.text
            # Check if below table
            elif block.y > table.y + table.height and block.y - (table.y + table.height) < max_distance:
                return block.text
    return None


# =============================================================================
# HEURISTIC TABLE DETECTION
# =============================================================================

def find_aligned_columns(blocks: List[TextBlock], tolerance: float = 8.0) -> List[List[TextBlock]]:
    """Group blocks by x-position (column alignment)."""
    if not blocks:
        return []
    
    sorted_blocks = sorted(blocks, key=lambda b: b.x)
    
    columns = []
    current_col = [sorted_blocks[0]]
    current_x = sorted_blocks[0].x
    
    for block in sorted_blocks[1:]:
        if abs(block.x - current_x) <= tolerance:
            current_col.append(block)
        else:
            columns.append(current_col)
            current_col = [block]
            current_x = block.x
    
    if current_col:
        columns.append(current_col)
    
    return columns


def find_aligned_rows(blocks: List[TextBlock], tolerance: float = 8.0) -> List[List[TextBlock]]:
    """Group blocks by y-position (row alignment)."""
    if not blocks:
        return []
    
    sorted_blocks = sorted(blocks, key=lambda b: b.y)
    
    rows = []
    current_row = [sorted_blocks[0]]
    current_y = sorted_blocks[0].y
    
    for block in sorted_blocks[1:]:
        if abs(block.y - current_y) <= tolerance:
            current_row.append(block)
        else:
            rows.append(current_row)
            current_row = [block]
            current_y = block.y
    
    if current_row:
        rows.append(current_row)
    
    return rows


def detect_header_row(rows: List[List[TextBlock]]) -> int:
    """
    Detect which row is the header row.
    
    Headers typically have:
    - Bold text
    - Different font size
    - All cells filled
    
    Returns row index or -1 if no header detected.
    """
    if not rows:
        return -1
    
    first_row = rows[0]
    
    # Check if first row is all bold
    all_bold = all(b.is_bold for b in first_row)
    if all_bold:
        return 0
    
    # Check if first row has different font size
    if len(rows) > 1:
        first_row_size = sum(b.font_size for b in first_row) / len(first_row)
        second_row_size = sum(b.font_size for b in rows[1]) / len(rows[1])
        
        if abs(first_row_size - second_row_size) > 1.0:
            return 0
    
    # Check if first row has more cells (fully filled header)
    if len(rows) > 1 and len(first_row) > len(rows[1]):
        return 0
    
    return -1


def detect_table_region(blocks: List[TextBlock], 
                        min_rows: int = 2, 
                        min_cols: int = 2,
                        alignment_tolerance: float = 10.0) -> Optional[DetectedTable]:
    """
    Detect if a group of blocks forms a table.
    
    Improved criteria:
    1. At least min_rows rows with similar y-positions
    2. At least min_cols columns with similar x-positions
    3. Consistent grid structure (most cells filled)
    4. Header detection
    """
    if len(blocks) < min_rows * min_cols:
        return None
    
    rows = find_aligned_rows(blocks, alignment_tolerance)
    cols = find_aligned_columns(blocks, alignment_tolerance)
    
    if len(rows) < min_rows or len(cols) < min_cols:
        return None
    
    # Check grid consistency
    items_per_row = [len(row) for row in rows]
    avg_items = sum(items_per_row) / len(items_per_row)
    
    # Allow some variation but not too much
    if max(items_per_row) - min(items_per_row) > max(2, avg_items * 0.5):
        return None
    
    # Detect header row
    header_row_idx = detect_header_row(rows)
    
    # Build cell grid
    cells = []
    col_x_positions = sorted([cols[i][0].x for i in range(len(cols))])
    row_y_positions = sorted([rows[i][0].y for i in range(len(rows))])
    
    for block in blocks:
        row_idx = -1
        col_idx = -1
        
        for i, y_pos in enumerate(row_y_positions):
            if abs(block.y - y_pos) <= alignment_tolerance:
                row_idx = i
                break
        
        for i, x_pos in enumerate(col_x_positions):
            if abs(block.x - x_pos) <= alignment_tolerance:
                col_idx = i
                break
        
        if row_idx >= 0 and col_idx >= 0:
            cells.append(TableCell(
                text=block.text,
                row=row_idx,
                col=col_idx,
                x=block.x,
                y=block.y,
                is_header=(row_idx == header_row_idx)
            ))
    
    # Calculate confidence
    expected_cells = len(rows) * len(cols)
    actual_cells = len(cells)
    fill_rate = actual_cells / expected_cells if expected_cells > 0 else 0
    
    # Bonus confidence for:
    # - Having a header
    # - Consistent column widths
    # - Multiple rows
    confidence = fill_rate
    if header_row_idx >= 0:
        confidence += 0.1
    if len(rows) >= 3:
        confidence += 0.05
    
    confidence = min(confidence, 1.0)
    
    if confidence < 0.5:
        return None
    
    # Calculate bounding box
    all_x = [b.x for b in blocks] + [b.x1 for b in blocks]
    all_y = [b.y for b in blocks] + [b.y1 for b in blocks]
    
    return DetectedTable(
        cells=cells,
        rows=len(rows),
        cols=len(cols),
        x=min(all_x),
        y=min(all_y),
        width=max(all_x) - min(all_x),
        height=max(all_y) - min(all_y),
        confidence=confidence,
        has_header=(header_row_idx >= 0),
        detection_method="heuristic"
    )


# =============================================================================
# ML-BASED TABLE DETECTION (Optional)
# =============================================================================

_TATR_MODEL = None


def get_tatr_model():
    """Get or create the Table Transformer model (singleton)."""
    global _TATR_MODEL
    
    if _TATR_MODEL is not None:
        return _TATR_MODEL
    
    if not is_tatr_available():
        return None
    
    try:
        from transformers import TableTransformerForObjectDetection, AutoImageProcessor
        import torch
        
        logger.info("Loading Table Transformer model...")
        
        model_name = "microsoft/table-transformer-detection"
        _TATR_MODEL = {
            "model": TableTransformerForObjectDetection.from_pretrained(model_name),
            "processor": AutoImageProcessor.from_pretrained(model_name)
        }
        
        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _TATR_MODEL["model"] = _TATR_MODEL["model"].to(device)
        _TATR_MODEL["device"] = device
        
        logger.info(f"Table Transformer loaded on {device}")
        return _TATR_MODEL
        
    except Exception as e:
        logger.error(f"Failed to load Table Transformer: {e}")
        return None


def detect_tables_ml(page_image) -> List[Dict]:
    """
    Detect tables in a page image using Table Transformer.
    
    Args:
        page_image: PIL Image of the page
    
    Returns:
        List of dicts with 'bbox', 'confidence', 'label'
    """
    model_data = get_tatr_model()
    if model_data is None:
        return []
    
    try:
        import torch
        
        model = model_data["model"]
        processor = model_data["processor"]
        device = model_data["device"]
        
        # Prepare image
        inputs = processor(images=page_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Detect
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([page_image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, threshold=0.7, target_sizes=target_sizes
        )[0]
        
        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            tables.append({
                "bbox": box.tolist(),
                "confidence": score.item(),
                "label": model.config.id2label[label.item()]
            })
        
        logger.info(f"TATR detected {len(tables)} tables")
        return tables
        
    except Exception as e:
        logger.warning(f"ML table detection failed: {e}")
        return []


# =============================================================================
# MAIN DETECTION FUNCTION
# =============================================================================

def detect_tables_in_page(
    blocks: List[Dict], 
    page_width: float,
    page_height: float,
    page_image=None,
    use_ml: bool = True,
) -> Tuple[List[DetectedTable], List[Dict]]:
    """
    Detect all tables in a page and separate them from regular text.
    
    Args:
        blocks: List of text block dicts
        page_width: Page width
        page_height: Page height
        page_image: Optional PIL Image for ML detection
        use_ml: Whether to try ML detection first
    
    Returns:
        Tuple of (detected_tables, remaining_blocks)
    """
    if not blocks:
        return [], []
    
    # Convert to TextBlock objects
    text_blocks = []
    for b in blocks:
        text_blocks.append(TextBlock(
            text=b.get('text', ''),
            x=b.get('x', 0),
            y=b.get('y', 0),
            width=b.get('width', 100),
            height=b.get('height', 20),
            font_size=b.get('font_size', 10),
            is_bold=b.get('is_bold', False),
            font_name=b.get('font_name', '')
        ))
    
    detected_tables = []
    used_block_indices = set()
    
    # Try ML detection first if available and requested
    if use_ml and page_image and is_tatr_available():
        ml_tables = detect_tables_ml(page_image)
        
        for ml_table in ml_tables:
            bbox = ml_table["bbox"]
            # Find blocks within this bbox
            table_blocks = []
            for i, tb in enumerate(text_blocks):
                if (bbox[0] <= tb.x <= bbox[2] and 
                    bbox[1] <= tb.y <= bbox[3]):
                    table_blocks.append(tb)
                    used_block_indices.add(i)
            
            if len(table_blocks) >= 4:
                table = detect_table_region(table_blocks)
                if table:
                    table.confidence = ml_table["confidence"]
                    table.detection_method = "ml"
                    table.caption = find_table_caption(text_blocks, table)
                    detected_tables.append(table)
    
    # Heuristic detection for remaining blocks
    remaining_text_blocks = [tb for i, tb in enumerate(text_blocks) 
                            if i not in used_block_indices]
    
    if remaining_text_blocks:
        rows = find_aligned_rows(remaining_text_blocks, tolerance=15.0)
        
        for start_idx in range(len(rows)):
            table_blocks = []
            for row_idx in range(start_idx, min(start_idx + 20, len(rows))):
                row = rows[row_idx]
                if len(row) >= 2:
                    table_blocks.extend(row)
                else:
                    break
            
            if len(table_blocks) >= 4:
                table = detect_table_region(table_blocks)
                if table and table.confidence >= 0.6:
                    table.caption = find_table_caption(text_blocks, table)
                    detected_tables.append(table)
                    
                    for block in table_blocks:
                        for i, tb in enumerate(text_blocks):
                            if tb.x == block.x and tb.y == block.y:
                                used_block_indices.add(i)
    
    # Build remaining blocks list
    remaining = [b for i, b in enumerate(blocks) if i not in used_block_indices]
    
    # Apply table hardening if available
    if TABLE_HARDENING_AVAILABLE and detected_tables:
        detected_tables = [harden_detected_table(t) for t in detected_tables]
    
    logger.info(f"Detected {len(detected_tables)} tables, {len(remaining)} remaining blocks")
    
    return detected_tables, remaining


# =============================================================================
# TABLE HARDENING INTEGRATION
# =============================================================================

def harden_detected_table(table: DetectedTable) -> DetectedTable:
    """
    Apply table hardening to repair artifacts in cell content.
    
    Uses the TableArtifactResolver from table_hardening module.
    """
    if not TABLE_HARDENING_AVAILABLE:
        return table
    
    resolver = TableArtifactResolver()
    
    for cell in table.cells:
        # Create a StitchedCell for the resolver
        stitched = StitchedCell(
            row=cell.row,
            col=cell.col,
            text=cell.text,
            tokens=[],
            is_header=cell.is_header,
            is_numeric=bool(re.match(r'^[\d.,\-−+×]+', cell.text))
        )
        
        # Apply repairs
        resolved = resolver.resolve_cell(stitched)
        cell.text = resolved.text
    
    logger.debug(f"Table hardening: {resolver.fixes_applied} fixes, "
                 f"{resolver.unresolved_count} unresolved")
    
    return table


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    
    print("=== Table Detector Test ===\n")
    
    # Check ML availability
    if is_tatr_available():
        print("[OK] Table Transformer is available")
    else:
        print("[INFO] Table Transformer not available (using heuristics only)")
    
    # Test with sample data
    test_blocks = [
        {'text': 'System', 'x': 50, 'y': 100, 'width': 80, 'height': 15, 'is_bold': True},
        {'text': 'Lifetime', 'x': 150, 'y': 100, 'width': 80, 'height': 15, 'is_bold': True},
        {'text': 'Factor', 'x': 250, 'y': 100, 'width': 80, 'height': 15, 'is_bold': True},
        {'text': 'Ion trap', 'x': 50, 'y': 120, 'width': 80, 'height': 15},
        {'text': '10 ms', 'x': 150, 'y': 120, 'width': 80, 'height': 15},
        {'text': 'Heating', 'x': 250, 'y': 120, 'width': 80, 'height': 15},
        {'text': 'SC qubit', 'x': 50, 'y': 140, 'width': 80, 'height': 15},
        {'text': '100 µs', 'x': 150, 'y': 140, 'width': 80, 'height': 15},
        {'text': 'T1 decay', 'x': 250, 'y': 140, 'width': 80, 'height': 15},
    ]
    
    tables, remaining = detect_tables_in_page(test_blocks, 600, 800, use_ml=False)
    
    print(f"\nFound {len(tables)} tables")
    for table in tables:
        print(f"\nTable: {table.rows}x{table.cols}")
        print(f"Confidence: {table.confidence:.2f}")
        print(f"Has header: {table.has_header}")
        print(f"Method: {table.detection_method}")
        print("\nLaTeX:")
        print(table.to_latex())
        print("\nMarkdown:")
        print(table.to_markdown())
