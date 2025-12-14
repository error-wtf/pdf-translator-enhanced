"""
Table Detector - Detect and reconstruct tables from PDF text blocks

Detects table regions by analyzing aligned text boxes and reconstructs
them as LaTeX tabular environments.

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger("pdf_translator.table_detector")


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
    
    def to_latex(self) -> str:
        """Convert table to LaTeX tabular environment."""
        if not self.cells:
            return ""
        
        # Build grid
        grid = [['' for _ in range(self.cols)] for _ in range(self.rows)]
        for cell in self.cells:
            if 0 <= cell.row < self.rows and 0 <= cell.col < self.cols:
                grid[cell.row][cell.col] = cell.text.strip()
        
        # Generate LaTeX
        col_spec = '|' + 'l|' * self.cols
        lines = [f"\\begin{{tabular}}{{{col_spec}}}"]
        lines.append("\\hline")
        
        for row_idx, row in enumerate(grid):
            # Escape special LaTeX characters
            escaped_row = [self._escape_latex(cell) for cell in row]
            lines.append(' & '.join(escaped_row) + ' \\\\')
            lines.append("\\hline")
        
        lines.append("\\end{tabular}")
        
        # Add caption if present
        if self.caption:
            latex = f"\\begin{{table}}[H]\n\\centering\n"
            latex += '\n'.join(lines)
            latex += f"\n\\caption{{{self._escape_latex(self.caption)}}}"
            latex += "\n\\end{table}"
            return latex
        
        return '\n'.join(lines)
    
    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters."""
        replacements = [
            ('\\', '\\textbackslash{}'),
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
            if old != '\\':  # Handle backslash separately
                text = text.replace(old, new)
        return text


def find_aligned_columns(blocks: List[TextBlock], tolerance: float = 5.0) -> List[List[TextBlock]]:
    """
    Group blocks by their x-position (column alignment).
    
    Returns list of column groups, each containing blocks in that column.
    """
    if not blocks:
        return []
    
    # Sort by x position
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


def find_aligned_rows(blocks: List[TextBlock], tolerance: float = 5.0) -> List[List[TextBlock]]:
    """
    Group blocks by their y-position (row alignment).
    
    Returns list of row groups, each containing blocks in that row.
    """
    if not blocks:
        return []
    
    # Sort by y position
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


def detect_table_region(blocks: List[TextBlock], 
                        min_rows: int = 2, 
                        min_cols: int = 2,
                        alignment_tolerance: float = 10.0) -> Optional[DetectedTable]:
    """
    Detect if a group of blocks forms a table.
    
    Criteria for table detection:
    1. At least min_rows rows with similar y-positions
    2. At least min_cols columns with similar x-positions
    3. Consistent grid structure (most cells filled)
    
    Returns DetectedTable if found, None otherwise.
    """
    if len(blocks) < min_rows * min_cols:
        return None
    
    # Find row and column alignments
    rows = find_aligned_rows(blocks, alignment_tolerance)
    cols = find_aligned_columns(blocks, alignment_tolerance)
    
    if len(rows) < min_rows or len(cols) < min_cols:
        return None
    
    # Check grid consistency
    # A good table should have similar number of items per row
    items_per_row = [len(row) for row in rows]
    if max(items_per_row) - min(items_per_row) > 2:
        # Too much variation - probably not a table
        return None
    
    # Build cell grid
    cells = []
    col_x_positions = sorted([cols[i][0].x for i in range(len(cols))])
    row_y_positions = sorted([rows[i][0].y for i in range(len(rows))])
    
    for block in blocks:
        # Find which row and column this block belongs to
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
                y=block.y
            ))
    
    # Calculate confidence based on grid fill rate
    expected_cells = len(rows) * len(cols)
    actual_cells = len(cells)
    confidence = actual_cells / expected_cells if expected_cells > 0 else 0
    
    if confidence < 0.5:
        # Less than 50% of grid filled - probably not a table
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
        confidence=confidence
    )


def is_table_header(text: str) -> bool:
    """Check if text looks like a table header/caption."""
    text = text.strip().lower()
    return bool(re.match(r'^(table|tab\.?)\s*\d+', text))


def find_table_caption(blocks: List[TextBlock], table: DetectedTable, 
                       max_distance: float = 50.0) -> Optional[str]:
    """
    Find caption for a detected table.
    
    Looks for "Table X:" pattern above or below the table.
    """
    for block in blocks:
        if is_table_header(block.text):
            # Check if it's close to the table
            if block.y < table.y and table.y - block.y1 < max_distance:
                # Caption above table
                return block.text
            elif block.y > table.y + table.height and block.y - (table.y + table.height) < max_distance:
                # Caption below table
                return block.text
    
    return None


def detect_tables_in_page(blocks: List[Dict], 
                          page_width: float,
                          page_height: float) -> Tuple[List[DetectedTable], List[Dict]]:
    """
    Detect all tables in a page and separate them from regular text.
    
    Args:
        blocks: List of text block dicts with 'text', 'x', 'y', 'width', 'height'
        page_width: Page width
        page_height: Page height
    
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
            is_bold=b.get('is_bold', False)
        ))
    
    detected_tables = []
    used_blocks = set()
    
    # Try to find table regions
    # Strategy: Look for clusters of aligned blocks
    
    # Group blocks by vertical regions (potential table areas)
    rows = find_aligned_rows(text_blocks, tolerance=15.0)
    
    # Look for consecutive rows that might form a table
    for start_idx in range(len(rows)):
        if start_idx in used_blocks:
            continue
        
        # Collect consecutive rows
        table_blocks = []
        for row_idx in range(start_idx, min(start_idx + 20, len(rows))):
            row = rows[row_idx]
            # Check if this row has multiple columns (table-like)
            if len(row) >= 2:
                table_blocks.extend(row)
            else:
                break
        
        if len(table_blocks) >= 4:  # At least 2x2
            table = detect_table_region(table_blocks)
            if table and table.confidence >= 0.6:
                # Find caption
                table.caption = find_table_caption(text_blocks, table)
                detected_tables.append(table)
                
                # Mark blocks as used
                for block in table_blocks:
                    for i, tb in enumerate(text_blocks):
                        if tb.x == block.x and tb.y == block.y:
                            used_blocks.add(i)
    
    # Build remaining blocks list
    remaining = []
    for i, block in enumerate(blocks):
        if i not in used_blocks:
            remaining.append(block)
    
    logger.info(f"Detected {len(detected_tables)} tables, {len(remaining)} remaining blocks")
    
    return detected_tables, remaining


# Test
if __name__ == "__main__":
    # Test with sample data
    test_blocks = [
        {'text': 'System', 'x': 50, 'y': 100, 'width': 80, 'height': 15},
        {'text': 'Lifetime', 'x': 150, 'y': 100, 'width': 80, 'height': 15},
        {'text': 'Factor', 'x': 250, 'y': 100, 'width': 80, 'height': 15},
        {'text': 'Ion trap', 'x': 50, 'y': 120, 'width': 80, 'height': 15},
        {'text': '10 ms', 'x': 150, 'y': 120, 'width': 80, 'height': 15},
        {'text': 'Heating', 'x': 250, 'y': 120, 'width': 80, 'height': 15},
        {'text': 'SC qubit', 'x': 50, 'y': 140, 'width': 80, 'height': 15},
        {'text': '100 µs', 'x': 150, 'y': 140, 'width': 80, 'height': 15},
        {'text': 'T1 decay', 'x': 250, 'y': 140, 'width': 80, 'height': 15},
    ]
    
    tables, remaining = detect_tables_in_page(test_blocks, 600, 800)
    
    print(f"Found {len(tables)} tables")
    for table in tables:
        print(f"\nTable: {table.rows}x{table.cols}, confidence={table.confidence:.2f}")
        print(table.to_latex())
