"""
Table Handler - Special handling for table content in PDFs
© 2025 Sven Kalinowski - Anti-Capitalist Software License v1.4
"""
from __future__ import annotations
import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger("pdf_translator.table")

@dataclass
class TableCell:
    text: str
    row: int
    col: int
    is_header: bool = False
    is_numeric: bool = False

@dataclass  
class Table:
    cells: List[TableCell]
    rows: int
    cols: int

def is_numeric_content(text: str) -> bool:
    """Check if cell contains primarily numeric data."""
    cleaned = re.sub(r'[,.\s%$€£¥±×·]', '', text)
    if not cleaned:
        return False
    numeric_chars = sum(1 for c in cleaned if c.isdigit() or c in '⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉⁻⁺')
    return numeric_chars / len(cleaned) > 0.7

def is_header_row(cells: List[str]) -> bool:
    """Detect if row is likely a header."""
    if not cells:
        return False
    # Headers often have no numbers, or are short
    numeric_count = sum(1 for c in cells if is_numeric_content(c))
    return numeric_count < len(cells) * 0.3

def should_translate_cell(text: str) -> bool:
    """Determine if a table cell should be translated."""
    if not text.strip():
        return False
    if is_numeric_content(text):
        return False
    # Skip pure symbols/units
    if re.match(r'^[\d\s.,±×·%$€£¥°]+$', text):
        return False
    # Skip single characters
    if len(text.strip()) <= 2:
        return False
    return True

def extract_table_structure(blocks: List[Dict]) -> List[Table]:
    """Extract table structures from PDF blocks."""
    tables = []
    # Simple heuristic: blocks with similar y-coordinates and aligned x form tables
    # This is a basic implementation - can be enhanced with ML
    
    # Group by y-coordinate (same row)
    y_groups = {}
    for block in blocks:
        y = round(block.get('y', 0), 0)
        if y not in y_groups:
            y_groups[y] = []
        y_groups[y].append(block)
    
    # Find rows with multiple aligned blocks (potential table rows)
    potential_rows = []
    for y, row_blocks in sorted(y_groups.items()):
        if len(row_blocks) >= 2:
            # Sort by x position
            row_blocks.sort(key=lambda b: b.get('x', 0))
            potential_rows.append((y, row_blocks))
    
    # Group consecutive rows into tables
    if potential_rows:
        current_table_rows = [potential_rows[0]]
        for i in range(1, len(potential_rows)):
            y_prev = potential_rows[i-1][0]
            y_curr = potential_rows[i][0]
            # If rows are close together, same table
            if y_curr - y_prev < 30:  # threshold
                current_table_rows.append(potential_rows[i])
            else:
                # New table
                if len(current_table_rows) >= 2:
                    tables.append(_create_table(current_table_rows))
                current_table_rows = [potential_rows[i]]
        
        if len(current_table_rows) >= 2:
            tables.append(_create_table(current_table_rows))
    
    return tables

def _create_table(rows: List[Tuple[float, List[Dict]]]) -> Table:
    """Create Table object from row data."""
    cells = []
    for row_idx, (y, blocks) in enumerate(rows):
        is_header = row_idx == 0 and is_header_row([b.get('text', '') for b in blocks])
        for col_idx, block in enumerate(blocks):
            text = block.get('text', '')
            cells.append(TableCell(
                text=text,
                row=row_idx,
                col=col_idx,
                is_header=is_header,
                is_numeric=is_numeric_content(text)
            ))
    
    return Table(
        cells=cells,
        rows=len(rows),
        cols=max(len(r[1]) for r in rows) if rows else 0
    )

def translate_table(table: Table, translate_func) -> Table:
    """Translate table content while preserving structure."""
    translated_cells = []
    
    for cell in table.cells:
        if should_translate_cell(cell.text) and not cell.is_numeric:
            translated_text = translate_func(cell.text)
        else:
            translated_text = cell.text
        
        translated_cells.append(TableCell(
            text=translated_text,
            row=cell.row,
            col=cell.col,
            is_header=cell.is_header,
            is_numeric=cell.is_numeric
        ))
    
    return Table(cells=translated_cells, rows=table.rows, cols=table.cols)

def protect_table_numbers(text: str) -> Tuple[str, Dict[str, str]]:
    """Protect numeric values in table-like content."""
    protected = {}
    counter = [0]
    
    def replacer(match):
        key = f"⟦NUM_{counter[0]}⟧"
        protected[key] = match.group(0)
        counter[0] += 1
        return key
    
    # Protect numbers with units, percentages, currency
    patterns = [
        r'\d+[.,]\d+\s*%',           # 12.5%
        r'[€$£¥]\s*\d+[.,]?\d*',     # $100.00
        r'\d+[.,]?\d*\s*[€$£¥]',     # 100€
        r'[±]?\d+[.,]\d+',            # ±1.5
        r'\d+\s*×\s*10[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+',  # 1.5 × 10⁻³
    ]
    
    result = text
    for pattern in patterns:
        result = re.sub(pattern, replacer, result)
    
    return result, protected

def restore_table_numbers(text: str, protected: Dict[str, str]) -> str:
    """Restore protected numeric values."""
    result = text
    for key, value in protected.items():
        result = result.replace(key, value)
    return result
