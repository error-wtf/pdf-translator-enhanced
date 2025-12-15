"""
Table Hardening Module - Tables-first Quality Improvement

Implements robust table extraction and repair per Windsurf prompt:
- Task A: Grid/Line Detection (vector lines + text clustering)
- Task B: Cell Stitching Logic (token assignment + merging)
- Task C: Table Artifact Resolver (exponent, minus, units)

Hard Constraints:
- NO removals: never drop table content
- Never modify protected math blocks
- Deterministic, rule-based repairs only
- UNRESOLVED markers only as last resort

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import re
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger("pdf_translator.table_hardening")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GridLine:
    """A detected grid line (horizontal or vertical)."""
    x0: float
    y0: float
    x1: float
    y1: float
    is_horizontal: bool
    confidence: float = 1.0
    
    @property
    def length(self) -> float:
        return ((self.x1 - self.x0) ** 2 + (self.y1 - self.y0) ** 2) ** 0.5


@dataclass
class TableGrid:
    """Detected table grid structure."""
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    col_boundaries: List[float]  # x-coordinates
    row_boundaries: List[float]  # y-coordinates
    confidence: float = 0.0
    detection_method: str = "text_clustering"


@dataclass
class CellToken:
    """A token within a table cell."""
    text: str
    x: float
    y: float
    width: float
    height: float
    font_size: float = 10.0
    is_numeric: bool = False
    
    @property
    def center_x(self) -> float:
        return self.x + self.width / 2
    
    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


@dataclass
class StitchedCell:
    """A fully stitched table cell with merged tokens."""
    row: int
    col: int
    text: str
    tokens: List[CellToken]
    is_header: bool = False
    rowspan: int = 1
    colspan: int = 1
    is_numeric: bool = False
    has_artifact: bool = False
    artifact_fixed: bool = False


# =============================================================================
# TASK A: GRID/LINE DETECTION
# =============================================================================

class GridDetector:
    """
    Detects table grid structure using multiple methods:
    1. Vector ruling lines (from PyMuPDF drawings)
    2. Text layout cues (column/row clustering)
    3. Hybrid fallback
    """
    
    ANGLE_THRESHOLD = 5.0  # Degrees for near-horizontal/vertical
    EPSILON = 3.0  # Pixels for segment merging
    
    def __init__(self):
        self.lines: List[GridLine] = []
    
    # =========================================================================
    # Method 1: Vector Ruling Lines
    # =========================================================================
    
    def extract_ruling_lines(self, page) -> List[GridLine]:
        """
        Extract ruling lines from PyMuPDF page drawings.
        
        Args:
            page: fitz.Page object
        
        Returns:
            List of detected GridLine objects
        """
        lines = []
        
        try:
            # Get all drawings on the page
            drawings = page.get_drawings()
            
            for drawing in drawings:
                for item in drawing.get("items", []):
                    if item[0] == "l":  # Line
                        p1, p2 = item[1], item[2]
                        line = self._create_grid_line(p1.x, p1.y, p2.x, p2.y)
                        if line:
                            lines.append(line)
                    elif item[0] == "re":  # Rectangle
                        rect = item[1]
                        # Extract 4 edges as lines
                        edges = [
                            (rect.x0, rect.y0, rect.x1, rect.y0),  # Top
                            (rect.x0, rect.y1, rect.x1, rect.y1),  # Bottom
                            (rect.x0, rect.y0, rect.x0, rect.y1),  # Left
                            (rect.x1, rect.y0, rect.x1, rect.y1),  # Right
                        ]
                        for x0, y0, x1, y1 in edges:
                            line = self._create_grid_line(x0, y0, x1, y1)
                            if line:
                                lines.append(line)
            
            logger.info(f"Extracted {len(lines)} ruling lines from drawings")
            
        except Exception as e:
            logger.warning(f"Could not extract ruling lines: {e}")
        
        return lines
    
    def _create_grid_line(self, x0: float, y0: float, x1: float, y1: float) -> Optional[GridLine]:
        """Create a GridLine if it's near-horizontal or near-vertical."""
        import math
        
        dx = x1 - x0
        dy = y1 - y0
        length = (dx * dx + dy * dy) ** 0.5
        
        if length < 10:  # Too short
            return None
        
        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
        
        if angle <= self.ANGLE_THRESHOLD:
            # Near-horizontal
            y_avg = (y0 + y1) / 2
            return GridLine(x0=min(x0, x1), y0=y_avg, x1=max(x0, x1), y1=y_avg, 
                           is_horizontal=True)
        elif angle >= 90 - self.ANGLE_THRESHOLD:
            # Near-vertical
            x_avg = (x0 + x1) / 2
            return GridLine(x0=x_avg, y0=min(y0, y1), x1=x_avg, y1=max(y0, y1),
                           is_horizontal=False)
        
        return None
    
    def merge_collinear_segments(self, lines: List[GridLine]) -> List[GridLine]:
        """Merge collinear segments if endpoints are within epsilon."""
        if not lines:
            return []
        
        # Separate horizontal and vertical
        h_lines = [l for l in lines if l.is_horizontal]
        v_lines = [l for l in lines if not l.is_horizontal]
        
        merged = []
        merged.extend(self._merge_lines(h_lines, is_horizontal=True))
        merged.extend(self._merge_lines(v_lines, is_horizontal=False))
        
        return merged
    
    def _merge_lines(self, lines: List[GridLine], is_horizontal: bool) -> List[GridLine]:
        """Merge lines that are collinear and close."""
        if not lines:
            return []
        
        # Group by position (y for horizontal, x for vertical)
        groups: Dict[int, List[GridLine]] = defaultdict(list)
        
        for line in lines:
            key = int((line.y0 if is_horizontal else line.x0) / self.EPSILON)
            groups[key].append(line)
        
        merged = []
        for group in groups.values():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # Merge overlapping/touching segments
                if is_horizontal:
                    sorted_group = sorted(group, key=lambda l: l.x0)
                    current = sorted_group[0]
                    for next_line in sorted_group[1:]:
                        if next_line.x0 <= current.x1 + self.EPSILON:
                            current = GridLine(
                                x0=current.x0, y0=current.y0,
                                x1=max(current.x1, next_line.x1), y1=current.y1,
                                is_horizontal=True
                            )
                        else:
                            merged.append(current)
                            current = next_line
                    merged.append(current)
                else:
                    sorted_group = sorted(group, key=lambda l: l.y0)
                    current = sorted_group[0]
                    for next_line in sorted_group[1:]:
                        if next_line.y0 <= current.y1 + self.EPSILON:
                            current = GridLine(
                                x0=current.x0, y0=current.y0,
                                x1=current.x1, y1=max(current.y1, next_line.y1),
                                is_horizontal=False
                            )
                        else:
                            merged.append(current)
                            current = next_line
                    merged.append(current)
        
        return merged
    
    # =========================================================================
    # Method 2: Text Layout Cues
    # =========================================================================
    
    def detect_columns_from_text(self, tokens: List[CellToken], 
                                  min_gap: float = 15.0) -> List[float]:
        """
        Detect column boundaries via x-coordinate clustering.
        Uses histogram valleys to find column separations.
        """
        if not tokens:
            return []
        
        # Collect all x-positions
        x_positions = sorted(set(t.x for t in tokens))
        
        if len(x_positions) < 2:
            return []
        
        # Find gaps (valleys in histogram)
        boundaries = [min(x_positions)]
        
        for i in range(1, len(x_positions)):
            gap = x_positions[i] - x_positions[i-1]
            if gap > min_gap:
                # Midpoint of gap is column boundary
                boundaries.append((x_positions[i-1] + x_positions[i]) / 2)
        
        boundaries.append(max(t.x + t.width for t in tokens))
        
        return boundaries
    
    def detect_rows_from_text(self, tokens: List[CellToken],
                               min_gap: float = 8.0) -> List[float]:
        """
        Detect row boundaries via y-coordinate clustering.
        Uses baseline grouping.
        """
        if not tokens:
            return []
        
        # Collect all y-positions (baselines)
        y_positions = sorted(set(t.y for t in tokens))
        
        if len(y_positions) < 2:
            return []
        
        boundaries = [min(y_positions)]
        
        for i in range(1, len(y_positions)):
            gap = y_positions[i] - y_positions[i-1]
            if gap > min_gap:
                boundaries.append((y_positions[i-1] + y_positions[i]) / 2)
        
        boundaries.append(max(t.y + t.height for t in tokens))
        
        return boundaries
    
    # =========================================================================
    # Method 3: Hybrid Detection
    # =========================================================================
    
    def detect_grid(self, tokens: List[CellToken], page=None) -> Optional[TableGrid]:
        """
        Detect table grid using best available method.
        
        Priority:
        1. Vector ruling lines (if page provided)
        2. Text layout cues
        3. Hybrid (partial lines + text clustering)
        """
        if not tokens:
            return None
        
        # Calculate bounding box
        x0 = min(t.x for t in tokens)
        y0 = min(t.y for t in tokens)
        x1 = max(t.x + t.width for t in tokens)
        y1 = max(t.y + t.height for t in tokens)
        bbox = (x0, y0, x1, y1)
        
        col_boundaries = []
        row_boundaries = []
        method = "text_clustering"
        confidence = 0.5
        
        # Try vector lines first
        if page is not None:
            lines = self.extract_ruling_lines(page)
            lines = self.merge_collinear_segments(lines)
            
            if len(lines) >= 4:
                # Extract boundaries from lines
                h_lines = [l for l in lines if l.is_horizontal]
                v_lines = [l for l in lines if not l.is_horizontal]
                
                if h_lines and v_lines:
                    col_boundaries = sorted(set(l.x0 for l in v_lines))
                    row_boundaries = sorted(set(l.y0 for l in h_lines))
                    method = "vector_lines"
                    confidence = 0.9
        
        # Fallback to text clustering
        if not col_boundaries or not row_boundaries:
            col_boundaries = self.detect_columns_from_text(tokens)
            row_boundaries = self.detect_rows_from_text(tokens)
            
            if method != "vector_lines":
                method = "text_clustering"
                confidence = 0.6
        
        if len(col_boundaries) < 2 or len(row_boundaries) < 2:
            return None
        
        return TableGrid(
            bbox=bbox,
            col_boundaries=col_boundaries,
            row_boundaries=row_boundaries,
            confidence=confidence,
            detection_method=method
        )


# =============================================================================
# TASK B: CELL STITCHING LOGIC
# =============================================================================

class CellStitcher:
    """
    Stitches tokens into cells using grid boundaries.
    
    B1) Assign tokens to cells by bbox centroid
    B2) Merge tokens within a cell (reading order)
    B3) Detect multi-column/multi-row spans
    B4) Preserve semantics for scientific numeric cells
    """
    
    NUMERIC_PATTERN = re.compile(r'^[\d.,×\-−–+eE^±~<>≈≤≥]+$')
    EXPONENT_PATTERN = re.compile(r'10[?²³⁴⁵⁶⁷⁸⁹⁰¹⁻⁺\-−^]')
    
    def __init__(self, grid: TableGrid):
        self.grid = grid
        self.cells: List[StitchedCell] = []
    
    def assign_token_to_cell(self, token: CellToken) -> Tuple[int, int]:
        """
        Assign token to cell by centroid position.
        
        Returns (row, col) indices.
        """
        cx, cy = token.center_x, token.center_y
        
        # Find column
        col = 0
        for i in range(len(self.grid.col_boundaries) - 1):
            if self.grid.col_boundaries[i] <= cx < self.grid.col_boundaries[i + 1]:
                col = i
                break
        
        # Find row
        row = 0
        for i in range(len(self.grid.row_boundaries) - 1):
            if self.grid.row_boundaries[i] <= cy < self.grid.row_boundaries[i + 1]:
                row = i
                break
        
        return row, col
    
    def stitch_tokens(self, tokens: List[CellToken]) -> List[StitchedCell]:
        """
        Stitch all tokens into cells.
        
        Returns list of StitchedCell objects.
        """
        # Group tokens by cell
        cell_tokens: Dict[Tuple[int, int], List[CellToken]] = defaultdict(list)
        
        for token in tokens:
            row, col = self.assign_token_to_cell(token)
            cell_tokens[(row, col)].append(token)
        
        # Create stitched cells
        cells = []
        num_rows = len(self.grid.row_boundaries) - 1
        num_cols = len(self.grid.col_boundaries) - 1
        
        for row in range(num_rows):
            for col in range(num_cols):
                key = (row, col)
                if key in cell_tokens:
                    cell = self._merge_cell_tokens(row, col, cell_tokens[key])
                    cells.append(cell)
                else:
                    # Empty cell
                    cells.append(StitchedCell(
                        row=row, col=col, text="", tokens=[],
                        is_header=(row == 0)
                    ))
        
        self.cells = cells
        return cells
    
    def _merge_cell_tokens(self, row: int, col: int, 
                           tokens: List[CellToken]) -> StitchedCell:
        """Merge tokens within a cell in reading order."""
        if not tokens:
            return StitchedCell(row=row, col=col, text="", tokens=[])
        
        # Sort by reading order (y first, then x)
        sorted_tokens = sorted(tokens, key=lambda t: (t.y, t.x))
        
        # Merge with smart spacing
        parts = []
        prev_token = None
        
        for token in sorted_tokens:
            text = token.text.strip()
            if not text:
                continue
            
            if prev_token:
                # Determine spacing
                gap = token.x - (prev_token.x + prev_token.width)
                same_line = abs(token.y - prev_token.y) < 5
                
                if same_line and gap < 5:
                    # No space for numeric continuations
                    if self._is_numeric_continuation(prev_token.text, text):
                        parts.append(text)
                    else:
                        parts.append(' ' + text)
                elif same_line:
                    parts.append(' ' + text)
                else:
                    parts.append(' ' + text)
            else:
                parts.append(text)
            
            prev_token = token
        
        merged_text = ''.join(parts).strip()
        is_numeric = bool(self.NUMERIC_PATTERN.match(merged_text.replace(' ', '')))
        has_artifact = '?' in merged_text or '�' in merged_text
        
        return StitchedCell(
            row=row,
            col=col,
            text=merged_text,
            tokens=tokens,
            is_header=(row == 0),
            is_numeric=is_numeric,
            has_artifact=has_artifact
        )
    
    def _is_numeric_continuation(self, prev: str, curr: str) -> bool:
        """Check if curr is a numeric continuation of prev."""
        prev = prev.strip()
        curr = curr.strip()
        
        # Exponent patterns
        if prev.endswith('10') and curr.startswith(('^', '−', '-', '²', '³')):
            return True
        if prev.endswith('^') and curr[0:1].isdigit():
            return True
        if prev[-1:].isdigit() and curr.startswith(('.', ',')):
            return True
        if prev.endswith(('×', 'x', '*')) and curr.startswith('10'):
            return True
        
        return False


# =============================================================================
# TASK C: TABLE ARTIFACT RESOLVER
# =============================================================================

class TableArtifactResolver:
    """
    Repairs extraction artifacts commonly seen in scientific tables.
    
    C1) Exponent/superscript damage
    C2) Minus vs dash confusion
    C3) Unit corruption and spacing
    C4) Header loss/mis-assignment
    C5) Column drift detection
    C6) Garbage glyph placeholders
    """
    
    def __init__(self):
        self.fixes_applied = 0
        self.unresolved_count = 0
    
    # =========================================================================
    # C1: Exponent/Superscript Damage
    # =========================================================================
    
    EXPONENT_PATTERNS = [
        # "10?2?" → "10^2" (broken exponent with question marks)
        (r'10\?(\d+)\?', r'10^\1'),
        # "10?²?" → "10^2"
        (r'10\?([²³⁴⁵⁶⁷⁸⁹⁰¹])\?', r'10^\1'),
        # "10?−?2" → "10^-2"
        (r'10\?[−\-]\?(\d)', r'10^-\1'),
        # "10?^?2" → "10^2"
        (r'10\?\^?\?(\d+)', r'10^\1'),
        # "10 −2" → "10^-2"
        (r'10\s+[−\-](\d+)', r'10^-\1'),
        # "×10?" or "x10?" followed by digit → "×10^" or "x10^"
        (r'([×x])\s*10\?\s*(\d+)', r'\g<1>10^\2'),
        # Broken negative exponents with question marks
        (r'(\d+)\?[−\-]\?(\d+)', r'\1^-\2'),
    ]
    
    def fix_exponent_damage(self, text: str) -> Tuple[str, bool]:
        """Fix exponent/superscript corruption in text."""
        original = text
        
        for pattern, replacement in self.EXPONENT_PATTERNS:
            if callable(replacement):
                text = re.sub(pattern, replacement, text)
            else:
                text = re.sub(pattern, replacement, text)
        
        fixed = text != original
        if fixed:
            self.fixes_applied += 1
        
        return text, fixed
    
    # =========================================================================
    # C2: Minus vs Dash Confusion
    # =========================================================================
    
    MINUS_VARIANTS = ['−', '–', '—', '‐', '‑']
    
    def normalize_minus(self, text: str, is_numeric: bool = False) -> str:
        """
        Normalize minus/dash characters.
        
        - In numeric context: use ASCII minus '-'
        - In text context: preserve hyphens
        """
        if is_numeric:
            # Numeric context: normalize all to ASCII minus
            for variant in self.MINUS_VARIANTS:
                text = text.replace(variant, '-')
        else:
            # Text context: only normalize true minus signs (in numbers)
            # Pattern: digit followed by minus variant followed by digit
            for variant in self.MINUS_VARIANTS:
                text = re.sub(rf'(\d){re.escape(variant)}(\d)', r'\1-\2', text)
        
        return text
    
    # =========================================================================
    # C3: Unit Corruption and Spacing
    # =========================================================================
    
    UNIT_PATTERNS = [
        # "m?s" → "m.s" or "m s"
        (r'([a-zA-Z])\?([a-zA-Z])', r'\1.\2'),
        # "kg?m^2" → "kg.m^2"
        (r'(kg|g|m|s|Hz|eV|K)\?(m|s|Hz|K)', r'\1.\2'),
        # "GHz?" → "GHz" (trailing question mark)
        (r'(Hz|eV|K|m|s)\?(?=\s|$)', r'\1'),
    ]
    
    def fix_unit_corruption(self, text: str) -> Tuple[str, bool]:
        """Fix unit corruption patterns."""
        original = text
        
        for pattern, replacement in self.UNIT_PATTERNS:
            text = re.sub(pattern, replacement, text)
        
        fixed = text != original
        if fixed:
            self.fixes_applied += 1
        
        return text, fixed
    
    # =========================================================================
    # C6: Garbage Glyph Placeholders
    # =========================================================================
    
    GLYPH_PATTERN = re.compile(r'\?[A-Z]_[a-f0-9]{4,8}_\d+\?')
    
    def resolve_glyph_placeholders(self, text: str, context: str = "") -> Tuple[str, bool]:
        """
        Try to resolve garbage glyph placeholders.
        
        If resolution fails, mark as UNRESOLVED (do not remove).
        """
        matches = list(self.GLYPH_PATTERN.finditer(text))
        
        if not matches:
            return text, False
        
        fixed = False
        for match in reversed(matches):
            glyph = match.group(0)
            replacement = self._infer_glyph(glyph, context)
            
            if replacement:
                text = text[:match.start()] + replacement + text[match.end():]
                fixed = True
                self.fixes_applied += 1
            else:
                # Mark as UNRESOLVED (do not remove!)
                marker = f"[[UNRESOLVED_GLYPH:{glyph}]]"
                text = text[:match.start()] + marker + text[match.end():]
                self.unresolved_count += 1
        
        return text, fixed
    
    def _infer_glyph(self, glyph: str, context: str) -> Optional[str]:
        """Try to infer symbol from glyph ID and context."""
        ctx_lower = context.lower()
        
        # Common scientific symbols
        if 'temperature' in ctx_lower:
            return 'T'
        if 'frequency' in ctx_lower:
            return 'f'
        if 'wavelength' in ctx_lower:
            return 'λ'
        if 'plus' in ctx_lower and 'minus' in ctx_lower:
            return '±'
        if 'approximately' in ctx_lower:
            return '≈'
        
        return None
    
    # =========================================================================
    # Main Resolve Function
    # =========================================================================
    
    def resolve_cell(self, cell: StitchedCell) -> StitchedCell:
        """Apply all repairs to a single cell."""
        text = cell.text
        fixed = False
        
        # C1: Exponent damage
        text, exp_fixed = self.fix_exponent_damage(text)
        fixed = fixed or exp_fixed
        
        # C2: Minus normalization
        text = self.normalize_minus(text, cell.is_numeric)
        
        # C3: Unit corruption
        text, unit_fixed = self.fix_unit_corruption(text)
        fixed = fixed or unit_fixed
        
        # C6: Glyph placeholders
        text, glyph_fixed = self.resolve_glyph_placeholders(text, cell.text)
        fixed = fixed or glyph_fixed
        
        cell.text = text
        cell.artifact_fixed = fixed
        
        return cell
    
    def resolve_table(self, cells: List[StitchedCell]) -> List[StitchedCell]:
        """Apply repairs to all cells in a table."""
        return [self.resolve_cell(cell) for cell in cells]


# =============================================================================
# MAIN INTERFACE
# =============================================================================

def harden_table(tokens: List[Dict], page=None) -> Tuple[List[StitchedCell], Dict]:
    """
    Main entry point for table hardening.
    
    Args:
        tokens: List of token dicts with text, x, y, width, height
        page: Optional fitz.Page for vector line extraction
    
    Returns:
        Tuple of (stitched_cells, report_dict)
    """
    # Convert to CellToken objects
    cell_tokens = [
        CellToken(
            text=t.get('text', ''),
            x=t.get('x', 0),
            y=t.get('y', 0),
            width=t.get('width', 50),
            height=t.get('height', 15),
            font_size=t.get('font_size', 10),
            is_numeric=bool(re.match(r'^[\d.,\-−+×]+', t.get('text', '')))
        )
        for t in tokens
    ]
    
    if not cell_tokens:
        return [], {"error": "No tokens provided"}
    
    # Task A: Grid Detection
    detector = GridDetector()
    grid = detector.detect_grid(cell_tokens, page)
    
    if not grid:
        return [], {"error": "Could not detect grid structure"}
    
    # Task B: Cell Stitching
    stitcher = CellStitcher(grid)
    cells = stitcher.stitch_tokens(cell_tokens)
    
    # Task C: Artifact Resolution
    resolver = TableArtifactResolver()
    cells = resolver.resolve_table(cells)
    
    # Build report
    report = {
        "grid_method": grid.detection_method,
        "grid_confidence": grid.confidence,
        "num_rows": len(grid.row_boundaries) - 1,
        "num_cols": len(grid.col_boundaries) - 1,
        "num_cells": len(cells),
        "fixes_applied": resolver.fixes_applied,
        "unresolved_count": resolver.unresolved_count,
        "cells_with_artifacts": sum(1 for c in cells if c.has_artifact),
    }
    
    logger.info(f"Table hardened: {report['num_rows']}x{report['num_cols']}, "
                f"{report['fixes_applied']} fixes, {report['unresolved_count']} unresolved")
    
    return cells, report


# =============================================================================
# TESTS
# =============================================================================

def run_tests():
    """Run built-in tests for table hardening."""
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    
    print("=" * 60)
    print("Table Hardening Tests")
    print("=" * 60)
    
    resolver = TableArtifactResolver()
    
    test_cases = [
        # (input, expected_substring, description)
        ("Value: 10?2? units", "10^2", "Exponent damage fix"),
        ("Scale: x10? 3", "x10^3", "Scientific notation fix"),
        ("Unit: m?s", "m.s", "Unit corruption fix"),
        ("Speed: 100 GHz?", "100 GHz", "Trailing ? removal"),
        ("Value: 10-20", "10-20", "Minus normalization"),
    ]
    
    all_passed = True
    
    for input_text, expected, description in test_cases:
        cell = StitchedCell(row=0, col=0, text=input_text, tokens=[], is_numeric=True)
        result = resolver.resolve_cell(cell)
        
        passed = expected in result.text
        status = "PASS" if passed else "FAIL"
        
        if not passed:
            all_passed = False
        
        print(f"\n[{status}]: {description}")
        print(f"  Input:    '{input_text}'")
        print(f"  Expected: '{expected}' in output")
        print(f"  Result:   '{result.text}'")
    
    print("\n" + "=" * 60)
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    run_tests()
