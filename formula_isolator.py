"""
Formula Isolator - 100% LaTeX Formula Preservation

This module ensures ZERO formula corruption during translation by:
1. Extracting ALL LaTeX content before translation
2. Replacing with unique hash-based placeholders
3. Translating only the prose
4. Restoring formulas with verification

The key insight: LLMs can corrupt formulas even when told not to.
The ONLY safe approach is to remove them completely before translation.

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import re
import hashlib
import logging
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger("pdf_translator.formula_isolator")


# =============================================================================
# PLACEHOLDER GENERATION
# =============================================================================

def generate_placeholder(content: str, index: int, prefix: str = "FORMULA") -> str:
    """
    Generate a unique placeholder that cannot be corrupted by LLM.
    
    Uses hash to make placeholder unique and unambiguous.
    Format: ⟦FORMULA_a1b2c3_0⟧
    
    The ⟦⟧ brackets are chosen because:
    - Extremely rare in natural text
    - Not LaTeX syntax
    - Visually distinct
    - UTF-8 safe
    """
    # Short hash of content for uniqueness
    content_hash = hashlib.md5(content.encode()).hexdigest()[:6]
    return f"⟦{prefix}_{content_hash}_{index}⟧"


def is_placeholder(text: str) -> bool:
    """Check if text looks like a placeholder."""
    return bool(re.match(r'⟦[A-Z]+_[a-f0-9]+_\d+⟧', text))


# =============================================================================
# FORMULA PATTERNS
# =============================================================================

# Ordered by specificity (most specific first to avoid partial matches)
FORMULA_PATTERNS = [
    # Display math environments (must come before simpler patterns)
    (r'\\begin\{equation\*?\}.*?\\end\{equation\*?\}', 'equation'),
    (r'\\begin\{align\*?\}.*?\\end\{align\*?\}', 'align'),
    (r'\\begin\{gather\*?\}.*?\\end\{gather\*?\}', 'gather'),
    (r'\\begin\{multline\*?\}.*?\\end\{multline\*?\}', 'multline'),
    (r'\\begin\{eqnarray\*?\}.*?\\end\{eqnarray\*?\}', 'eqnarray'),
    (r'\\begin\{flalign\*?\}.*?\\end\{flalign\*?\}', 'flalign'),
    (r'\\begin\{split\}.*?\\end\{split\}', 'split'),
    (r'\\begin\{cases\}.*?\\end\{cases\}', 'cases'),
    (r'\\begin\{array\}.*?\\end\{array\}', 'array'),
    (r'\\begin\{matrix\}.*?\\end\{matrix\}', 'matrix'),
    (r'\\begin\{pmatrix\}.*?\\end\{pmatrix\}', 'pmatrix'),
    (r'\\begin\{bmatrix\}.*?\\end\{bmatrix\}', 'bmatrix'),
    (r'\\begin\{vmatrix\}.*?\\end\{vmatrix\}', 'vmatrix'),
    
    # Display math delimiters
    (r'\$\$[^$]+\$\$', 'display_dollar'),
    (r'\\\[.*?\\\]', 'display_bracket'),
    
    # Inline math (must come after display to avoid partial matches)
    (r'(?<!\$)\$(?!\$)[^$]+(?<!\$)\$(?!\$)', 'inline_math'),
    (r'\\\(.*?\\\)', 'inline_paren'),
]

# Commands that should be extracted but aren't math
COMMAND_PATTERNS = [
    # Citations
    (r'\\cite\{[^}]*\}', 'cite'),
    (r'\\citet\{[^}]*\}', 'citet'),
    (r'\\citep\{[^}]*\}', 'citep'),
    (r'\\citeauthor\{[^}]*\}', 'citeauthor'),
    (r'\\citeyear\{[^}]*\}', 'citeyear'),
    
    # References
    (r'\\ref\{[^}]*\}', 'ref'),
    (r'\\eqref\{[^}]*\}', 'eqref'),
    (r'\\autoref\{[^}]*\}', 'autoref'),
    (r'\\cref\{[^}]*\}', 'cref'),
    (r'\\Cref\{[^}]*\}', 'Cref'),
    (r'\\pageref\{[^}]*\}', 'pageref'),
    
    # Labels
    (r'\\label\{[^}]*\}', 'label'),
]


# =============================================================================
# EXTRACTION
# =============================================================================

@dataclass
class ExtractedFormula:
    """An extracted formula with its metadata."""
    original: str
    placeholder: str
    formula_type: str
    start_pos: int
    end_pos: int


@dataclass
class IsolationResult:
    """Result of formula isolation."""
    text_with_placeholders: str
    formulas: List[ExtractedFormula] = field(default_factory=list)
    formula_map: Dict[str, str] = field(default_factory=dict)  # placeholder -> original
    
    def restore(self, translated_text: str) -> str:
        """Restore all formulas in translated text."""
        result = translated_text
        for placeholder, original in self.formula_map.items():
            result = result.replace(placeholder, original)
        return result
    
    def verify_restoration(self, restored_text: str) -> List[str]:
        """Verify all formulas were restored. Returns list of issues."""
        issues = []
        
        # Check no placeholders remain
        remaining = re.findall(r'⟦[A-Z]+_[a-f0-9]+_\d+⟧', restored_text)
        if remaining:
            issues.append(f"Unrestored placeholders: {remaining}")
        
        # Check all original formulas are present
        for formula in self.formulas:
            if formula.original not in restored_text:
                # Check if it might have minor whitespace differences
                normalized_orig = re.sub(r'\s+', ' ', formula.original)
                normalized_text = re.sub(r'\s+', ' ', restored_text)
                if normalized_orig not in normalized_text:
                    issues.append(f"Formula not found: {formula.original[:50]}...")
        
        return issues


def extract_formulas(text: str) -> IsolationResult:
    """
    Extract all formulas from text, replacing with placeholders.
    
    This is the core function for formula preservation.
    """
    result = IsolationResult(text_with_placeholders=text)
    
    # Track positions to avoid overlapping extractions
    extracted_ranges = []
    formula_index = 0
    
    def is_overlapping(start: int, end: int) -> bool:
        for r_start, r_end in extracted_ranges:
            if start < r_end and end > r_start:
                return True
        return False
    
    # First pass: extract all formula patterns
    all_patterns = FORMULA_PATTERNS + COMMAND_PATTERNS
    
    for pattern, formula_type in all_patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            start, end = match.start(), match.end()
            
            if is_overlapping(start, end):
                continue
            
            original = match.group()
            placeholder = generate_placeholder(original, formula_index, "F")
            
            formula = ExtractedFormula(
                original=original,
                placeholder=placeholder,
                formula_type=formula_type,
                start_pos=start,
                end_pos=end
            )
            
            result.formulas.append(formula)
            result.formula_map[placeholder] = original
            extracted_ranges.append((start, end))
            formula_index += 1
    
    # Sort by position (reverse) for safe replacement
    result.formulas.sort(key=lambda f: f.start_pos, reverse=True)
    
    # Replace formulas with placeholders
    modified_text = text
    for formula in result.formulas:
        modified_text = (
            modified_text[:formula.start_pos] + 
            formula.placeholder + 
            modified_text[formula.end_pos:]
        )
    
    result.text_with_placeholders = modified_text
    
    logger.info(f"Extracted {len(result.formulas)} formulas")
    return result


def extract_and_protect(text: str) -> Tuple[str, Callable[[str], str]]:
    """
    Extract formulas and return a restoration function.
    
    Usage:
        protected_text, restore = extract_and_protect(original_text)
        translated = translate(protected_text)
        final = restore(translated)
    """
    result = extract_formulas(text)
    
    def restore(translated: str) -> str:
        restored = result.restore(translated)
        issues = result.verify_restoration(restored)
        if issues:
            logger.warning(f"Formula restoration issues: {issues}")
        return restored
    
    return result.text_with_placeholders, restore


# =============================================================================
# ADVANCED: NESTED FORMULA HANDLING
# =============================================================================

def extract_nested_environments(text: str) -> IsolationResult:
    """
    Handle nested LaTeX environments correctly.
    
    Example: equation containing cases containing array
    """
    result = IsolationResult(text_with_placeholders=text)
    
    # Find outermost environments first
    env_pattern = r'\\begin\{([a-zA-Z*]+)\}'
    
    formula_index = 0
    current_text = text
    
    while True:
        # Find first \begin
        begin_match = re.search(env_pattern, current_text)
        if not begin_match:
            break
        
        env_name = begin_match.group(1)
        start_pos = begin_match.start()
        
        # Find matching \end (handling nesting)
        depth = 1
        pos = begin_match.end()
        
        while depth > 0 and pos < len(current_text):
            next_begin = re.search(rf'\\begin\{{{env_name}\}}', current_text[pos:])
            next_end = re.search(rf'\\end\{{{env_name}\}}', current_text[pos:])
            
            if next_end is None:
                logger.warning(f"Unmatched \\begin{{{env_name}}}")
                break
            
            if next_begin and next_begin.start() < next_end.start():
                depth += 1
                pos += next_begin.end()
            else:
                depth -= 1
                if depth == 0:
                    end_pos = pos + next_end.end()
                    
                    # Extract this environment
                    original = current_text[start_pos:end_pos]
                    placeholder = generate_placeholder(original, formula_index, "ENV")
                    
                    result.formulas.append(ExtractedFormula(
                        original=original,
                        placeholder=placeholder,
                        formula_type=env_name,
                        start_pos=start_pos,
                        end_pos=end_pos
                    ))
                    result.formula_map[placeholder] = original
                    
                    # Replace in text
                    current_text = current_text[:start_pos] + placeholder + current_text[end_pos:]
                    formula_index += 1
                    break
                pos += next_end.end()
    
    result.text_with_placeholders = current_text
    
    # Also extract inline math that's not in environments
    inline_result = extract_formulas(current_text)
    
    # Merge results
    for formula in inline_result.formulas:
        if formula.placeholder not in result.formula_map:
            result.formulas.append(formula)
            result.formula_map[formula.placeholder] = formula.original
    
    result.text_with_placeholders = inline_result.text_with_placeholders
    
    return result


# =============================================================================
# SAFE TRANSLATION WRAPPER
# =============================================================================

def safe_translate(
    text: str,
    translate_func: Callable[[str], str],
    verify: bool = True
) -> Tuple[str, List[str]]:
    """
    Safely translate text while preserving all formulas.
    
    Args:
        text: Original text with formulas
        translate_func: Function that translates text
        verify: Whether to verify restoration
    
    Returns:
        Tuple of (translated_text, list_of_issues)
    """
    # Extract formulas
    protected_text, restore = extract_and_protect(text)
    
    # Translate (formulas are now placeholders)
    translated_protected = translate_func(protected_text)
    
    # Restore formulas
    final_text = restore(translated_protected)
    
    # Verify
    issues = []
    if verify:
        result = extract_formulas(text)
        issues = result.verify_restoration(final_text)
    
    return final_text, issues


def batch_safe_translate(
    blocks: List[str],
    translate_func: Callable[[str], str],
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Tuple[List[str], int]:
    """
    Safely translate multiple blocks.
    
    Returns (translated_blocks, issue_count)
    """
    results = []
    total_issues = 0
    
    for i, block in enumerate(blocks):
        if progress_callback:
            progress_callback(i + 1, len(blocks), f"Block {i + 1}/{len(blocks)}")
        
        translated, issues = safe_translate(block, translate_func)
        results.append(translated)
        total_issues += len(issues)
        
        if issues:
            logger.warning(f"Block {i}: {len(issues)} formula issues")
    
    return results, total_issues


# =============================================================================
# FORMULA INTEGRITY CHECK
# =============================================================================

def check_formula_integrity(original: str, translated: str) -> Dict:
    """
    Comprehensive check that formulas weren't corrupted.
    
    Returns dict with:
    - is_valid: bool
    - original_count: int
    - translated_count: int
    - missing: list of missing formulas
    - corrupted: list of corrupted formulas
    """
    result = {
        "is_valid": True,
        "original_count": 0,
        "translated_count": 0,
        "missing": [],
        "corrupted": []
    }
    
    # Extract formulas from both
    orig_result = extract_formulas(original)
    trans_result = extract_formulas(translated)
    
    result["original_count"] = len(orig_result.formulas)
    result["translated_count"] = len(trans_result.formulas)
    
    # Check each original formula exists in translation
    orig_formulas = {f.original for f in orig_result.formulas}
    trans_formulas = {f.original for f in trans_result.formulas}
    
    for formula in orig_formulas:
        if formula not in trans_formulas:
            # Check for whitespace-normalized match
            normalized = re.sub(r'\s+', '', formula)
            found = False
            for tf in trans_formulas:
                if re.sub(r'\s+', '', tf) == normalized:
                    found = True
                    break
            
            if not found:
                result["missing"].append(formula[:100])
                result["is_valid"] = False
    
    # Check for formulas that appear corrupted
    for formula in trans_formulas:
        # Check for common corruption patterns
        if re.search(r'(?<![a-zA-Z])the\s+\$', formula):  # "the $x$" inside formula
            result["corrupted"].append(formula[:100])
            result["is_valid"] = False
    
    return result


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Formula Isolator Test ===\n")
    
    # Test text with various formulas
    test_text = """
The famous equation $E = mc^2$ shows mass-energy equivalence.

For the Schwarzschild metric, we have:
$$ds^2 = -\\left(1 - \\frac{r_s}{r}\\right)dt^2 + \\left(1 - \\frac{r_s}{r}\\right)^{-1}dr^2$$

As shown in \\cite{einstein1915}, the field equations are:
\\begin{equation}
R_{\\mu\\nu} - \\frac{1}{2}g_{\\mu\\nu}R = \\frac{8\\pi G}{c^4}T_{\\mu\\nu}
\\label{eq:einstein}
\\end{equation}

See Equation \\eqref{eq:einstein} for details.
"""
    
    print("Original text:")
    print(test_text[:200] + "...\n")
    
    # Extract formulas
    result = extract_formulas(test_text)
    
    print(f"Extracted {len(result.formulas)} formulas:")
    for f in result.formulas:
        print(f"  [{f.formula_type}] {f.original[:50]}...")
    
    print(f"\nText with placeholders:")
    print(result.text_with_placeholders[:300] + "...\n")
    
    # Simulate translation
    def mock_translate(text):
        return text.replace("famous", "berühmte").replace("shows", "zeigt")
    
    protected, restore = extract_and_protect(test_text)
    translated = mock_translate(protected)
    restored = restore(translated)
    
    print("After mock translation + restoration:")
    print(restored[:300] + "...\n")
    
    # Verify integrity
    integrity = check_formula_integrity(test_text, restored)
    print(f"Formula integrity: {'✅ VALID' if integrity['is_valid'] else '❌ INVALID'}")
    print(f"  Original formulas: {integrity['original_count']}")
    print(f"  Translated formulas: {integrity['translated_count']}")
    if integrity['missing']:
        print(f"  Missing: {integrity['missing']}")
    
    print("\n✅ Formula Isolator ready")
