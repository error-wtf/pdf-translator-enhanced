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

# =============================================================================
# UNICODE MATH PATTERNS - CRITICAL FOR SCIENTIFIC PDFs
# =============================================================================
# These patterns protect Unicode math symbols that get destroyed during translation

UNICODE_MATH_PATTERNS = [
    # Physics equations with Unicode (e.g., Schrödinger: iħ∂Ψ/∂t = ĤΨ)
    (r'[iℏħ]\s*[∂∇]\s*[ΨΦψφ]\s*/\s*[∂∇]\s*[trxyz]', 'schrodinger'),
    (r'[ĤĥH]\s*[ΨΦψφ]\s*[=]\s*[EΕε]\s*[ΨΦψφ]', 'eigenvalue'),
    
    # Greek letter sequences (2+ chars = likely formula)
    (r'[αβγδεζηθικλμνξοπρσςτυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩϕϵϑϱϖ]{2,}', 'greek_seq'),
    
    # Math operator sequences
    (r'[∂∇∫∮∑∏√∞±∓≈≠≤≥≪≫∝∈∉⊂⊃∪∩∧∨⊕⊗⊥∥†‡×÷·]+', 'operator_seq'),
    
    # Scientific notation with Unicode
    (r'\d+\.?\d*\s*[×·]\s*10\s*[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+', 'sci_notation'),
    
    # Subscript/superscript sequences
    (r'[₀₁₂₃₄₅₆₇₈₉₊₋₌ₐₑₒₓₕₖₗₘₙₚₛₜ]+', 'subscript_seq'),
    (r'[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼ⁿⁱ]+', 'superscript_seq'),
    
    # Special physics constants
    (r'[ℏħℓ℘ℜℑ]', 'physics_const'),
    
    # Modified letters (hat, etc.)
    (r'[ĤĥĜĝŴŵŜŝÂâÊêÎîÔôÛû]', 'modified_letter'),
    
    # Arrows in equations
    (r'[→←↔⇒⇐⇔↑↓⟶⟵⟷⟹⟸⟺]', 'arrow'),
    
    # Numbers with units (protect the whole expression)
    (r'\d+\.?\d*\s*(?:Hz|kHz|MHz|GHz|THz|nm|μm|µm|mm|cm|m|km|Å|pm|fm|ns|μs|µs|ms|s|eV|keV|MeV|GeV|TeV|K|°C|Pa|kPa|MPa|GPa|J|kJ|W|kW|MW|V|mV|kV|A|mA|μA|Ω|kΩ|MΩ|F|μF|nF|pF|H|mH|T|mol|M|L|mL|g|mg|μg|kg)\b', 'unit_expr'),
]

# All patterns combined for extraction
ALL_MATH_PATTERNS = FORMULA_PATTERNS + [(p, t) for p, t in UNICODE_MATH_PATTERNS]

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
            # Case-insensitive replacement (LLM might change case)
            pattern = re.escape(placeholder)
            result = re.sub(pattern, original.replace('\\', '\\\\'), result, flags=re.IGNORECASE)
        return result
    
    def verify_restoration(self, restored_text: str) -> List[str]:
        """Verify all formulas were restored. Returns list of issues."""
        issues = []
        
        # Check no placeholders remain (case-insensitive for hex chars)
        remaining = re.findall(r'⟦[A-Z]+_[a-fA-F0-9]+_\d+⟧', restored_text, re.IGNORECASE)
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
    
    # First pass: extract all formula patterns (including Unicode math)
    all_patterns = ALL_MATH_PATTERNS + COMMAND_PATTERNS
    
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
# UTF-8 ENFORCEMENT
# =============================================================================

def audit_utf8(text: str, source: str = "unknown") -> List[str]:
    """
    Audit text for UTF-8 issues and potential encoding problems.
    
    Returns list of warnings. Empty list = all good.
    """
    warnings = []
    
    # Check for replacement characters (indicates encoding loss)
    if '?' in text and text.count('?') > 5:
        # Could be legitimate question marks or encoding failures
        # Check if ? appears in suspicious patterns
        if re.search(r'\?\?+', text):  # Multiple ?? in a row
            warnings.append(f"[{source}] Possible encoding loss: found '??' pattern")
    
    if '\ufffd' in text:
        count = text.count('\ufffd')
        warnings.append(f"[{source}] Found {count} replacement characters (U+FFFD)")
    
    # Check for common encoding corruption patterns
    corruption_patterns = [
        (r'Ã¤', 'ä'),  # UTF-8 decoded as Latin-1
        (r'Ã¶', 'ö'),
        (r'Ã¼', 'ü'),
        (r'Ã©', 'é'),
        (r'â€™', "'"),
        (r'â€"', '–'),
        (r'â€œ', '"'),
    ]
    
    for pattern, should_be in corruption_patterns:
        if pattern in text:
            warnings.append(f"[{source}] Encoding corruption: '{pattern}' should be '{should_be}'")
    
    return warnings


def ensure_utf8_safe(text: str) -> str:
    """
    Ensure text is safely encodable as UTF-8.
    
    Does NOT drop characters - raises error if impossible.
    """
    try:
        # Encode and decode to verify
        encoded = text.encode('utf-8')
        decoded = encoded.decode('utf-8')
        return decoded
    except UnicodeError as e:
        logger.error(f"UTF-8 encoding failed: {e}")
        raise ValueError(f"Text contains characters that cannot be encoded as UTF-8: {e}")


# =============================================================================
# OUTPUT NORMALIZATION
# =============================================================================

def normalize_output(text: str, mode: str = "unicode") -> str:
    """
    Normalize output to a single consistent representation.
    
    Modes:
    - "unicode": Keep Unicode math, remove HTML/raw LaTeX
    - "latex": Convert to LaTeX math mode
    - "plain": Convert to plain text approximation
    
    This prevents mixing HTML, LaTeX, and Unicode in the same output.
    """
    result = text
    
    if mode == "unicode":
        # Remove HTML tags, keep Unicode
        result = re.sub(r'<sup>([^<]+)</sup>', lambda m: _to_superscript(m.group(1)), result)
        result = re.sub(r'<sub>([^<]+)</sub>', lambda m: _to_subscript(m.group(1)), result)
        result = re.sub(r'</?[a-zA-Z][^>]*>', '', result)  # Remove remaining HTML
        
        # Remove raw LaTeX commands that weren't rendered
        # But keep math-mode content
        result = re.sub(r'\\textbf\{([^}]+)\}', r'\1', result)
        result = re.sub(r'\\textit\{([^}]+)\}', r'\1', result)
        result = re.sub(r'\\emph\{([^}]+)\}', r'\1', result)
        
    elif mode == "latex":
        # Convert Unicode to LaTeX
        result = _unicode_to_latex(result)
        
        # Convert HTML to LaTeX
        result = re.sub(r'<sup>([^<]+)</sup>', r'^{\1}', result)
        result = re.sub(r'<sub>([^<]+)</sub>', r'_{\1}', result)
        result = re.sub(r'<b>([^<]+)</b>', r'\\textbf{\1}', result)
        result = re.sub(r'<i>([^<]+)</i>', r'\\textit{\1}', result)
        result = re.sub(r'</?[a-zA-Z][^>]*>', '', result)
        
    elif mode == "plain":
        # Remove all formatting
        result = re.sub(r'<[^>]+>', '', result)
        result = re.sub(r'\$[^$]+\$', '[formula]', result)
        result = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', result)
    
    return result


def _to_superscript(text: str) -> str:
    """Convert text to Unicode superscript."""
    sup_map = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
               '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
               '+': '⁺', '-': '⁻', '=': '⁼', 'n': 'ⁿ', 'i': 'ⁱ',
               '*': '∗', ',': '⸴'}
    return ''.join(sup_map.get(c, c) for c in text)


def _to_subscript(text: str) -> str:
    """Convert text to Unicode subscript."""
    sub_map = {'0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
               '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
               '+': '₊', '-': '₋', '=': '₌', 'a': 'ₐ', 'e': 'ₑ',
               'o': 'ₒ', 'x': 'ₓ', 'n': 'ₙ'}
    return ''.join(sub_map.get(c, c) for c in text)


def _unicode_to_latex(text: str) -> str:
    """Convert Unicode math to LaTeX."""
    # Import from latex_build to avoid duplication
    try:
        from latex_build import sanitize_for_latex
        return sanitize_for_latex(text)
    except ImportError:
        # Fallback minimal conversion
        conversions = {
            'α': r'$\alpha$', 'β': r'$\beta$', 'γ': r'$\gamma$',
            'Ψ': r'$\Psi$', 'ψ': r'$\psi$', 'Φ': r'$\Phi$', 'φ': r'$\varphi$',
            '∇': r'$\nabla$', '∂': r'$\partial$', '∫': r'$\int$',
            'ℏ': r'$\hbar$', 'ħ': r'$\hbar$',
            '∞': r'$\infty$', '±': r'$\pm$', '×': r'$\times$',
        }
        for unicode_char, latex in conversions.items():
            text = text.replace(unicode_char, latex)
        return text


# =============================================================================
# REGRESSION CHECKS
# =============================================================================

def regression_check(original: str, translated: str, restored: str) -> Dict:
    """
    Comprehensive regression check for translation quality.
    
    Returns dict with pass/fail status and detailed issues.
    """
    result = {
        "passed": True,
        "issues": [],
        "stats": {
            "original_length": len(original),
            "translated_length": len(translated),
            "restored_length": len(restored),
            "formulas_original": 0,
            "formulas_restored": 0,
        }
    }
    
    # Check 1: No '??' corruption patterns
    if re.search(r'\?\?+', restored):
        result["passed"] = False
        result["issues"].append("CRITICAL: Found '??' pattern indicating encoding loss")
    
    # Check 2: No replacement characters
    if '\ufffd' in restored:
        result["passed"] = False
        result["issues"].append(f"CRITICAL: Found {restored.count(chr(0xFFFD))} replacement characters")
    
    # Check 3: All placeholders were reinserted
    remaining_placeholders = re.findall(r'⟦[A-Z]+_[a-f0-9]+_\d+⟧', restored)
    if remaining_placeholders:
        result["passed"] = False
        result["issues"].append(f"ERROR: {len(remaining_placeholders)} unrestored placeholders")
    
    # Check 4: Formula count preserved
    orig_result = extract_formulas(original)
    rest_result = extract_formulas(restored)
    
    result["stats"]["formulas_original"] = len(orig_result.formulas)
    result["stats"]["formulas_restored"] = len(rest_result.formulas)
    
    if len(rest_result.formulas) < len(orig_result.formulas) * 0.9:  # Allow 10% tolerance
        result["passed"] = False
        result["issues"].append(
            f"WARNING: Formula count dropped from {len(orig_result.formulas)} to {len(rest_result.formulas)}"
        )
    
    # Check 5: No raw HTML leaked
    if re.search(r'<(sup|sub|b|i|em|strong)[^>]*>', restored, re.IGNORECASE):
        result["issues"].append("WARNING: Raw HTML tags found in output")
    
    # Check 6: Length sanity (translated shouldn't be drastically different)
    length_ratio = len(restored) / max(len(original), 1)
    if length_ratio < 0.5 or length_ratio > 2.0:
        result["issues"].append(f"WARNING: Unusual length ratio: {length_ratio:.2f}")
    
    return result


def assert_no_corruption(text: str, raise_on_fail: bool = False) -> bool:
    """
    Quick assertion that text has no obvious corruption.
    
    Use this as a sanity check at key pipeline stages.
    """
    issues = []
    
    if re.search(r'\?\?+', text):
        issues.append("Found '??' corruption pattern")
    
    if '\ufffd' in text:
        issues.append("Found replacement character")
    
    if re.search(r'⟦[A-Z]+_[a-f0-9]+_\d+⟧', text):
        issues.append("Found unrestored placeholder")
    
    if issues:
        msg = f"Corruption detected: {'; '.join(issues)}"
        logger.error(msg)
        if raise_on_fail:
            raise ValueError(msg)
        return False
    
    return True


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
