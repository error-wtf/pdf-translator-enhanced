"""
Quality Validator - End-to-End validation for perfect translations
© 2025 Sven Kalinowski - Anti-Capitalist Software License v1.4
"""
from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger("pdf_translator.validator")

@dataclass
class ValidationResult:
    passed: bool
    score: float
    issues: List[str] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)

def count_formulas(text: str) -> int:
    patterns = [r'\$[^$]+\$', r'[αβγδεζηθικλμνξοπρσςτυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]',
                r'[∂∇∫∮∑∏√∞±∓≈≠≤≥]', r'[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]', r'[₀₁₂₃₄₅₆₇₈₉₊₋]']
    return sum(len(re.findall(p, text)) for p in patterns)

def count_references(text: str) -> int:
    patterns = [r'\[\d+(?:\s*[-–,]\s*\d+)*\]', r'(?:Fig\.|Figure)\s*\d+', r'Eq\.\s*\(\d+\)']
    return sum(len(re.findall(p, text)) for p in patterns)

def detect_corruption(text: str) -> List[Tuple[str, str]]:
    issues = []
    if '\ufffd' in text: issues.append(("critical", "Replacement char found"))
    if '??' in text: issues.append(("critical", "Double ?? found"))
    if re.search(r'⟦[A-Z]+_[a-fA-F0-9]+_\d+⟧', text): issues.append(("critical", "Unrestored placeholder"))
    if re.search(r'<(?:sup|sub|b|i)[^>]*>', text): issues.append(("warning", "HTML tags in output"))
    return issues

def validate_translation(original: str, translated: str) -> ValidationResult:
    result = ValidationResult(passed=True, score=1.0)
    result.stats = {
        'orig_formulas': count_formulas(original),
        'trans_formulas': count_formulas(translated),
        'orig_refs': count_references(original),
        'trans_refs': count_references(translated),
    }
    
    # Check formula loss
    if result.stats['orig_formulas'] > result.stats['trans_formulas']:
        lost = result.stats['orig_formulas'] - result.stats['trans_formulas']
        result.issues.append(f"CRITICAL: Lost {lost} formulas")
        result.score -= 0.3
        result.passed = False
    
    # Check corruption
    for severity, msg in detect_corruption(translated):
        result.issues.append(f"{severity.upper()}: {msg}")
        if severity == "critical":
            result.score -= 0.2
            result.passed = False
    
    result.score = max(0.0, result.score)
    return result

def assert_quality(original: str, translated: str, min_score: float = 0.8) -> str:
    """Return translated if quality OK, else original."""
    result = validate_translation(original, translated)
    if result.score >= min_score:
        return translated
    logger.warning(f"Quality failed ({result.score:.0%}), using original")
    return original
