"""
Quality Assurance - Comprehensive Translation Quality Verification

Ensures translation quality through multiple validation methods:
1. Back-translation comparison
2. Semantic similarity scoring
3. Formula integrity checking
4. Terminology consistency
5. Overall quality scoring

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import logging
import re
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("pdf_translator.qa")


# =============================================================================
# QUALITY METRICS
# =============================================================================

class QualityLevel(Enum):
    """Translation quality levels."""
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"            # 75-89
    ACCEPTABLE = "acceptable"  # 60-74
    POOR = "poor"            # 40-59
    FAILED = "failed"        # 0-39


@dataclass
class QualityMetrics:
    """Detailed quality metrics for a translation."""
    # Core scores (0-100)
    overall_score: float = 0.0
    semantic_score: float = 0.0
    formula_score: float = 0.0
    terminology_score: float = 0.0
    completeness_score: float = 0.0
    
    # Details
    word_count_original: int = 0
    word_count_translated: int = 0
    formula_count_original: int = 0
    formula_count_translated: int = 0
    
    # Issues found
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def level(self) -> QualityLevel:
        if self.overall_score >= 90:
            return QualityLevel.EXCELLENT
        elif self.overall_score >= 75:
            return QualityLevel.GOOD
        elif self.overall_score >= 60:
            return QualityLevel.ACCEPTABLE
        elif self.overall_score >= 40:
            return QualityLevel.POOR
        else:
            return QualityLevel.FAILED
    
    @property
    def passed(self) -> bool:
        return self.overall_score >= 60
    
    def to_dict(self) -> Dict:
        return {
            "overall_score": self.overall_score,
            "level": self.level.value,
            "passed": self.passed,
            "semantic_score": self.semantic_score,
            "formula_score": self.formula_score,
            "terminology_score": self.terminology_score,
            "completeness_score": self.completeness_score,
            "issues": self.issues,
            "warnings": self.warnings,
        }


@dataclass
class QAReport:
    """Complete QA report for a translated document."""
    document_name: str
    target_language: str
    block_count: int
    metrics: QualityMetrics
    block_metrics: List[QualityMetrics] = field(default_factory=list)
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        level_emoji = {
            QualityLevel.EXCELLENT: "🌟",
            QualityLevel.GOOD: "✅",
            QualityLevel.ACCEPTABLE: "⚠️",
            QualityLevel.POOR: "❌",
            QualityLevel.FAILED: "🚫",
        }
        
        emoji = level_emoji[self.metrics.level]
        
        report = f"""# Translation Quality Report

## Summary
- **Document**: {self.document_name}
- **Target Language**: {self.target_language}
- **Blocks**: {self.block_count}
- **Overall Score**: {self.metrics.overall_score:.1f}/100 {emoji} {self.metrics.level.value.upper()}

## Detailed Scores
| Metric | Score |
|--------|-------|
| Semantic Similarity | {self.metrics.semantic_score:.1f} |
| Formula Preservation | {self.metrics.formula_score:.1f} |
| Terminology Consistency | {self.metrics.terminology_score:.1f} |
| Completeness | {self.metrics.completeness_score:.1f} |

## Statistics
- Original words: {self.metrics.word_count_original}
- Translated words: {self.metrics.word_count_translated}
- Original formulas: {self.metrics.formula_count_original}
- Translated formulas: {self.metrics.formula_count_translated}
"""
        
        if self.metrics.issues:
            report += "\n## Issues\n"
            for issue in self.metrics.issues:
                report += f"- ❌ {issue}\n"
        
        if self.metrics.warnings:
            report += "\n## Warnings\n"
            for warning in self.metrics.warnings:
                report += f"- ⚠️ {warning}\n"
        
        return report


# =============================================================================
# BACK-TRANSLATION
# =============================================================================

def back_translate(
    translated_text: str,
    translate_func: Callable[[str], str],
    source_language: str
) -> str:
    """
    Translate text back to source language for comparison.
    
    Args:
        translated_text: The translated text
        translate_func: Function to translate to source language
        source_language: Name of source language
    
    Returns:
        Back-translated text
    """
    return translate_func(translated_text)


def calculate_back_translation_similarity(
    original: str,
    back_translated: str
) -> float:
    """
    Calculate similarity between original and back-translated text.
    
    Uses word overlap and position similarity.
    Returns score 0-100.
    """
    if not original or not back_translated:
        return 0.0
    
    # Normalize texts
    orig_words = set(original.lower().split())
    back_words = set(back_translated.lower().split())
    
    if not orig_words:
        return 0.0
    
    # Word overlap
    overlap = len(orig_words & back_words)
    union = len(orig_words | back_words)
    
    jaccard = overlap / union if union > 0 else 0
    
    # Weighted overlap (favor matching original words)
    precision = overlap / len(orig_words) if orig_words else 0
    
    # Combine scores
    score = (jaccard * 0.4 + precision * 0.6) * 100
    
    return min(100, score)


def validate_with_back_translation(
    original: str,
    translated: str,
    back_translate_func: Callable[[str], str],
    threshold: float = 50.0
) -> Tuple[bool, float, str]:
    """
    Validate translation quality using back-translation.
    
    Returns (is_valid, score, back_translated_text)
    """
    back_translated = back_translate_func(translated)
    score = calculate_back_translation_similarity(original, back_translated)
    is_valid = score >= threshold
    
    return is_valid, score, back_translated


# =============================================================================
# FORMULA VERIFICATION
# =============================================================================

def extract_formulas(text: str) -> List[str]:
    """Extract all formulas from text."""
    formulas = []
    
    patterns = [
        r'\$\$.*?\$\$',
        r'\$[^$]+\$',
        r'\\\[.*?\\\]',
        r'\\\(.*?\\\)',
        r'\\begin\{equation\*?\}.*?\\end\{equation\*?\}',
        r'\\begin\{align\*?\}.*?\\end\{align\*?\}',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        formulas.extend(matches)
    
    return formulas


def verify_formula_preservation(
    original: str,
    translated: str
) -> Tuple[float, List[str]]:
    """
    Verify all formulas are preserved in translation.
    
    Returns (score 0-100, list of issues)
    """
    orig_formulas = extract_formulas(original)
    trans_formulas = extract_formulas(translated)
    
    if not orig_formulas:
        return 100.0, []  # No formulas to preserve
    
    issues = []
    preserved = 0
    
    for formula in orig_formulas:
        # Check exact match
        if formula in translated:
            preserved += 1
        else:
            # Check normalized match (whitespace differences)
            normalized = re.sub(r'\s+', '', formula)
            trans_normalized = re.sub(r'\s+', '', translated)
            
            if normalized in trans_normalized:
                preserved += 1
            else:
                issues.append(f"Formula possibly modified: {formula[:50]}...")
    
    score = (preserved / len(orig_formulas)) * 100
    
    # Check for extra formulas (shouldn't appear)
    extra = len(trans_formulas) - len(orig_formulas)
    if extra > 0:
        issues.append(f"{extra} extra formula(s) appeared in translation")
        score = max(0, score - extra * 5)
    
    return score, issues


# =============================================================================
# TERMINOLOGY CONSISTENCY
# =============================================================================

def extract_technical_terms(text: str) -> List[str]:
    """Extract technical terms from text."""
    terms = []
    
    # Capitalized phrases
    terms.extend(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text))
    
    # All-caps acronyms
    terms.extend(re.findall(r'\b[A-Z]{2,6}\b', text))
    
    # Terms with Greek letters
    terms.extend(re.findall(r'\\(?:alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)\b', text))
    
    return list(set(terms))


def check_terminology_consistency(
    blocks_original: List[str],
    blocks_translated: List[str]
) -> Tuple[float, List[str]]:
    """
    Check if terminology is used consistently across document.
    
    Returns (score 0-100, list of issues)
    """
    issues = []
    
    # Build term usage map
    term_translations: Dict[str, set] = {}
    
    for orig, trans in zip(blocks_original, blocks_translated):
        terms = extract_technical_terms(orig)
        
        for term in terms:
            if term not in term_translations:
                term_translations[term] = set()
            
            # Check if term appears unchanged in translation
            if term in trans:
                term_translations[term].add(term)
            else:
                term_translations[term].add("[translated]")
    
    # Check for inconsistencies
    inconsistent_terms = 0
    for term, usages in term_translations.items():
        if len(usages) > 1:
            inconsistent_terms += 1
            issues.append(f"Inconsistent translation of '{term}'")
    
    if not term_translations:
        return 100.0, []
    
    consistency_rate = 1 - (inconsistent_terms / len(term_translations))
    score = consistency_rate * 100
    
    return score, issues


# =============================================================================
# COMPLETENESS CHECK
# =============================================================================

def check_completeness(
    original: str,
    translated: str,
    min_ratio: float = 0.5,
    max_ratio: float = 2.0
) -> Tuple[float, List[str]]:
    """
    Check if translation is complete (not truncated or over-expanded).
    
    Returns (score 0-100, list of issues)
    """
    issues = []
    
    orig_words = len(original.split())
    trans_words = len(translated.split())
    
    if orig_words == 0:
        return 100.0, []
    
    ratio = trans_words / orig_words
    
    if ratio < min_ratio:
        issues.append(f"Translation too short: {ratio:.1%} of original")
        score = (ratio / min_ratio) * 80  # Max 80 if too short
    elif ratio > max_ratio:
        issues.append(f"Translation too long: {ratio:.1%} of original")
        score = (max_ratio / ratio) * 80  # Max 80 if too long
    else:
        # Good range - score based on how close to 1.0
        deviation = abs(1.0 - ratio)
        score = 100 - (deviation * 20)  # -20 points per 100% deviation
    
    return max(0, min(100, score)), issues


# =============================================================================
# COMPREHENSIVE QA
# =============================================================================

def run_quality_check(
    original: str,
    translated: str,
    back_translate_func: Optional[Callable[[str], str]] = None
) -> QualityMetrics:
    """
    Run comprehensive quality check on a single block.
    
    Args:
        original: Original text
        translated: Translated text
        back_translate_func: Optional function for back-translation
    
    Returns:
        QualityMetrics with all scores
    """
    metrics = QualityMetrics()
    
    # Word counts
    metrics.word_count_original = len(original.split())
    metrics.word_count_translated = len(translated.split())
    
    # Formula counts
    orig_formulas = extract_formulas(original)
    trans_formulas = extract_formulas(translated)
    metrics.formula_count_original = len(orig_formulas)
    metrics.formula_count_translated = len(trans_formulas)
    
    # Formula preservation
    metrics.formula_score, formula_issues = verify_formula_preservation(original, translated)
    metrics.issues.extend(formula_issues)
    
    # Completeness
    metrics.completeness_score, completeness_issues = check_completeness(original, translated)
    metrics.warnings.extend(completeness_issues)
    
    # Back-translation (if available)
    if back_translate_func:
        is_valid, back_score, _ = validate_with_back_translation(
            original, translated, back_translate_func
        )
        metrics.semantic_score = back_score
        if not is_valid:
            metrics.warnings.append("Back-translation similarity below threshold")
    else:
        metrics.semantic_score = 80  # Assume good if not tested
    
    # Terminology (single block - limited check)
    metrics.terminology_score = 90  # Default high for single block
    
    # Calculate overall score
    weights = {
        "formula": 0.35,       # Formulas most important
        "semantic": 0.30,      # Meaning preservation
        "completeness": 0.20,  # Complete translation
        "terminology": 0.15,   # Consistent terms
    }
    
    metrics.overall_score = (
        metrics.formula_score * weights["formula"] +
        metrics.semantic_score * weights["semantic"] +
        metrics.completeness_score * weights["completeness"] +
        metrics.terminology_score * weights["terminology"]
    )
    
    return metrics


def run_document_qa(
    blocks_original: List[str],
    blocks_translated: List[str],
    document_name: str,
    target_language: str,
    back_translate_func: Optional[Callable[[str], str]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> QAReport:
    """
    Run comprehensive QA on an entire translated document.
    
    Args:
        blocks_original: List of original text blocks
        blocks_translated: List of translated text blocks
        document_name: Name of the document
        target_language: Target language
        back_translate_func: Optional function for back-translation
        progress_callback: Optional progress callback
    
    Returns:
        Complete QAReport
    """
    report = QAReport(
        document_name=document_name,
        target_language=target_language,
        block_count=len(blocks_original),
        metrics=QualityMetrics()
    )
    
    all_issues = []
    all_warnings = []
    
    # Check each block
    for i, (orig, trans) in enumerate(zip(blocks_original, blocks_translated)):
        if progress_callback:
            progress_callback(i + 1, len(blocks_original), f"QA Block {i + 1}")
        
        block_metrics = run_quality_check(orig, trans, back_translate_func)
        report.block_metrics.append(block_metrics)
        
        all_issues.extend(block_metrics.issues)
        all_warnings.extend(block_metrics.warnings)
    
    # Document-level terminology check
    term_score, term_issues = check_terminology_consistency(blocks_original, blocks_translated)
    all_issues.extend(term_issues)
    
    # Aggregate metrics
    if report.block_metrics:
        n = len(report.block_metrics)
        report.metrics.formula_score = sum(m.formula_score for m in report.block_metrics) / n
        report.metrics.semantic_score = sum(m.semantic_score for m in report.block_metrics) / n
        report.metrics.completeness_score = sum(m.completeness_score for m in report.block_metrics) / n
        report.metrics.terminology_score = term_score
        
        report.metrics.word_count_original = sum(m.word_count_original for m in report.block_metrics)
        report.metrics.word_count_translated = sum(m.word_count_translated for m in report.block_metrics)
        report.metrics.formula_count_original = sum(m.formula_count_original for m in report.block_metrics)
        report.metrics.formula_count_translated = sum(m.formula_count_translated for m in report.block_metrics)
    
    # Calculate overall
    weights = {"formula": 0.35, "semantic": 0.30, "completeness": 0.20, "terminology": 0.15}
    report.metrics.overall_score = (
        report.metrics.formula_score * weights["formula"] +
        report.metrics.semantic_score * weights["semantic"] +
        report.metrics.completeness_score * weights["completeness"] +
        report.metrics.terminology_score * weights["terminology"]
    )
    
    report.metrics.issues = list(set(all_issues))[:10]  # Top 10 unique issues
    report.metrics.warnings = list(set(all_warnings))[:10]
    
    logger.info(f"QA complete: {report.metrics.overall_score:.1f}/100 ({report.metrics.level.value})")
    
    return report


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Quality Assurance Test ===\n")
    
    # Test data
    original = """The black hole has an event horizon at radius $r_s = 2GM/c^2$.
This is known as the Schwarzschild radius, named after Karl Schwarzschild."""
    
    translated_good = """Das schwarze Loch hat einen Ereignishorizont beim Radius $r_s = 2GM/c^2$.
Dies ist als Schwarzschild-Radius bekannt, benannt nach Karl Schwarzschild."""
    
    translated_bad = """The black hole has a horizon at r=2GM.
This is the Schwarzschild thing."""
    
    # Test good translation
    print("### Good Translation")
    metrics = run_quality_check(original, translated_good)
    print(f"Score: {metrics.overall_score:.1f}/100 ({metrics.level.value})")
    print(f"Formula: {metrics.formula_score:.1f}")
    print(f"Completeness: {metrics.completeness_score:.1f}")
    
    # Test bad translation
    print("\n### Bad Translation")
    metrics = run_quality_check(original, translated_bad)
    print(f"Score: {metrics.overall_score:.1f}/100 ({metrics.level.value})")
    print(f"Formula: {metrics.formula_score:.1f}")
    print(f"Issues: {metrics.issues}")
    
    # Test document QA
    print("\n### Document QA")
    blocks_orig = [original, "See Equation (1) for details."]
    blocks_trans = [translated_good, "Siehe Gleichung (1) für Details."]
    
    report = run_document_qa(blocks_orig, blocks_trans, "test.pdf", "German")
    print(report.to_markdown()[:500] + "...")
    
    print("\n✅ Quality Assurance module ready")
