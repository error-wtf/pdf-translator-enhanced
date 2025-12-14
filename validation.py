"""
Validation - LaTeX Syntax and Translation Quality Checks

Validates translated content for:
1. LaTeX syntax correctness (balanced braces, environments)
2. Translation completeness
3. Formula preservation
4. Common error patterns

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import logging
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("pdf_translator.validation")


# =============================================================================
# VALIDATION SEVERITY
# =============================================================================

class Severity(Enum):
    """Severity level for validation issues."""
    ERROR = "error"      # Must be fixed
    WARNING = "warning"  # Should be reviewed
    INFO = "info"        # Minor issue


@dataclass
class ValidationIssue:
    """A validation issue found in the text."""
    message: str
    severity: Severity
    position: int = 0
    line: int = 0
    context: str = ""
    auto_fixable: bool = False
    fix_suggestion: str = ""


@dataclass
class ValidationResult:
    """Result of validation with issues and score."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    score: float = 1.0  # 0.0 to 1.0
    fixed_text: Optional[str] = None
    
    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)
    
    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)


# =============================================================================
# LATEX SYNTAX VALIDATION
# =============================================================================

def check_balanced_braces(text: str) -> List[ValidationIssue]:
    """Check for balanced curly braces {}."""
    issues = []
    stack = []
    
    i = 0
    while i < len(text):
        char = text[i]
        
        # Skip escaped braces
        if i > 0 and text[i-1] == '\\':
            i += 1
            continue
        
        if char == '{':
            stack.append(i)
        elif char == '}':
            if not stack:
                issues.append(ValidationIssue(
                    message="Unmatched closing brace '}'",
                    severity=Severity.ERROR,
                    position=i,
                    context=text[max(0, i-20):i+20],
                    auto_fixable=False
                ))
            else:
                stack.pop()
        
        i += 1
    
    # Check for unclosed braces
    for pos in stack:
        issues.append(ValidationIssue(
            message="Unclosed opening brace '{'",
            severity=Severity.ERROR,
            position=pos,
            context=text[max(0, pos-10):pos+30],
            auto_fixable=True,
            fix_suggestion="Add closing '}'"
        ))
    
    return issues


def check_balanced_brackets(text: str) -> List[ValidationIssue]:
    """Check for balanced square brackets []."""
    issues = []
    stack = []
    
    i = 0
    while i < len(text):
        char = text[i]
        
        if char == '[':
            stack.append(i)
        elif char == ']':
            if not stack:
                issues.append(ValidationIssue(
                    message="Unmatched closing bracket ']'",
                    severity=Severity.WARNING,
                    position=i,
                    context=text[max(0, i-20):i+20]
                ))
            else:
                stack.pop()
        
        i += 1
    
    for pos in stack:
        issues.append(ValidationIssue(
            message="Unclosed opening bracket '['",
            severity=Severity.WARNING,
            position=pos,
            context=text[max(0, pos-10):pos+30]
        ))
    
    return issues


def check_math_delimiters(text: str) -> List[ValidationIssue]:
    """Check for balanced math delimiters $ and $$."""
    issues = []
    
    # Check inline math $...$
    # Count $ that are not $$ and not escaped
    single_dollars = []
    i = 0
    while i < len(text):
        if text[i] == '$':
            # Check if escaped
            if i > 0 and text[i-1] == '\\':
                i += 1
                continue
            
            # Check if double $$
            if i + 1 < len(text) and text[i+1] == '$':
                i += 2
                continue
            
            single_dollars.append(i)
        i += 1
    
    if len(single_dollars) % 2 != 0:
        issues.append(ValidationIssue(
            message="Odd number of $ delimiters - math mode not closed",
            severity=Severity.ERROR,
            position=single_dollars[-1] if single_dollars else 0,
            auto_fixable=True,
            fix_suggestion="Add closing $"
        ))
    
    # Check display math $$...$$
    double_dollar_pattern = r'\$\$'
    matches = list(re.finditer(double_dollar_pattern, text))
    
    if len(matches) % 2 != 0:
        issues.append(ValidationIssue(
            message="Odd number of $$ delimiters - display math not closed",
            severity=Severity.ERROR,
            position=matches[-1].start() if matches else 0,
            auto_fixable=True,
            fix_suggestion="Add closing $$"
        ))
    
    # Check \[...\] and \(...\)
    open_bracket = len(re.findall(r'(?<!\\)\\\[', text))
    close_bracket = len(re.findall(r'(?<!\\)\\\]', text))
    
    if open_bracket != close_bracket:
        issues.append(ValidationIssue(
            message=f"Mismatched \\[...\\] delimiters: {open_bracket} open, {close_bracket} close",
            severity=Severity.ERROR
        ))
    
    open_paren = len(re.findall(r'(?<!\\)\\\(', text))
    close_paren = len(re.findall(r'(?<!\\)\\\)', text))
    
    if open_paren != close_paren:
        issues.append(ValidationIssue(
            message=f"Mismatched \\(...\\) delimiters: {open_paren} open, {close_paren} close",
            severity=Severity.ERROR
        ))
    
    return issues


def check_environments(text: str) -> List[ValidationIssue]:
    """Check for matched \\begin{} and \\end{} environments."""
    issues = []
    
    # Find all \begin{...}
    begin_pattern = r'\\begin\{([^}]+)\}'
    end_pattern = r'\\end\{([^}]+)\}'
    
    begins = [(m.group(1), m.start()) for m in re.finditer(begin_pattern, text)]
    ends = [(m.group(1), m.start()) for m in re.finditer(end_pattern, text)]
    
    # Build stack to match environments
    env_stack = []
    all_envs = sorted(
        [(name, pos, 'begin') for name, pos in begins] +
        [(name, pos, 'end') for name, pos in ends],
        key=lambda x: x[1]
    )
    
    for name, pos, typ in all_envs:
        if typ == 'begin':
            env_stack.append((name, pos))
        else:  # end
            if not env_stack:
                issues.append(ValidationIssue(
                    message=f"\\end{{{name}}} without matching \\begin",
                    severity=Severity.ERROR,
                    position=pos,
                    context=text[max(0, pos-20):pos+40]
                ))
            elif env_stack[-1][0] != name:
                issues.append(ValidationIssue(
                    message=f"Environment mismatch: \\begin{{{env_stack[-1][0]}}} closed by \\end{{{name}}}",
                    severity=Severity.ERROR,
                    position=pos,
                    context=text[max(0, pos-20):pos+40]
                ))
                env_stack.pop()
            else:
                env_stack.pop()
    
    # Check for unclosed environments
    for name, pos in env_stack:
        issues.append(ValidationIssue(
            message=f"Unclosed environment: \\begin{{{name}}}",
            severity=Severity.ERROR,
            position=pos,
            auto_fixable=True,
            fix_suggestion=f"Add \\end{{{name}}}"
        ))
    
    return issues


def check_common_errors(text: str) -> List[ValidationIssue]:
    """Check for common LaTeX errors."""
    issues = []
    
    # Check for broken commands (space after backslash)
    broken_cmd = re.findall(r'\\ [a-zA-Z]', text)
    if broken_cmd:
        issues.append(ValidationIssue(
            message=f"Possible broken command (space after \\): {broken_cmd[:3]}",
            severity=Severity.WARNING
        ))
    
    # Check for \\ at end of text without proper context
    if text.strip().endswith('\\\\'):
        issues.append(ValidationIssue(
            message="Text ends with \\\\ - may cause LaTeX error",
            severity=Severity.WARNING,
            auto_fixable=True,
            fix_suggestion="Remove trailing \\\\"
        ))
    
    # Check for empty braces that might be errors
    empty_braces = re.findall(r'\\[a-zA-Z]+\{\s*\}', text)
    if empty_braces:
        issues.append(ValidationIssue(
            message=f"Empty command arguments: {empty_braces[:3]}",
            severity=Severity.INFO
        ))
    
    # Check for potential encoding issues
    if '�' in text or '\ufffd' in text:
        issues.append(ValidationIssue(
            message="Replacement character found - encoding issue",
            severity=Severity.ERROR,
            auto_fixable=True,
            fix_suggestion="Remove or replace � characters"
        ))
    
    return issues


def validate_latex(text: str) -> ValidationResult:
    """
    Full LaTeX syntax validation.
    
    Returns ValidationResult with all issues found.
    """
    all_issues = []
    
    # Run all checks
    all_issues.extend(check_balanced_braces(text))
    all_issues.extend(check_balanced_brackets(text))
    all_issues.extend(check_math_delimiters(text))
    all_issues.extend(check_environments(text))
    all_issues.extend(check_common_errors(text))
    
    # Calculate score
    error_count = sum(1 for i in all_issues if i.severity == Severity.ERROR)
    warning_count = sum(1 for i in all_issues if i.severity == Severity.WARNING)
    
    # Score: start at 1.0, subtract for issues
    score = 1.0 - (error_count * 0.2) - (warning_count * 0.05)
    score = max(0.0, min(1.0, score))
    
    is_valid = error_count == 0
    
    return ValidationResult(
        is_valid=is_valid,
        issues=all_issues,
        score=score
    )


# =============================================================================
# TRANSLATION QUALITY VALIDATION
# =============================================================================

def check_translation_completeness(
    original: str,
    translated: str,
    length_tolerance: float = 0.5
) -> List[ValidationIssue]:
    """
    Check if translation seems complete.
    
    Flags if translation is much shorter than original.
    """
    issues = []
    
    orig_len = len(original)
    trans_len = len(translated)
    
    if orig_len == 0:
        return issues
    
    ratio = trans_len / orig_len
    
    # Translation shouldn't be too short
    if ratio < length_tolerance:
        issues.append(ValidationIssue(
            message=f"Translation seems incomplete: {ratio:.1%} of original length",
            severity=Severity.WARNING
        ))
    
    # Or too long (might have duplications)
    if ratio > 2.0:
        issues.append(ValidationIssue(
            message=f"Translation unusually long: {ratio:.1%} of original length",
            severity=Severity.INFO
        ))
    
    return issues


def check_formula_preservation(original: str, translated: str) -> List[ValidationIssue]:
    """
    Check if formulas were preserved during translation.
    
    Formulas should be unchanged in the translation.
    """
    issues = []
    
    # Extract formulas from original
    formula_patterns = [
        r'\$\$.*?\$\$',
        r'\$[^$]+\$',
        r'\\\[.*?\\\]',
        r'\\\(.*?\\\)',
        r'\\begin\{equation\*?\}.*?\\end\{equation\*?\}',
        r'\\begin\{align\*?\}.*?\\end\{align\*?\}',
    ]
    
    original_formulas = []
    for pattern in formula_patterns:
        original_formulas.extend(re.findall(pattern, original, re.DOTALL))
    
    # Check if formulas appear in translation
    missing_formulas = []
    for formula in original_formulas:
        if formula not in translated:
            # Check if it's a minor variation
            formula_normalized = re.sub(r'\s+', '', formula)
            trans_normalized = re.sub(r'\s+', '', translated)
            
            if formula_normalized not in trans_normalized:
                missing_formulas.append(formula[:50])
    
    if missing_formulas:
        issues.append(ValidationIssue(
            message=f"Formulas may have been modified: {len(missing_formulas)} potentially changed",
            severity=Severity.WARNING,
            context=str(missing_formulas[:3])
        ))
    
    return issues


def check_placeholder_restoration(text: str) -> List[ValidationIssue]:
    """Check if any placeholders were not restored."""
    issues = []
    
    # Look for placeholder patterns
    placeholder_patterns = [
        r'__LATEX_\d+__',
        r'__LATEX_CMD_\d+__',
        r'__GLOSS_[A-Z]+_\d+__',
        r'__TERM_.*?__',
        r'__NAME_.*?__',
    ]
    
    for pattern in placeholder_patterns:
        matches = re.findall(pattern, text)
        if matches:
            issues.append(ValidationIssue(
                message=f"Unrestored placeholders found: {matches[:5]}",
                severity=Severity.ERROR,
                auto_fixable=False
            ))
    
    return issues


def validate_translation(
    original: str,
    translated: str
) -> ValidationResult:
    """
    Validate translation quality.
    
    Checks:
    - LaTeX syntax
    - Completeness
    - Formula preservation
    - Placeholder restoration
    """
    all_issues = []
    
    # LaTeX validation
    latex_result = validate_latex(translated)
    all_issues.extend(latex_result.issues)
    
    # Translation-specific checks
    all_issues.extend(check_translation_completeness(original, translated))
    all_issues.extend(check_formula_preservation(original, translated))
    all_issues.extend(check_placeholder_restoration(translated))
    
    # Calculate score
    error_count = sum(1 for i in all_issues if i.severity == Severity.ERROR)
    warning_count = sum(1 for i in all_issues if i.severity == Severity.WARNING)
    
    score = 1.0 - (error_count * 0.2) - (warning_count * 0.05)
    score = max(0.0, min(1.0, score))
    
    is_valid = error_count == 0
    
    return ValidationResult(
        is_valid=is_valid,
        issues=all_issues,
        score=score
    )


# =============================================================================
# AUTO-FIX
# =============================================================================

def auto_fix_latex(text: str, result: ValidationResult) -> str:
    """
    Attempt to auto-fix simple LaTeX issues.
    
    Only fixes issues marked as auto_fixable.
    """
    fixed = text
    
    for issue in result.issues:
        if not issue.auto_fixable:
            continue
        
        # Remove replacement characters
        if '�' in issue.message or 'Replacement character' in issue.message:
            fixed = fixed.replace('�', '')
            fixed = fixed.replace('\ufffd', '')
        
        # Remove trailing \\
        if 'ends with \\\\' in issue.message:
            fixed = fixed.rstrip()
            if fixed.endswith('\\\\'):
                fixed = fixed[:-2]
    
    # Balance $ if odd
    single_dollars = len(re.findall(r'(?<!\\)(?<!\$)\$(?!\$)', fixed))
    if single_dollars % 2 != 0:
        fixed = fixed + '$'
    
    return fixed


# =============================================================================
# BATCH VALIDATION
# =============================================================================

def validate_document(
    blocks: List[Tuple[str, str]],  # List of (original, translated) pairs
    fix_errors: bool = True
) -> Tuple[List[ValidationResult], float]:
    """
    Validate an entire translated document.
    
    Args:
        blocks: List of (original, translated) tuples
        fix_errors: Whether to attempt auto-fixes
    
    Returns:
        Tuple of (results_per_block, overall_score)
    """
    results = []
    total_score = 0.0
    
    for original, translated in blocks:
        result = validate_translation(original, translated)
        
        if fix_errors and not result.is_valid:
            fixed = auto_fix_latex(translated, result)
            result.fixed_text = fixed
            # Re-validate fixed text
            fixed_result = validate_latex(fixed)
            if fixed_result.score > result.score:
                result.score = fixed_result.score
        
        results.append(result)
        total_score += result.score
    
    overall_score = total_score / len(blocks) if blocks else 1.0
    
    logger.info(f"Validated {len(blocks)} blocks, overall score: {overall_score:.2f}")
    
    return results, overall_score


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=== LaTeX Validation Test ===\n")
    
    # Test cases
    test_cases = [
        # Valid LaTeX
        ("Valid", r"This is $E=mc^2$ in text."),
        # Unbalanced braces
        ("Unbalanced braces", r"Test {unbalanced"),
        # Unbalanced math
        ("Unbalanced math", r"Formula $x + y without closing"),
        # Mismatched environments
        ("Mismatched env", r"\begin{equation}x=1\end{align}"),
        # Replacement char
        ("Encoding issue", "Text with � character"),
        # Placeholder not restored
        ("Placeholder", "Text with __LATEX_5__ placeholder"),
    ]
    
    for name, text in test_cases:
        print(f"\n### {name}")
        print(f"Input: {text[:50]}...")
        
        result = validate_latex(text)
        print(f"Valid: {result.is_valid}, Score: {result.score:.2f}")
        
        for issue in result.issues:
            print(f"  [{issue.severity.value}] {issue.message}")
    
    # Test translation validation
    print("\n\n### Translation Validation")
    original = r"The equation $E=mc^2$ shows energy-mass equivalence."
    translated = r"Die Gleichung $E=mc^2$ zeigt die Energie-Masse-Äquivalenz."
    
    result = validate_translation(original, translated)
    print(f"Original: {original}")
    print(f"Translated: {translated}")
    print(f"Valid: {result.is_valid}, Score: {result.score:.2f}")
    
    print("\n✅ Validation module ready")
