"""Tests for quality_validator module."""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quality_validator import (
    count_formulas, count_references, detect_corruption,
    validate_translation, assert_quality
)

class TestFormulaCount:
    def test_latex_formulas(self):
        text = "The equation $E = mc^2$ and $F = ma$ are famous."
        assert count_formulas(text) >= 2
    
    def test_greek_letters(self):
        text = "Parameters α, β, γ are used."
        assert count_formulas(text) >= 3
    
    def test_math_operators(self):
        text = "We have ∂f/∂x ≈ ∇f and ∫dx"
        assert count_formulas(text) >= 4
    
    def test_subscripts_superscripts(self):
        text = "Energy E₀ and 10⁻¹⁶"
        assert count_formulas(text) >= 3

class TestReferenceCount:
    def test_numeric_refs(self):
        text = "See references [1], [2, 3], and [4-6]."
        assert count_references(text) >= 3
    
    def test_figure_refs(self):
        text = "As shown in Fig. 1 and Figure 2."
        assert count_references(text) >= 2
    
    def test_equation_refs(self):
        text = "From Eq. (1) we derive Eq. (2)."
        assert count_references(text) >= 2

class TestCorruptionDetection:
    def test_clean_text(self):
        text = "This is clean text with α and β."
        issues = detect_corruption(text)
        critical = [i for s, i in issues if s == "critical"]
        assert len(critical) == 0
    
    def test_replacement_char(self):
        text = "This has \ufffd corruption."
        issues = detect_corruption(text)
        assert any("Replacement" in msg for _, msg in issues)
    
    def test_double_question_marks(self):
        text = "Schr??dinger equation"
        issues = detect_corruption(text)
        assert any("??" in msg for _, msg in issues)
    
    def test_unrestored_placeholder(self):
        text = "The formula ⟦F_ABC123_0⟧ was not restored."
        issues = detect_corruption(text)
        assert any("placeholder" in msg.lower() for _, msg in issues)
    
    def test_html_tags(self):
        text = "Value is 10<sup>-3</sup>"
        issues = detect_corruption(text)
        assert any("HTML" in msg for _, msg in issues)

class TestValidation:
    def test_good_translation(self):
        original = "The Schrödinger equation iħ∂Ψ/∂t = ĤΨ is fundamental [1]."
        translated = "Die Schrödinger-Gleichung iħ∂Ψ/∂t = ĤΨ ist fundamental [1]."
        result = validate_translation(original, translated)
        assert result.passed
        assert result.score >= 0.8
    
    def test_formula_loss(self):
        original = "Energy E = mc² and momentum p = mv."
        translated = "Energie und Impuls."  # Lost formulas
        result = validate_translation(original, translated)
        assert result.score < 1.0
    
    def test_corrupted_translation(self):
        original = "The wave function Ψ describes the state."
        translated = "Die Wellenfunktion ?? beschreibt den Zustand."
        result = validate_translation(original, translated)
        assert not result.passed

class TestAssertQuality:
    def test_good_returns_translated(self):
        original = "Hello world"
        translated = "Hallo Welt"
        result = assert_quality(original, translated)
        assert result == translated
    
    def test_bad_returns_original(self):
        original = "Hello α β γ"
        translated = "Hallo ??"  # Corrupted
        result = assert_quality(original, translated)
        assert result == original

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
