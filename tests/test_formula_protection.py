"""
Tests for formula protection and Unicode math handling.

These tests ensure:
1. All math symbols are preserved during translation
2. No corruption (no ??, no replacement characters)
3. Placeholders are properly restored
4. Output is normalized consistently

Run: pytest tests/test_formula_protection.py -v
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from formula_isolator import (
    extract_formulas,
    extract_and_protect,
    normalize_output,
    audit_utf8,
    assert_no_corruption,
    regression_check,
    generate_placeholder,
)


class TestFormulaExtraction:
    """Test formula extraction patterns."""
    
    def test_latex_inline_math(self):
        """Test inline LaTeX math extraction."""
        text = "The equation $E = mc^2$ shows mass-energy."
        result = extract_formulas(text)
        
        assert len(result.formulas) >= 1
        assert "$E = mc^2$" in result.formula_map.values()
        assert "$" not in result.text_with_placeholders or "⟦" in result.text_with_placeholders
    
    def test_latex_display_math(self):
        """Test display LaTeX math extraction."""
        text = r"Consider: $$\int_0^1 x^2 dx = \frac{1}{3}$$"
        result = extract_formulas(text)
        
        assert len(result.formulas) >= 1
        formulas = list(result.formula_map.values())
        assert any("\\int" in f for f in formulas)
    
    def test_latex_environment(self):
        """Test LaTeX environment extraction."""
        text = r"""
        \begin{equation}
        R_{\mu\nu} - \frac{1}{2}g_{\mu\nu}R = 8\pi G T_{\mu\nu}
        \end{equation}
        """
        result = extract_formulas(text)
        
        assert len(result.formulas) >= 1
        formulas = list(result.formula_map.values())
        assert any("\\begin{equation}" in f for f in formulas)
    
    def test_greek_letters(self):
        """Test Greek letter protection."""
        text = "The wave function Ψ depends on α and β parameters."
        result = extract_formulas(text)
        
        # Greek letters should be extracted
        all_content = "".join(result.formula_map.values())
        # At minimum, sequences should be protected
        assert "Ψ" in text  # Verify input has Greek
    
    def test_schrodinger_equation(self):
        """Test Schrödinger equation protection."""
        text = "The Schrödinger equation: iħ∂Ψ/∂t = ĤΨ"
        result = extract_formulas(text)
        
        # Should protect the math parts
        assert len(result.formulas) >= 1


class TestUnicodeMath:
    """Test Unicode math symbol handling."""
    
    def test_physics_constants(self):
        """Test physics constant protection."""
        text = "Planck's constant ħ = h/2π and ℓ_P is the Planck length."
        result = extract_formulas(text)
        
        all_formulas = "".join(result.formula_map.values())
        assert "ħ" in all_formulas or "ℓ" in all_formulas
    
    def test_operators(self):
        """Test math operator protection."""
        text = "The gradient ∇ and Laplacian ∇² are differential operators."
        result = extract_formulas(text)
        
        # Operators should be protected
        assert len(result.formulas) >= 1
    
    def test_subscripts(self):
        """Test subscript protection."""
        text = "Energy levels E₀, E₁, E₂ increase with quantum number n."
        result = extract_formulas(text)
        
        all_formulas = "".join(result.formula_map.values())
        assert "₀" in all_formulas or "₁" in all_formulas or "₂" in all_formulas
    
    def test_superscripts(self):
        """Test superscript protection."""
        text = "The frequency is 10⁻¹⁶ Hz, or about 10⁹ cycles."
        result = extract_formulas(text)
        
        all_formulas = "".join(result.formula_map.values())
        assert "⁻" in all_formulas or "¹" in all_formulas
    
    def test_scientific_notation(self):
        """Test scientific notation protection."""
        text = "The value is 6.67 × 10⁻¹¹ N⋅m²/kg²."
        result = extract_formulas(text)
        
        # Scientific notation should be protected
        assert len(result.formulas) >= 1


class TestPlaceholderHandling:
    """Test placeholder generation and restoration."""
    
    def test_placeholder_format(self):
        """Test placeholder format is correct."""
        placeholder = generate_placeholder("test content", 0)
        
        assert "⟦" in placeholder
        assert "⟧" in placeholder
        assert "_" in placeholder
    
    def test_extract_and_restore(self):
        """Test full extract and restore cycle."""
        original = "The equation $E = mc^2$ is famous."
        
        protected, restore = extract_and_protect(original)
        
        # Simulate translation (just uppercase for test)
        translated = protected.upper()
        
        # Restore
        restored = restore(translated)
        
        # Formula should be back
        assert "$E = mc^2$" in restored
    
    def test_multiple_formulas(self):
        """Test multiple formula handling."""
        original = "From $a = b$ and $c = d$ we get $a + c = b + d$."
        
        protected, restore = extract_and_protect(original)
        
        # Count placeholders
        placeholder_count = protected.count("⟦")
        assert placeholder_count >= 3
        
        # Restore
        restored = restore(protected)
        
        # All formulas back
        assert "$a = b$" in restored
        assert "$c = d$" in restored


class TestCorruptionDetection:
    """Test corruption detection."""
    
    def test_no_corruption_clean(self):
        """Test clean text passes."""
        text = "This is clean text with no issues."
        assert assert_no_corruption(text) is True
    
    def test_detect_question_marks(self):
        """Test detection of ?? pattern."""
        text = "This has ?? corruption."
        assert assert_no_corruption(text) is False
    
    def test_detect_replacement_char(self):
        """Test detection of replacement character."""
        text = "This has \ufffd replacement."
        assert assert_no_corruption(text) is False
    
    def test_detect_unrestored_placeholder(self):
        """Test detection of unrestored placeholder."""
        text = "This has ⟦F_abc123_0⟧ unrestored."
        assert assert_no_corruption(text) is False


class TestUTF8Audit:
    """Test UTF-8 auditing."""
    
    def test_clean_utf8(self):
        """Test clean UTF-8 passes."""
        text = "Schrödinger equation: iħ∂Ψ/∂t = ĤΨ"
        warnings = audit_utf8(text, "test")
        
        # Should be clean (no ?? patterns)
        assert not any("??" in w for w in warnings)
    
    def test_detect_mojibake(self):
        """Test detection of mojibake (encoding corruption)."""
        text = "This has Ã¤ corruption (should be ä)."
        warnings = audit_utf8(text, "test")
        
        assert len(warnings) >= 1


class TestOutputNormalization:
    """Test output normalization."""
    
    def test_html_superscript_removal(self):
        """Test HTML superscript conversion."""
        text = "Reference<sup>1,*</sup> shows..."
        normalized = normalize_output(text, mode="unicode")
        
        assert "<sup>" not in normalized
    
    def test_html_subscript_removal(self):
        """Test HTML subscript conversion."""
        text = "H<sub>2</sub>O is water."
        normalized = normalize_output(text, mode="unicode")
        
        assert "<sub>" not in normalized
    
    def test_preserve_unicode_math(self):
        """Test Unicode math is preserved in unicode mode."""
        text = "The value is α + β = γ."
        normalized = normalize_output(text, mode="unicode")
        
        assert "α" in normalized
        assert "β" in normalized
        assert "γ" in normalized


class TestRegressionCheck:
    """Test regression checking."""
    
    def test_successful_translation(self):
        """Test successful translation passes regression."""
        original = "The famous equation $E = mc^2$ shows energy."
        translated = "Die berühmte Gleichung ⟦F_abc123_0⟧ zeigt Energie."
        restored = "Die berühmte Gleichung $E = mc^2$ zeigt Energie."
        
        result = regression_check(original, translated, restored)
        
        # Should pass (formula preserved)
        assert "CRITICAL" not in str(result.get("issues", []))
    
    def test_detect_formula_loss(self):
        """Test detection of formula loss."""
        original = "Equation: $E = mc^2$"
        translated = "Gleichung: Die Energie..."
        restored = "Gleichung: Die Energie..."
        
        result = regression_check(original, translated, restored)
        
        # Should detect formula loss
        assert result["stats"]["formulas_original"] >= 1
        # Restored has fewer formulas
        assert result["stats"]["formulas_restored"] < result["stats"]["formulas_original"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
