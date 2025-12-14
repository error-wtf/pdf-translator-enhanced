#!/usr/bin/env python3
"""
End-to-End Perfection Test for PDF Translator
Tests real translation quality with scientific content
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

from formula_isolator import extract_and_protect, normalize_output, assert_no_corruption
from quality_validator import validate_translation, assert_quality
from table_handler import should_translate_cell, is_numeric_content

# =============================================================================
# TEST DATA - Real scientific content
# =============================================================================

SCIENTIFIC_TEXTS = [
    {
        "name": "Schrödinger Equation",
        "original": "The time-dependent Schrödinger equation iħ∂Ψ/∂t = ĤΨ describes quantum evolution [1].",
        "must_preserve": ["iħ∂Ψ/∂t", "ĤΨ", "[1]", "ħ", "Ψ"],
    },
    {
        "name": "Einstein Field Equations",
        "original": "Einstein's field equations Rμν - ½gμνR = 8πG/c⁴ Tμν relate spacetime curvature to energy [2, 3].",
        "must_preserve": ["Rμν", "gμν", "8πG/c⁴", "Tμν", "[2, 3]"],
    },
    {
        "name": "Scientific Notation",
        "original": "The Planck length ℓP = 1.616 × 10⁻³⁵ m and Planck time tP = 5.39 × 10⁻⁴⁴ s are fundamental (see Fig. 1).",
        "must_preserve": ["ℓP", "1.616 × 10⁻³⁵ m", "5.39 × 10⁻⁴⁴ s", "Fig. 1"],
    },
    {
        "name": "Greek Letters",
        "original": "Parameters α, β, and γ determine the coupling strength Γ = αβ/γ as shown in Eq. (5).",
        "must_preserve": ["α", "β", "γ", "Γ", "αβ/γ", "Eq. (5)"],
    },
    {
        "name": "Subscripts/Superscripts",
        "original": "Energy levels E₀, E₁, E₂ follow En = -13.6 eV/n² for hydrogen (Table 2).",
        "must_preserve": ["E₀", "E₁", "E₂", "En", "n²", "-13.6 eV", "Table 2"],
    },
    {
        "name": "Math Operators",
        "original": "The gradient ∇φ and Laplacian ∇²φ ≈ ∂²φ/∂x² satisfy ∫∇²φ dV = ∮∇φ·dA.",
        "must_preserve": ["∇φ", "∇²φ", "∂²φ/∂x²", "∫", "∮", "≈"],
    },
    {
        "name": "Footnotes",
        "original": "This result was first derived by Feynman¹ and later confirmed by Schwinger²*.",
        "must_preserve": ["¹", "²", "*"],
    },
    {
        "name": "Units",
        "original": "The frequency f = 10.5 GHz corresponds to wavelength λ = 2.86 cm at T = 300 K.",
        "must_preserve": ["10.5 GHz", "2.86 cm", "300 K"],
    },
]

# =============================================================================
# SIMULATED TRANSLATION (mimics LLM behavior)
# =============================================================================

def simulate_llm_translation(protected_text):
    """Simulate LLM translation - uppercase words, keep placeholders."""
    import re
    
    # Split into words, translate non-placeholders
    result = []
    for word in protected_text.split():
        if '⟦' in word or re.match(r'^[0-9.,×·⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹]+$', word):
            result.append(word)  # Keep as-is
        elif word.lower() in ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'for', 'and', 'or']:
            # Translate common words
            translations = {
                'the': 'die', 'a': 'ein', 'an': 'ein', 'is': 'ist', 'are': 'sind',
                'was': 'war', 'were': 'waren', 'to': 'zu', 'of': 'von',
                'in': 'in', 'for': 'für', 'and': 'und', 'or': 'oder'
            }
            result.append(translations.get(word.lower(), word))
        else:
            result.append(word)  # Keep scientific terms
    
    return ' '.join(result)

# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_formula_protection():
    """Test that formulas are protected during translation."""
    print("\n" + "="*60)
    print("TEST 1: Formula Protection")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for test in SCIENTIFIC_TEXTS:
        print(f"\n### {test['name']}")
        print(f"Original: {test['original'][:60]}...")
        
        # Extract and protect
        protected, restore = extract_and_protect(test['original'])
        print(f"Protected: {protected[:60]}...")
        
        # Simulate translation
        translated = simulate_llm_translation(protected)
        
        # Restore
        restored = restore(translated)
        restored = normalize_output(restored, mode="unicode")
        print(f"Restored: {restored[:60]}...")
        
        # Check preservation
        all_preserved = True
        missing = []
        for item in test['must_preserve']:
            if item not in restored:
                all_preserved = False
                missing.append(item)
        
        if all_preserved:
            print(f"✅ All {len(test['must_preserve'])} items preserved")
            passed += 1
        else:
            print(f"❌ MISSING: {missing}")
            failed += 1
    
    return passed, failed

def test_corruption_detection():
    """Test that corruption is detected."""
    print("\n" + "="*60)
    print("TEST 2: Corruption Detection")
    print("="*60)
    
    tests = [
        ("Clean text with α β γ", True, "Clean Unicode"),
        ("Corrupted ?? text", False, "Double ??"),
        ("Has \ufffd char", False, "Replacement char"),
        ("Unrestored ⟦F_ABC123_0⟧", False, "Placeholder"),
        ("HTML <sup>1</sup>", True, "HTML (warning only)"),
    ]
    
    passed = 0
    for text, expected_clean, name in tests:
        result = assert_no_corruption(text)
        status = "✅" if result == expected_clean else "❌"
        print(f"{status} {name}: {'clean' if result else 'corrupt'} (expected: {'clean' if expected_clean else 'corrupt'})")
        if result == expected_clean:
            passed += 1
    
    return passed, len(tests) - passed

def test_quality_validation():
    """Test quality validation scoring."""
    print("\n" + "="*60)
    print("TEST 3: Quality Validation")
    print("="*60)
    
    tests = [
        {
            "name": "Good translation",
            "original": "The Schrödinger equation iħ∂Ψ/∂t = ĤΨ is fundamental [1].",
            "translated": "Die Schrödinger-Gleichung iħ∂Ψ/∂t = ĤΨ ist fundamental [1].",
            "expect_pass": True,
        },
        {
            "name": "Lost formulas",
            "original": "Energy E = mc² and momentum p = mv are related.",
            "translated": "Energie und Impuls sind verwandt.",
            "expect_pass": False,
        },
        {
            "name": "Corrupted",
            "original": "Wave function Ψ describes state.",
            "translated": "Wellenfunktion ?? beschreibt Zustand.",
            "expect_pass": False,
        },
    ]
    
    passed = 0
    for test in tests:
        result = validate_translation(test['original'], test['translated'])
        status = "✅" if result.passed == test['expect_pass'] else "❌"
        print(f"{status} {test['name']}: score={result.score:.1%}, passed={result.passed}")
        if result.issues:
            for issue in result.issues[:2]:
                print(f"   → {issue}")
        if result.passed == test['expect_pass']:
            passed += 1
    
    return passed, len(tests) - passed

def test_table_handling():
    """Test table cell handling."""
    print("\n" + "="*60)
    print("TEST 4: Table Handling")
    print("="*60)
    
    tests = [
        ("12.345", False, "Pure number"),
        ("95.5%", False, "Percentage"),
        ("$100.00", False, "Currency"),
        ("1.5 × 10⁻³", False, "Scientific notation"),
        ("Description", True, "Text header"),
        ("Energy level", True, "Text content"),
        ("x", False, "Single char"),
        ("", False, "Empty"),
    ]
    
    passed = 0
    for text, should_translate, name in tests:
        result = should_translate_cell(text)
        status = "✅" if result == should_translate else "❌"
        print(f"{status} {name}: '{text}' → {'translate' if result else 'keep'}")
        if result == should_translate:
            passed += 1
    
    return passed, len(tests) - passed

def test_end_to_end():
    """Full end-to-end translation test."""
    print("\n" + "="*60)
    print("TEST 5: End-to-End Pipeline")
    print("="*60)
    
    original = """
    Abstract: We present a unified theory of quantum gravity based on the 
    Schrödinger equation iħ∂Ψ/∂t = ĤΨ and Einstein's field equations 
    Rμν - ½gμνR = 8πG/c⁴ Tμν. Our results show that at the Planck scale 
    (ℓP = 1.616 × 10⁻³⁵ m), quantum effects dominate [1-5]. See Fig. 1 
    and Table 2 for numerical results. The coupling constants α ≈ 1/137 
    and β = 0.511 MeV/c² determine the energy spectrum E₀, E₁, E₂.
    """
    
    print("Original text (truncated):")
    print(original[:200] + "...")
    
    # Step 1: Extract and protect
    protected, restore = extract_and_protect(original)
    placeholder_count = protected.count('⟦')
    print(f"\n→ Extracted {placeholder_count} formulas/references")
    
    # Step 2: Simulate translation
    translated = simulate_llm_translation(protected)
    
    # Step 3: Restore
    restored = restore(translated)
    restored = normalize_output(restored, mode="unicode")
    
    # Step 4: Validate
    result = validate_translation(original, restored)
    
    print(f"\n→ Quality Score: {result.score:.1%}")
    print(f"→ Passed: {result.passed}")
    print(f"→ Stats: {result.stats}")
    
    if result.issues:
        print("→ Issues:")
        for issue in result.issues:
            print(f"   • {issue}")
    
    # Check critical items
    critical_items = ["iħ∂Ψ/∂t", "ĤΨ", "Rμν", "8πG/c⁴", "1.616 × 10⁻³⁵", "[1-5]", "Fig. 1", "α", "E₀"]
    preserved = sum(1 for item in critical_items if item in restored)
    
    print(f"\n→ Critical items preserved: {preserved}/{len(critical_items)}")
    
    return (1 if preserved >= 7 else 0), (0 if preserved >= 7 else 1)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("PDF TRANSLATOR - PERFECTION TEST")
    print("="*60)
    
    total_passed = 0
    total_failed = 0
    
    # Run all tests
    p, f = test_formula_protection()
    total_passed += p; total_failed += f
    
    p, f = test_corruption_detection()
    total_passed += p; total_failed += f
    
    p, f = test_quality_validation()
    total_passed += p; total_failed += f
    
    p, f = test_table_handling()
    total_passed += p; total_failed += f
    
    p, f = test_end_to_end()
    total_passed += p; total_failed += f
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total: {total_passed} passed, {total_failed} failed")
    
    if total_failed == 0:
        print("\n🎉 PERFECTION ACHIEVED! All tests passed!")
    else:
        print(f"\n⚠️  {total_failed} tests need attention")
    
    sys.exit(0 if total_failed == 0 else 1)
