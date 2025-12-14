#!/usr/bin/env python3
"""Real LLM test - Tests actual translation with Ollama"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

import requests
import json
from formula_isolator import extract_and_protect, normalize_output, assert_no_corruption
from quality_validator import validate_translation

def translate_with_ollama(text, model="qwen2.5:7b", target="German"):
    """Real LLM translation via Ollama."""
    # Protect formulas first
    protected, restore = extract_and_protect(text)
    
    prompt = f"""Translate to {target}. Keep ALL ⟦...⟧ placeholders EXACTLY unchanged.
Keep all math symbols: α β γ ∇ ∂ ∫ ≈ × ⁻ ¹ ² ³ unchanged.

Text: {protected}

Translation:"""
    
    try:
        resp = requests.post("http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60)
        if resp.status_code == 200:
            result = resp.json().get("response", "").strip()
            # Restore formulas
            restored = restore(result)
            restored = normalize_output(restored, mode="unicode")
            return restored, True
    except Exception as e:
        return str(e), False
    return text, False

# Test cases
TESTS = [
    {
        "name": "Schrödinger Equation",
        "text": "The Schrödinger equation iħ∂Ψ/∂t = ĤΨ describes quantum mechanics [1].",
        "must_have": ["iħ∂Ψ/∂t", "ĤΨ", "[1]"],
    },
    {
        "name": "Greek Letters",
        "text": "Parameters α, β, γ determine the coupling constant Γ = αβ/γ.",
        "must_have": ["α", "β", "γ", "Γ"],
    },
    {
        "name": "Scientific Notation", 
        "text": "The value is 1.616 × 10⁻³⁵ m at frequency 10.5 GHz.",
        "must_have": ["1.616 × 10⁻³⁵", "10.5 GHz"],
    },
    {
        "name": "References",
        "text": "See Fig. 1 and Eq. (3) for details on Table 2.",
        "must_have": ["Fig. 1", "Eq. (3)", "Table 2"],
    },
]

print("="*60)
print("REAL LLM TRANSLATION TEST (Ollama)")
print("="*60)

passed = 0
failed = 0

for test in TESTS:
    print(f"\n### {test['name']}")
    print(f"Original: {test['text']}")
    
    translated, success = translate_with_ollama(test['text'])
    
    if not success:
        print(f"❌ LLM Error: {translated}")
        failed += 1
        continue
    
    print(f"Translated: {translated}")
    
    # Check preservation
    missing = [item for item in test['must_have'] if item not in translated]
    
    # Validate quality
    result = validate_translation(test['text'], translated)
    
    if not missing and result.passed:
        print(f"✅ PERFECT: All {len(test['must_have'])} items preserved, score={result.score:.0%}")
        passed += 1
    else:
        if missing:
            print(f"❌ MISSING: {missing}")
        if not result.passed:
            print(f"❌ Quality issues: {result.issues}")
        failed += 1

print("\n" + "="*60)
print(f"RESULT: {passed}/{passed+failed} tests passed")
if failed == 0:
    print("🎉 PERFEKTE QUALITÄT BESTÄTIGT!")
else:
    print(f"⚠️ {failed} tests need improvement")
print("="*60)
