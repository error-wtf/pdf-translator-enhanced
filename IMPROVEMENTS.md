# PDF Translator - Verbesserungen für Publishing-Grade Qualität

## Datum: 2025-12-14

## Problem-Analyse (Screenshot-basiert)

| Problem | Symptom | Ursache |
|---------|---------|---------|
| Unicode-Symbole zerstört | `ħ ∇ Ψ` → `??` | Kein UTF-8 Schutz, Font kann Symbole nicht |
| Formeln verändert | `iħ∂Ψ/∂t = ĤΨ` korrupt | Math nicht vor Übersetzung geschützt |
| HTML im Output | `<sup>1,*</sup>` sichtbar | LLM-Output nicht normalisiert |
| LaTeX/Unicode gemischt | `$\alpha$ α` | Keine einheitliche Repräsentation |

---

## Implementierte Fixes

### 1. `formula_isolator.py` - Erweitert

**Neue Funktionen:**

```python
# Unicode-Math Patterns (NEU)
UNICODE_MATH_PATTERNS = [
    # Schrödinger-Gleichung
    (r'[iℏħ]\s*[∂∇]\s*[ΨΦψφ]...', 'schrodinger'),
    # Griechische Buchstaben-Sequenzen
    (r'[αβγδ...ΨΩ]{2,}', 'greek_seq'),
    # Physik-Konstanten
    (r'[ℏħℓ℘ℜℑ]', 'physics_const'),
    # Einheiten
    (r'\d+\.?\d*\s*(?:Hz|kHz|MHz|...)', 'unit_expr'),
]

# UTF-8 Enforcement
def audit_utf8(text, source) -> List[str]
def ensure_utf8_safe(text) -> str

# Output Normalisierung
def normalize_output(text, mode="unicode") -> str

# Regression Checks
def regression_check(original, translated, restored) -> Dict
def assert_no_corruption(text) -> bool
```

### 2. `latex_build.py` - `sanitize_for_latex()`

**Vorher:** 8 Zeichen gemappt
**Nachher:** 100+ Zeichen gemappt

- Griechische Buchstaben (Upper + Lower)
- Math-Operatoren (∇, ∂, ∫, ∑, ∏, √, ∞, ...)
- Subscripts (₀-₉, ₐ, ₑ, ₙ, ...)
- Superscripts (⁰-⁹, ⁺, ⁻, ⁿ, ...)
- Physik-Konstanten (ℏ, ħ, ℓ, ...)
- Pfeile (→, ←, ⇒, ...)

### 3. `pdf_overlay_translator.py` - Integration

```python
# VORHER
translated = translate_text_ollama(block.text, ...)

# NACHHER
protected_text, restore_func = extract_and_protect(block.text)
translated_protected = translate_text_ollama(protected_text, ...)
translated = restore_func(translated_protected)
translated = normalize_output(translated, mode="unicode")
if not assert_no_corruption(translated):
    translated = block.text  # Fallback
```

### 4. `docx_translator.py` - Integration

Gleiche Änderungen wie `pdf_overlay_translator.py`:
- Formula protection vor LLM
- Restore + Normalize nach LLM
- Corruption detection mit Fallback

---

## Pipeline-Flow (NEU)

```
PDF/DOCX Input
    │
    ▼
┌─────────────────────────────────┐
│ 1. Text-Block Extraktion        │
│    (PyMuPDF mit BBox)           │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ 2. MATH PROTECTION              │  ← NEU
│    extract_and_protect()        │
│    - LaTeX: $...$, \[...\]      │
│    - Unicode: ħ, ∇, Ψ, α, β     │
│    - Units: 10 MHz, 5 eV        │
│    → Placeholders: ⟦F_abc123⟧  │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ 3. LLM Translation              │
│    (nur Text, keine Formeln)    │
│    Prompt: "Keep ⟦...⟧ unchanged" │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ 4. RESTORE + NORMALIZE          │  ← NEU
│    restore_func()               │
│    normalize_output()           │
│    - HTML → Unicode             │
│    - Konsistente Repräsentation │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ 5. VALIDATION                   │  ← NEU
│    assert_no_corruption()       │
│    - Keine ?? Patterns          │
│    - Keine U+FFFD               │
│    - Alle Placeholders restored │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ 6. PDF Rebuild                  │
│    - Unicode Font (Cambria)     │
│    - Original BBox Position     │
│    - sanitize_for_latex()       │
└─────────────────────────────────┘
    │
    ▼
Output PDF/DOCX
```

---

## Commits

| Hash | Beschreibung |
|------|--------------|
| `6ca7395` | `sanitize_for_latex` mit 100+ Unicode-Math Mapping |
| `848727c` | Unicode math protection, UTF-8 enforcement, regression checks |
| `aa925d6` | Integration in `pdf_overlay_translator.py` |
| `ac9b29b` | Integration in `docx_translator.py` |

---

## Was jetzt funktionieren sollte

| Eingabe | Ausgabe |
|---------|---------|
| `ħ` | `ħ` (geschützt) oder `$\hbar$` (LaTeX) |
| `∇²Ψ` | `∇²Ψ` (geschützt) |
| `iħ∂Ψ/∂t = ĤΨ` | Formel unverändert |
| `<sup>1,*</sup>` | `¹⸴∗` oder entfernt |
| `10⁻¹⁶ Hz` | `10⁻¹⁶ Hz` (geschützt) |

---

## Noch offen (für 100% Perfektion)

1. **Layout-Drift**: Wenn übersetzte Texte länger sind als Original
2. **Font-Embedding**: Sicherstellen dass Unicode-Font immer verfügbar
3. **Tabellen**: Spezialbehandlung für Tabellen-Zellen
4. **Bilder mit Text**: OCR-Fallback für Text in Bildern
5. **Hyphenation**: Silbentrennung für lange Wörter

---

## Test-Befehl

```bash
cd E:\clone\pdf-translator
python gradio_app.py
# → PDF mit Schrödinger-Gleichung hochladen
# → Auf Deutsch übersetzen
# → Prüfen: ħ, ∇, Ψ erhalten?
```
