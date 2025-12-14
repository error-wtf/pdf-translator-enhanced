"""
Languages - Complete language support configuration

Supports 25+ languages with:
- Native names and codes
- Writing direction (LTR/RTL)
- Script type
- Font recommendations
- Hyphenation patterns
- Scientific terminology

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class WritingDirection(Enum):
    LTR = "ltr"  # Left to right
    RTL = "rtl"  # Right to left


class ScriptType(Enum):
    LATIN = "latin"
    CYRILLIC = "cyrillic"
    GREEK = "greek"
    ARABIC = "arabic"
    HEBREW = "hebrew"
    CJK = "cjk"  # Chinese, Japanese, Korean
    DEVANAGARI = "devanagari"
    THAI = "thai"


@dataclass
class LanguageConfig:
    """Complete configuration for a target language."""
    code: str              # ISO 639-1 code
    name: str              # English name
    native_name: str       # Name in the language itself
    direction: WritingDirection
    script: ScriptType
    
    # Font configuration
    recommended_fonts: List[str]
    fallback_fonts: List[str]
    
    # Typography
    uses_hyphenation: bool = True
    quote_chars: Tuple[str, str] = ('"', '"')
    decimal_separator: str = "."
    thousands_separator: str = ","
    
    # Scientific writing style
    formal_register: str = ""  # Brief style note
    keep_english_terms: List[str] = None  # Terms to keep in English
    
    def __post_init__(self):
        if self.keep_english_terms is None:
            self.keep_english_terms = []


# =============================================================================
# LANGUAGE DEFINITIONS
# =============================================================================

LANGUAGES: Dict[str, LanguageConfig] = {
    # === WESTERN EUROPEAN ===
    "German": LanguageConfig(
        code="de",
        name="German",
        native_name="Deutsch",
        direction=WritingDirection.LTR,
        script=ScriptType.LATIN,
        recommended_fonts=["DejaVu Sans", "Liberation Sans", "Arial"],
        fallback_fonts=["Helvetica", "sans-serif"],
        quote_chars=("„", """),
        decimal_separator=",",
        thousands_separator=".",
        formal_register="Use formal 'Sie', compound nouns, verb-final in subordinates",
        keep_english_terms=["Software", "Hardware", "Computer", "Online"],
    ),
    
    "French": LanguageConfig(
        code="fr",
        name="French",
        native_name="Français",
        direction=WritingDirection.LTR,
        script=ScriptType.LATIN,
        recommended_fonts=["DejaVu Sans", "Liberation Sans", "Arial"],
        fallback_fonts=["Helvetica", "sans-serif"],
        quote_chars=("«\u00A0", "\u00A0»"),  # With non-breaking spaces
        decimal_separator=",",
        thousands_separator="\u00A0",  # Non-breaking space
        formal_register="Use 'nous' or impersonal, space before : ; ? !",
    ),
    
    "Spanish": LanguageConfig(
        code="es",
        name="Spanish",
        native_name="Español",
        direction=WritingDirection.LTR,
        script=ScriptType.LATIN,
        recommended_fonts=["DejaVu Sans", "Liberation Sans", "Arial"],
        fallback_fonts=["Helvetica", "sans-serif"],
        quote_chars=("«", "»"),
        decimal_separator=",",
        thousands_separator=".",
        formal_register="Include ¿ and ¡, formal register for academic",
    ),
    
    "Italian": LanguageConfig(
        code="it",
        name="Italian",
        native_name="Italiano",
        direction=WritingDirection.LTR,
        script=ScriptType.LATIN,
        recommended_fonts=["DejaVu Sans", "Liberation Sans", "Arial"],
        fallback_fonts=["Helvetica", "sans-serif"],
        quote_chars=("«", "»"),
        decimal_separator=",",
        thousands_separator=".",
        formal_register="Formal register, proper article agreement",
    ),
    
    "Portuguese": LanguageConfig(
        code="pt",
        name="Portuguese",
        native_name="Português",
        direction=WritingDirection.LTR,
        script=ScriptType.LATIN,
        recommended_fonts=["DejaVu Sans", "Liberation Sans", "Arial"],
        fallback_fonts=["Helvetica", "sans-serif"],
        quote_chars=("«", "»"),
        decimal_separator=",",
        thousands_separator=".",
        formal_register="Neutral Brazilian/European, formal register",
    ),
    
    "Dutch": LanguageConfig(
        code="nl",
        name="Dutch",
        native_name="Nederlands",
        direction=WritingDirection.LTR,
        script=ScriptType.LATIN,
        recommended_fonts=["DejaVu Sans", "Liberation Sans", "Arial"],
        fallback_fonts=["Helvetica", "sans-serif"],
        quote_chars=("„", """),
        decimal_separator=",",
        thousands_separator=".",
    ),
    
    # === NORDIC ===
    "Swedish": LanguageConfig(
        code="sv",
        name="Swedish",
        native_name="Svenska",
        direction=WritingDirection.LTR,
        script=ScriptType.LATIN,
        recommended_fonts=["DejaVu Sans", "Liberation Sans", "Arial"],
        fallback_fonts=["Helvetica", "sans-serif"],
        quote_chars=(""", """),
        decimal_separator=",",
        thousands_separator="\u00A0",
    ),
    
    "Norwegian": LanguageConfig(
        code="no",
        name="Norwegian",
        native_name="Norsk",
        direction=WritingDirection.LTR,
        script=ScriptType.LATIN,
        recommended_fonts=["DejaVu Sans", "Liberation Sans", "Arial"],
        fallback_fonts=["Helvetica", "sans-serif"],
        quote_chars=("«", "»"),
        decimal_separator=",",
        thousands_separator="\u00A0",
    ),
    
    "Danish": LanguageConfig(
        code="da",
        name="Danish",
        native_name="Dansk",
        direction=WritingDirection.LTR,
        script=ScriptType.LATIN,
        recommended_fonts=["DejaVu Sans", "Liberation Sans", "Arial"],
        fallback_fonts=["Helvetica", "sans-serif"],
        quote_chars=("»", "«"),  # Reversed!
        decimal_separator=",",
        thousands_separator=".",
    ),
    
    "Finnish": LanguageConfig(
        code="fi",
        name="Finnish",
        native_name="Suomi",
        direction=WritingDirection.LTR,
        script=ScriptType.LATIN,
        recommended_fonts=["DejaVu Sans", "Liberation Sans", "Arial"],
        fallback_fonts=["Helvetica", "sans-serif"],
        quote_chars=(""", """),
        decimal_separator=",",
        thousands_separator="\u00A0",
    ),
    
    # === EASTERN EUROPEAN ===
    "Polish": LanguageConfig(
        code="pl",
        name="Polish",
        native_name="Polski",
        direction=WritingDirection.LTR,
        script=ScriptType.LATIN,
        recommended_fonts=["DejaVu Sans", "Liberation Sans", "Arial"],
        fallback_fonts=["Helvetica", "sans-serif"],
        quote_chars=("„", """),
        decimal_separator=",",
        thousands_separator="\u00A0",
    ),
    
    "Czech": LanguageConfig(
        code="cs",
        name="Czech",
        native_name="Čeština",
        direction=WritingDirection.LTR,
        script=ScriptType.LATIN,
        recommended_fonts=["DejaVu Sans", "Liberation Sans", "Arial"],
        fallback_fonts=["Helvetica", "sans-serif"],
        quote_chars=("„", """),
        decimal_separator=",",
        thousands_separator="\u00A0",
    ),
    
    "Hungarian": LanguageConfig(
        code="hu",
        name="Hungarian",
        native_name="Magyar",
        direction=WritingDirection.LTR,
        script=ScriptType.LATIN,
        recommended_fonts=["DejaVu Sans", "Liberation Sans", "Arial"],
        fallback_fonts=["Helvetica", "sans-serif"],
        quote_chars=("„", """),
        decimal_separator=",",
        thousands_separator="\u00A0",
    ),
    
    "Romanian": LanguageConfig(
        code="ro",
        name="Romanian",
        native_name="Română",
        direction=WritingDirection.LTR,
        script=ScriptType.LATIN,
        recommended_fonts=["DejaVu Sans", "Liberation Sans", "Arial"],
        fallback_fonts=["Helvetica", "sans-serif"],
        quote_chars=("„", """),
        decimal_separator=",",
        thousands_separator=".",
    ),
    
    # === CYRILLIC ===
    "Russian": LanguageConfig(
        code="ru",
        name="Russian",
        native_name="Русский",
        direction=WritingDirection.LTR,
        script=ScriptType.CYRILLIC,
        recommended_fonts=["DejaVu Sans", "Liberation Sans", "PT Sans"],
        fallback_fonts=["Arial", "sans-serif"],
        quote_chars=("«", "»"),
        decimal_separator=",",
        thousands_separator="\u00A0",
        formal_register="Formal academic register, proper case endings",
    ),
    
    "Ukrainian": LanguageConfig(
        code="uk",
        name="Ukrainian",
        native_name="Українська",
        direction=WritingDirection.LTR,
        script=ScriptType.CYRILLIC,
        recommended_fonts=["DejaVu Sans", "Liberation Sans", "PT Sans"],
        fallback_fonts=["Arial", "sans-serif"],
        quote_chars=("«", "»"),
        decimal_separator=",",
        thousands_separator="\u00A0",
    ),
    
    "Bulgarian": LanguageConfig(
        code="bg",
        name="Bulgarian",
        native_name="Български",
        direction=WritingDirection.LTR,
        script=ScriptType.CYRILLIC,
        recommended_fonts=["DejaVu Sans", "Liberation Sans"],
        fallback_fonts=["Arial", "sans-serif"],
        quote_chars=("„", """),
        decimal_separator=",",
        thousands_separator="\u00A0",
    ),
    
    # === GREEK ===
    "Greek": LanguageConfig(
        code="el",
        name="Greek",
        native_name="Ελληνικά",
        direction=WritingDirection.LTR,
        script=ScriptType.GREEK,
        recommended_fonts=["DejaVu Sans", "Liberation Sans", "Noto Sans"],
        fallback_fonts=["Arial", "sans-serif"],
        quote_chars=("«", "»"),
        decimal_separator=",",
        thousands_separator=".",
        formal_register="Keep Greek letters in formulas unchanged",
    ),
    
    # === CJK ===
    "Chinese": LanguageConfig(
        code="zh",
        name="Chinese",
        native_name="中文",
        direction=WritingDirection.LTR,
        script=ScriptType.CJK,
        recommended_fonts=["Noto Sans CJK SC", "Source Han Sans", "SimSun"],
        fallback_fonts=["Microsoft YaHei", "sans-serif"],
        uses_hyphenation=False,
        quote_chars=(""", """),
        decimal_separator=".",
        thousands_separator=",",
        formal_register="Simplified Chinese, proper measure words",
    ),
    
    "Japanese": LanguageConfig(
        code="ja",
        name="Japanese",
        native_name="日本語",
        direction=WritingDirection.LTR,
        script=ScriptType.CJK,
        recommended_fonts=["Noto Sans CJK JP", "Source Han Sans", "MS Gothic"],
        fallback_fonts=["Meiryo", "sans-serif"],
        uses_hyphenation=False,
        quote_chars=("「", "」"),
        decimal_separator=".",
        thousands_separator=",",
        formal_register="Formal です/ます form, proper particles",
    ),
    
    "Korean": LanguageConfig(
        code="ko",
        name="Korean",
        native_name="한국어",
        direction=WritingDirection.LTR,
        script=ScriptType.CJK,
        recommended_fonts=["Noto Sans CJK KR", "Source Han Sans", "Malgun Gothic"],
        fallback_fonts=["Gulim", "sans-serif"],
        uses_hyphenation=False,
        quote_chars=(""", """),
        decimal_separator=".",
        thousands_separator=",",
    ),
    
    # === RTL LANGUAGES ===
    "Arabic": LanguageConfig(
        code="ar",
        name="Arabic",
        native_name="العربية",
        direction=WritingDirection.RTL,
        script=ScriptType.ARABIC,
        recommended_fonts=["Noto Sans Arabic", "Amiri", "Arial"],
        fallback_fonts=["sans-serif"],
        uses_hyphenation=False,
        quote_chars=("«", "»"),
        decimal_separator="٫",
        thousands_separator="٬",
        formal_register="Modern Standard Arabic, formal register",
    ),
    
    "Hebrew": LanguageConfig(
        code="he",
        name="Hebrew",
        native_name="עברית",
        direction=WritingDirection.RTL,
        script=ScriptType.HEBREW,
        recommended_fonts=["Noto Sans Hebrew", "David", "Arial"],
        fallback_fonts=["sans-serif"],
        uses_hyphenation=False,
        quote_chars=('"', '"'),
        decimal_separator=".",
        thousands_separator=",",
    ),
    
    # === OTHER ===
    "Turkish": LanguageConfig(
        code="tr",
        name="Turkish",
        native_name="Türkçe",
        direction=WritingDirection.LTR,
        script=ScriptType.LATIN,
        recommended_fonts=["DejaVu Sans", "Liberation Sans", "Arial"],
        fallback_fonts=["Helvetica", "sans-serif"],
        quote_chars=(""", """),
        decimal_separator=",",
        thousands_separator=".",
    ),
    
    "Vietnamese": LanguageConfig(
        code="vi",
        name="Vietnamese",
        native_name="Tiếng Việt",
        direction=WritingDirection.LTR,
        script=ScriptType.LATIN,
        recommended_fonts=["DejaVu Sans", "Liberation Sans", "Arial"],
        fallback_fonts=["sans-serif"],
        quote_chars=(""", """),
        decimal_separator=",",
        thousands_separator=".",
    ),
    
    "Thai": LanguageConfig(
        code="th",
        name="Thai",
        native_name="ไทย",
        direction=WritingDirection.LTR,
        script=ScriptType.THAI,
        recommended_fonts=["Noto Sans Thai", "Tahoma"],
        fallback_fonts=["sans-serif"],
        uses_hyphenation=False,
        quote_chars=(""", """),
        decimal_separator=".",
        thousands_separator=",",
    ),
    
    "Hindi": LanguageConfig(
        code="hi",
        name="Hindi",
        native_name="हिन्दी",
        direction=WritingDirection.LTR,
        script=ScriptType.DEVANAGARI,
        recommended_fonts=["Noto Sans Devanagari", "Mangal"],
        fallback_fonts=["sans-serif"],
        uses_hyphenation=False,
        quote_chars=(""", """),
        decimal_separator=".",
        thousands_separator=",",
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_language(name: str) -> Optional[LanguageConfig]:
    """Get language config by name (case-insensitive)."""
    name_lower = name.lower()
    
    for lang_name, config in LANGUAGES.items():
        if lang_name.lower() == name_lower or config.code == name_lower:
            return config
    
    return None


def get_all_language_names() -> List[str]:
    """Get list of all supported language names."""
    return list(LANGUAGES.keys())


def get_languages_by_script(script: ScriptType) -> List[LanguageConfig]:
    """Get all languages using a specific script."""
    return [lang for lang in LANGUAGES.values() if lang.script == script]


def get_rtl_languages() -> List[LanguageConfig]:
    """Get all RTL languages."""
    return [lang for lang in LANGUAGES.values() if lang.direction == WritingDirection.RTL]


def is_rtl(language: str) -> bool:
    """Check if a language is RTL."""
    config = get_language(language)
    return config.direction == WritingDirection.RTL if config else False


def get_font_for_language(language: str) -> str:
    """Get recommended font for a language."""
    config = get_language(language)
    if config and config.recommended_fonts:
        return config.recommended_fonts[0]
    return "sans-serif"


def detect_language_from_text(text: str) -> Optional[str]:
    """Attempt to detect language from text sample."""
    try:
        from langdetect import detect
        code = detect(text)
        
        # Map code to language name
        for name, config in LANGUAGES.items():
            if config.code == code:
                return name
        
        return None
    except Exception:
        return None


# =============================================================================
# LANGUAGE SELECTION UI HELPER
# =============================================================================

def get_language_choices() -> List[Tuple[str, str]]:
    """Get language choices for UI dropdowns (name, display_name)."""
    choices = []
    for name, config in LANGUAGES.items():
        display = f"{name} ({config.native_name})"
        choices.append((name, display))
    
    # Sort by name
    choices.sort(key=lambda x: x[0])
    return choices


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Language Support Test ===\n")
    
    print(f"Supported languages: {len(LANGUAGES)}")
    
    # Group by script
    for script in ScriptType:
        langs = get_languages_by_script(script)
        if langs:
            names = [l.name for l in langs]
            print(f"\n{script.value}: {', '.join(names)}")
    
    # RTL languages
    rtl = get_rtl_languages()
    print(f"\nRTL languages: {[l.name for l in rtl]}")
    
    # Test specific language
    print("\n### German Config")
    de = get_language("German")
    if de:
        print(f"  Code: {de.code}")
        print(f"  Native: {de.native_name}")
        print(f"  Quotes: {de.quote_chars}")
        print(f"  Decimal: {de.decimal_separator}")
        print(f"  Fonts: {de.recommended_fonts[0]}")
    
    # UI choices
    print(f"\n### Language Choices (first 5)")
    for name, display in get_language_choices()[:5]:
        print(f"  {display}")
    
    print(f"\n✅ {len(LANGUAGES)} languages supported")
