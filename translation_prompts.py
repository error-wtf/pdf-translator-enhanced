"""
Translation Prompts - Optimized prompts for maximum translation quality

Contains carefully crafted prompts for:
- Scientific document translation
- Domain-specific terminology
- Style and tone preservation
- Formula and structure protection

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

from typing import Dict, Optional
from dataclasses import dataclass


# =============================================================================
# DOMAIN-SPECIFIC TERMINOLOGY HINTS
# =============================================================================

DOMAIN_TERMINOLOGY = {
    "physics": {
        "en": "Use standard physics terminology. Keep SI units unchanged.",
        "de": "Verwende Fachterminologie der Physik. SI-Einheiten unverändert lassen.",
        "it": "Usa terminologia fisica standard. Mantieni le unità SI invariate.",
        "fr": "Utilisez la terminologie physique standard. Gardez les unités SI inchangées.",
        "es": "Usa terminología física estándar. Mantén las unidades SI sin cambios.",
    },
    "mathematics": {
        "en": "Preserve mathematical notation exactly. Translate only prose.",
        "de": "Mathematische Notation exakt beibehalten. Nur Prosa übersetzen.",
        "it": "Preserva la notazione matematica esattamente. Traduci solo la prosa.",
        "fr": "Préservez exactement la notation mathématique. Traduisez uniquement la prose.",
        "es": "Preserva la notación matemática exactamente. Traduce solo la prosa.",
    },
    "chemistry": {
        "en": "Keep chemical formulas, IUPAC names, and reaction equations unchanged.",
        "de": "Chemische Formeln, IUPAC-Namen und Reaktionsgleichungen unverändert lassen.",
        "it": "Mantieni formule chimiche, nomi IUPAC e equazioni di reazione invariate.",
        "fr": "Gardez les formules chimiques, noms IUPAC et équations de réaction inchangés.",
        "es": "Mantén fórmulas químicas, nombres IUPAC y ecuaciones de reacción sin cambios.",
    },
    "biology": {
        "en": "Keep Latin species names (italicized) and gene names unchanged.",
        "de": "Lateinische Artnamen (kursiv) und Gennamen unverändert lassen.",
        "it": "Mantieni nomi di specie latini (in corsivo) e nomi di geni invariati.",
        "fr": "Gardez les noms d'espèces latins (en italique) et les noms de gènes inchangés.",
        "es": "Mantén nombres de especies en latín (cursiva) y nombres de genes sin cambios.",
    },
    "computer_science": {
        "en": "Keep code snippets, variable names, and technical terms unchanged.",
        "de": "Code-Snippets, Variablennamen und technische Begriffe unverändert lassen.",
        "it": "Mantieni frammenti di codice, nomi di variabili e termini tecnici invariati.",
        "fr": "Gardez les extraits de code, noms de variables et termes techniques inchangés.",
        "es": "Mantén fragmentos de código, nombres de variables y términos técnicos sin cambios.",
    },
}


# =============================================================================
# LANGUAGE-SPECIFIC STYLE GUIDES
# =============================================================================

STYLE_GUIDES = {
    "German": """GERMAN SCIENTIFIC STYLE:
- Use formal "Sie" form, not informal "du"
- Prefer German technical terms when established (e.g., "Gleichung" not "Equation")
- Keep compound nouns together (e.g., "Schwarzschildradius")
- Use "ss" not "ß" in Swiss German contexts only
- Maintain sentence structure appropriate for German (verb at end in subordinate clauses)""",
    
    "Italian": """ITALIAN SCIENTIFIC STYLE:
- Use formal register appropriate for academic writing
- Prefer Italian technical terms when established
- Keep article agreement correct (il/la/lo)
- Maintain proper use of subjunctive in scientific discourse""",
    
    "French": """FRENCH SCIENTIFIC STYLE:
- Use formal register ("nous" or impersonal constructions)
- Prefer French technical terms when established
- Maintain proper accent marks (é, è, ê, ë, etc.)
- Use proper spacing before punctuation (: ; ? !)""",
    
    "Spanish": """SPANISH SCIENTIFIC STYLE:
- Use formal register appropriate for academic writing
- Prefer Spanish technical terms when established
- Include opening ¿ and ¡ for questions and exclamations
- Maintain proper accent marks (á, é, í, ó, ú, ñ)""",
    
    "Portuguese": """PORTUGUESE SCIENTIFIC STYLE:
- Use formal register appropriate for academic writing
- Brazilian vs European Portuguese: prefer neutral forms
- Maintain proper accent marks and cedilla (ç)""",
    
    "Russian": """RUSSIAN SCIENTIFIC STYLE:
- Use formal academic register
- Maintain proper case endings
- Keep Latin terms and formulas unchanged
- Use Cyrillic equivalents for Greek letters when appropriate""",
    
    "Chinese": """CHINESE SCIENTIFIC STYLE:
- Use simplified Chinese characters (unless Traditional specified)
- Keep mathematical notation in standard form
- Maintain proper measure words (量词)
- Use formal academic register""",
    
    "Japanese": """JAPANESE SCIENTIFIC STYLE:
- Use formal です/ます form
- Keep mathematical notation unchanged
- Use appropriate kanji for technical terms
- Maintain proper particle usage""",
}


# =============================================================================
# CORE TRANSLATION PROMPTS
# =============================================================================

@dataclass
class TranslationPrompt:
    """A complete translation prompt with system and user parts."""
    system: str
    user_template: str
    
    def format_user(self, text: str, **kwargs) -> str:
        return self.user_template.format(text=text, **kwargs)


def get_scientific_translation_prompt(
    target_language: str,
    domain: Optional[str] = None,
    source_language: Optional[str] = None,
) -> TranslationPrompt:
    """
    Get optimized translation prompt for scientific documents.
    
    Args:
        target_language: Target language name
        domain: Optional domain (physics, mathematics, etc.)
        source_language: Optional source language
    
    Returns:
        TranslationPrompt with system and user template
    """
    # Get style guide
    style_guide = STYLE_GUIDES.get(target_language, "")
    
    # Get domain hints
    domain_hint = ""
    if domain:
        lang_code = {
            "German": "de", "Italian": "it", "French": "fr",
            "Spanish": "es", "English": "en"
        }.get(target_language, "en")
        domain_hints = DOMAIN_TERMINOLOGY.get(domain, {})
        domain_hint = domain_hints.get(lang_code, domain_hints.get("en", ""))
    
    system_prompt = f"""You are an expert scientific translator with deep knowledge of academic writing conventions.

TARGET LANGUAGE: {target_language}

{style_guide}

{f"DOMAIN: {domain}" if domain else ""}
{domain_hint}

ABSOLUTE RULES - NEVER VIOLATE:
1. **FORMULAS**: Never modify anything inside $...$ or $$...$$ or \\[...\\] or \\begin{{...}}...\\end{{...}}
2. **CITATIONS**: Keep \\cite{{...}}, \\citet{{...}}, \\citep{{...}}, \\ref{{...}}, \\eqref{{...}} EXACTLY as they are
3. **AUTHOR NAMES**: Never translate proper names (Einstein, Schwarzschild, Newton, etc.)
4. **ACRONYMS**: Keep abbreviations like GR, QFT, SSZ, LIGO unchanged
5. **UNITS**: Keep SI units (m, kg, s, K, Hz, etc.) unchanged
6. **VARIABLES**: Single letters in math context (x, y, z, α, β, etc.) stay unchanged

QUALITY STANDARDS:
- Translate with the precision expected in peer-reviewed journals
- Preserve the exact meaning - do not paraphrase loosely
- Maintain the formal academic register throughout
- Keep paragraph structure and emphasis (bold, italic) intact
- Ensure technical accuracy over fluency when in conflict

OUTPUT: Only the {target_language} translation. No explanations, no original text, no meta-comments."""

    user_template = """Translate this scientific text to {target_language}:

{text}"""

    return TranslationPrompt(
        system=system_prompt,
        user_template=user_template.replace("{target_language}", target_language)
    )


def get_figure_caption_prompt(target_language: str) -> TranslationPrompt:
    """Get optimized prompt for figure captions."""
    return TranslationPrompt(
        system=f"""You are translating a scientific figure caption to {target_language}.

RULES:
1. Keep "Figure X:" or "Fig. X:" prefix EXACTLY as written
2. Keep all mathematical notation unchanged
3. Keep reference numbers (e.g., "from [12]") unchanged
4. Translate only the descriptive text
5. Be concise - captions should be brief

OUTPUT: Only the translated caption.""",
        user_template="Translate this figure caption to {target_lang}:\n\n{{text}}".replace("{target_lang}", target_language)
    )


def get_table_content_prompt(target_language: str) -> TranslationPrompt:
    """Get optimized prompt for table content."""
    return TranslationPrompt(
        system=f"""You are translating scientific table content to {target_language}.

RULES:
1. Keep "Table X:" prefix EXACTLY as written
2. Maintain column alignment and structure
3. Keep all numbers, units, and symbols unchanged
4. Translate only header text and descriptive cells
5. Keep abbreviations and acronyms unchanged

OUTPUT: Only the translated table content, preserving structure.""",
        user_template="Translate this table content to {target_lang}:\n\n{{text}}".replace("{target_lang}", target_language)
    )


def get_abstract_prompt(target_language: str) -> TranslationPrompt:
    """Get optimized prompt for abstracts (highest quality)."""
    style = STYLE_GUIDES.get(target_language, "")
    
    return TranslationPrompt(
        system=f"""You are translating a scientific paper abstract to {target_language}.

{style}

This is the MOST IMPORTANT part of the paper. Translate with maximum precision.

RULES:
1. Preserve exact meaning - abstracts summarize key findings
2. Keep all technical terms accurate
3. Maintain formal academic tone
4. Keep all formulas, numbers, and citations unchanged
5. Ensure the translation could be published in a {target_language} journal

OUTPUT: Only the {target_language} abstract.""",
        user_template="Translate this abstract to {target_lang}:\n\n{{text}}".replace("{target_lang}", target_language)
    )


def get_equation_context_prompt(target_language: str) -> TranslationPrompt:
    """Get prompt for text surrounding equations."""
    return TranslationPrompt(
        system=f"""You are translating text that contains or references mathematical equations to {target_language}.

CRITICAL: The text contains equation placeholders like __LATEX_0__, __LATEX_1__, etc.
These MUST appear in your output EXACTLY as they appear in the input.

RULES:
1. Keep ALL __LATEX_X__ placeholders exactly as they are
2. Translate the surrounding prose naturally
3. Ensure grammatical agreement with placeholders (they represent equations)
4. Keep "Equation (X)", "Eq. (X)", "formula (X)" references unchanged

OUTPUT: Only the {target_language} translation with all placeholders preserved.""",
        user_template="Translate (keep all __LATEX_X__ placeholders):\n\n{{text}}".replace("{target_lang}", target_language)
    )


# =============================================================================
# REFINEMENT PROMPTS (for two-pass translation)
# =============================================================================

def get_refinement_prompt(target_language: str) -> TranslationPrompt:
    """Get prompt for refining/improving an existing translation."""
    return TranslationPrompt(
        system=f"""You are reviewing and improving a {target_language} scientific translation.

TASK: Compare the original with the current translation and improve if needed.

CHECK FOR:
1. Terminology consistency with the provided context
2. Grammatical correctness in {target_language}
3. Preserved formulas and citations
4. Natural flow while maintaining precision
5. Correct technical term translations

If the translation is already good, output it unchanged.
If improvements are needed, output the improved version.

OUTPUT: Only the (possibly improved) {target_language} translation.""",
        user_template="""ORIGINAL:
{original}

CURRENT TRANSLATION:
{current}

CONTEXT FROM DOCUMENT:
{context}

Output the best {target_lang} translation:""".replace("{target_lang}", target_language)
    )


# =============================================================================
# PROMPT SELECTION
# =============================================================================

def select_prompt(
    target_language: str,
    block_type: str = "text",
    domain: Optional[str] = None,
    has_placeholders: bool = False,
) -> TranslationPrompt:
    """
    Select the best prompt for a given translation task.
    
    Args:
        target_language: Target language
        block_type: Type of content (text, abstract, figure_caption, table, equation_context)
        domain: Scientific domain
        has_placeholders: Whether text contains __LATEX_X__ placeholders
    
    Returns:
        Appropriate TranslationPrompt
    """
    if has_placeholders:
        return get_equation_context_prompt(target_language)
    
    if block_type == "abstract":
        return get_abstract_prompt(target_language)
    elif block_type == "figure_caption":
        return get_figure_caption_prompt(target_language)
    elif block_type == "table":
        return get_table_content_prompt(target_language)
    else:
        return get_scientific_translation_prompt(target_language, domain)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Translation Prompts Test ===\n")
    
    # Test scientific prompt
    prompt = get_scientific_translation_prompt("German", domain="physics")
    print("### German Physics Prompt")
    print(f"System ({len(prompt.system)} chars):")
    print(prompt.system[:500] + "...\n")
    
    # Test figure caption
    prompt = get_figure_caption_prompt("Italian")
    print("### Italian Figure Caption Prompt")
    print(f"System: {prompt.system[:200]}...\n")
    
    # Test with placeholders
    prompt = get_equation_context_prompt("French")
    print("### French Equation Context Prompt")
    print(f"System: {prompt.system[:200]}...\n")
    
    print("✅ All prompts available")
