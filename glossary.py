"""
Scientific Glossary - Consistent Terminology Translation

Ensures that:
1. Technical terms are translated consistently across the document
2. Author names, abbreviations, and formulas are never translated
3. Domain-specific terminology is handled correctly

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import re
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger("pdf_translator.glossary")


# =============================================================================
# NEVER TRANSLATE - These terms should remain unchanged in all languages
# =============================================================================

NEVER_TRANSLATE = [
    # Physics abbreviations
    "SSZ", "GR", "QFT", "QCD", "QED", "SM", "LCDM", "CMB",
    "LIGO", "VIRGO", "EHT", "JWST", "HST", "ESO", "NASA", "ESA",
    "PPN", "ADM", "BH", "NS", "WD", "AGN", "GRB", "SNR",
    
    # Mathematical notation
    "NFKC", "UTF-8", "LaTeX", "TeX", "BibTeX",
    
    # Units (keep as-is)
    "GeV", "MeV", "keV", "eV", "TeV",
    "pc", "kpc", "Mpc", "Gpc",
    "AU", "ly",
    
    # Greek letters (when written out)
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi",
    "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
    
    # Common physics symbols
    "hbar", "nabla",
    
    # Journal abbreviations
    "ApJ", "MNRAS", "A&A", "PRD", "PRL", "Nature", "Science",
    "Phys. Rev.", "Astrophys. J.",
]

# Author names that should never be translated
AUTHOR_NAMES = [
    # Famous physicists
    "Einstein", "Schwarzschild", "Kerr", "Newton", "Maxwell",
    "Planck", "Bohr", "Heisenberg", "Schrödinger", "Dirac",
    "Hawking", "Penrose", "Wheeler", "Thorne", "Misner",
    "Feynman", "Weinberg", "Glashow", "Salam",
    
    # SSZ authors
    "Wrede", "Casu", "Kalinowski",
    
    # Common collaborations
    "LIGO", "Virgo", "KAGRA", "Planck Collaboration",
]


# =============================================================================
# CONSISTENT TRANSLATIONS - Same term = same translation throughout document
# =============================================================================

CONSISTENT_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    # Spacetime & Geometry
    "spacetime": {
        "de": "Raumzeit",
        "it": "spaziotempo",
        "fr": "espace-temps",
        "es": "espacio-tiempo",
    },
    "space-time": {
        "de": "Raumzeit",
        "it": "spaziotempo",
        "fr": "espace-temps",
        "es": "espacio-tiempo",
    },
    "metric": {
        "de": "Metrik",
        "it": "metrica",
        "fr": "métrique",
        "es": "métrica",
    },
    "curvature": {
        "de": "Krümmung",
        "it": "curvatura",
        "fr": "courbure",
        "es": "curvatura",
    },
    "geodesic": {
        "de": "Geodäte",
        "it": "geodetica",
        "fr": "géodésique",
        "es": "geodésica",
    },
    
    # Black holes
    "black hole": {
        "de": "Schwarzes Loch",
        "it": "buco nero",
        "fr": "trou noir",
        "es": "agujero negro",
    },
    "event horizon": {
        "de": "Ereignishorizont",
        "it": "orizzonte degli eventi",
        "fr": "horizon des événements",
        "es": "horizonte de eventos",
    },
    "singularity": {
        "de": "Singularität",
        "it": "singolarità",
        "fr": "singularité",
        "es": "singularidad",
    },
    "Schwarzschild radius": {
        "de": "Schwarzschild-Radius",
        "it": "raggio di Schwarzschild",
        "fr": "rayon de Schwarzschild",
        "es": "radio de Schwarzschild",
    },
    
    # Relativity
    "general relativity": {
        "de": "Allgemeine Relativitätstheorie",
        "it": "relatività generale",
        "fr": "relativité générale",
        "es": "relatividad general",
    },
    "special relativity": {
        "de": "Spezielle Relativitätstheorie",
        "it": "relatività ristretta",
        "fr": "relativité restreinte",
        "es": "relatividad especial",
    },
    "time dilation": {
        "de": "Zeitdilatation",
        "it": "dilatazione del tempo",
        "fr": "dilatation du temps",
        "es": "dilatación del tiempo",
    },
    "gravitational redshift": {
        "de": "Gravitationsrotverschiebung",
        "it": "redshift gravitazionale",
        "fr": "décalage gravitationnel vers le rouge",
        "es": "corrimiento al rojo gravitacional",
    },
    
    # Quantum
    "quantum": {
        "de": "Quanten-",
        "it": "quantistico",
        "fr": "quantique",
        "es": "cuántico",
    },
    "wave function": {
        "de": "Wellenfunktion",
        "it": "funzione d'onda",
        "fr": "fonction d'onde",
        "es": "función de onda",
    },
    "entanglement": {
        "de": "Verschränkung",
        "it": "entanglement",
        "fr": "intrication",
        "es": "entrelazamiento",
    },
    "coherence": {
        "de": "Kohärenz",
        "it": "coerenza",
        "fr": "cohérence",
        "es": "coherencia",
    },
    "decoherence": {
        "de": "Dekohärenz",
        "it": "decoerenza",
        "fr": "décohérence",
        "es": "decoherencia",
    },
    
    # SSZ-specific
    "segment density": {
        "de": "Segmentdichte",
        "it": "densità di segmento",
        "fr": "densité de segment",
        "es": "densidad de segmento",
    },
    "segmented spacetime": {
        "de": "Segmentierte Raumzeit",
        "it": "spaziotempo segmentato",
        "fr": "espace-temps segmenté",
        "es": "espacio-tiempo segmentado",
    },
    
    # General physics
    "energy": {
        "de": "Energie",
        "it": "energia",
        "fr": "énergie",
        "es": "energía",
    },
    "momentum": {
        "de": "Impuls",
        "it": "impulso",
        "fr": "impulsion",
        "es": "momento",
    },
    "angular momentum": {
        "de": "Drehimpuls",
        "it": "momento angolare",
        "fr": "moment angulaire",
        "es": "momento angular",
    },
    "mass": {
        "de": "Masse",
        "it": "massa",
        "fr": "masse",
        "es": "masa",
    },
    "velocity": {
        "de": "Geschwindigkeit",
        "it": "velocità",
        "fr": "vitesse",
        "es": "velocidad",
    },
    "acceleration": {
        "de": "Beschleunigung",
        "it": "accelerazione",
        "fr": "accélération",
        "es": "aceleración",
    },
    
    # Document structure
    "abstract": {
        "de": "Zusammenfassung",
        "it": "sommario",
        "fr": "résumé",
        "es": "resumen",
    },
    "introduction": {
        "de": "Einleitung",
        "it": "introduzione",
        "fr": "introduction",
        "es": "introducción",
    },
    "conclusion": {
        "de": "Schlussfolgerung",
        "it": "conclusione",
        "fr": "conclusion",
        "es": "conclusión",
    },
    "references": {
        "de": "Literaturverzeichnis",
        "it": "riferimenti",
        "fr": "références",
        "es": "referencias",
    },
    "acknowledgments": {
        "de": "Danksagung",
        "it": "ringraziamenti",
        "fr": "remerciements",
        "es": "agradecimientos",
    },
}


# =============================================================================
# PLACEHOLDER SYSTEM
# =============================================================================

class GlossaryProcessor:
    """
    Processes text with glossary for consistent translation.
    
    Usage:
        processor = GlossaryProcessor("de")
        protected_text, restore_func = processor.protect(text)
        translated = translate(protected_text)
        final = restore_func(translated)
    """
    
    def __init__(self, target_language: str):
        self.target_language = target_language.lower()[:2]  # "German" -> "de"
        self.placeholders: Dict[str, str] = {}
        self.counter = 0
    
    def _make_placeholder(self, term: str, category: str) -> str:
        """Create a unique placeholder for a term."""
        placeholder = f"__GLOSS_{category}_{self.counter}__"
        self.placeholders[placeholder] = term
        self.counter += 1
        return placeholder
    
    def protect(self, text: str) -> Tuple[str, callable]:
        """
        Protect terms that should not be translated.
        
        Returns:
            Tuple of (protected_text, restore_function)
        """
        protected = text
        
        # 1. Protect author names (case-insensitive word boundaries)
        for name in AUTHOR_NAMES:
            pattern = rf'\b{re.escape(name)}\b'
            matches = list(re.finditer(pattern, protected, re.IGNORECASE))
            for match in reversed(matches):  # Reverse to preserve positions
                original = match.group()
                placeholder = self._make_placeholder(original, "NAME")
                protected = protected[:match.start()] + placeholder + protected[match.end():]
        
        # 2. Protect never-translate terms
        for term in sorted(NEVER_TRANSLATE, key=len, reverse=True):  # Longest first
            pattern = rf'\b{re.escape(term)}\b'
            matches = list(re.finditer(pattern, protected, re.IGNORECASE))
            for match in reversed(matches):
                original = match.group()
                placeholder = self._make_placeholder(original, "TERM")
                protected = protected[:match.start()] + placeholder + protected[match.end():]
        
        # 3. Apply consistent translations (replace with target language term)
        for term, translations in sorted(CONSISTENT_TRANSLATIONS.items(), key=lambda x: len(x[0]), reverse=True):
            if self.target_language in translations:
                pattern = rf'\b{re.escape(term)}\b'
                target_term = translations[self.target_language]
                # Use placeholder to ensure it's not re-translated
                placeholder = self._make_placeholder(target_term, "TRANS")
                protected = re.sub(pattern, placeholder, protected, flags=re.IGNORECASE)
        
        def restore(translated_text: str) -> str:
            """Restore all placeholders with their original/translated values."""
            result = translated_text
            for placeholder, value in self.placeholders.items():
                result = result.replace(placeholder, value)
            return result
        
        return protected, restore
    
    def get_glossary_prompt(self) -> str:
        """
        Generate a prompt section with glossary for LLM context.
        """
        lines = ["GLOSSARY - Use these translations consistently:"]
        
        for term, translations in CONSISTENT_TRANSLATIONS.items():
            if self.target_language in translations:
                lines.append(f"  {term} → {translations[self.target_language]}")
        
        lines.append("\nNEVER TRANSLATE these terms:")
        lines.append(f"  {', '.join(NEVER_TRANSLATE[:20])}...")
        
        lines.append("\nKEEP AUTHOR NAMES unchanged:")
        lines.append(f"  {', '.join(AUTHOR_NAMES[:10])}...")
        
        return "\n".join(lines)


def get_language_code(language: str) -> str:
    """Convert language name to 2-letter code."""
    lang_map = {
        "german": "de", "deutsch": "de",
        "english": "en", "englisch": "en",
        "italian": "it", "italiano": "it", "italienisch": "it",
        "french": "fr", "français": "fr", "französisch": "fr",
        "spanish": "es", "español": "es", "spanisch": "es",
        "portuguese": "pt", "português": "pt",
        "russian": "ru", "русский": "ru",
        "chinese": "zh", "中文": "zh",
        "japanese": "ja", "日本語": "ja",
    }
    return lang_map.get(language.lower(), language.lower()[:2])


def apply_glossary(text: str, target_language: str) -> Tuple[str, callable]:
    """
    Convenience function to apply glossary protection.
    
    Args:
        text: Text to process
        target_language: Target language (name or code)
    
    Returns:
        Tuple of (protected_text, restore_function)
    """
    lang_code = get_language_code(target_language)
    processor = GlossaryProcessor(lang_code)
    return processor.protect(text)


def get_glossary_context(target_language: str) -> str:
    """
    Get glossary context for LLM prompt.
    
    Args:
        target_language: Target language (name or code)
    
    Returns:
        Formatted glossary string for prompt
    """
    lang_code = get_language_code(target_language)
    processor = GlossaryProcessor(lang_code)
    return processor.get_glossary_prompt()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test the glossary
    test_text = """
    Einstein's theory of general relativity describes spacetime curvature.
    The Schwarzschild radius defines the event horizon of a black hole.
    SSZ theory proposes a segmented spacetime with segment density Xi.
    Authors: Wrede, Casu, and Kalinowski (2025).
    """
    
    print("=== Glossary Test ===\n")
    print("Original:")
    print(test_text)
    
    protected, restore = apply_glossary(test_text, "German")
    
    print("\nProtected:")
    print(protected)
    
    # Simulate translation (just uppercase for demo)
    fake_translated = protected.upper()
    
    print("\nRestored:")
    print(restore(fake_translated))
    
    print("\n=== Glossary Context ===")
    print(get_glossary_context("German"))
