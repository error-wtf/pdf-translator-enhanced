"""
Two-Pass Translator - Enhanced consistency through refinement

Strategy:
1. Pass 1: Translate all blocks independently (fast)
2. Pass 2: Review translations with full context for consistency

Benefits:
- Detects terminology inconsistencies
- Ensures coherent style across document
- Fixes errors that single-block translation misses

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import logging
import re
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger("pdf_translator.two_pass")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TranslationBlock:
    """A block of text with its translation."""
    original: str
    translated: str = ""
    block_type: str = "text"  # text, heading, caption, table
    page: int = 0
    position: int = 0  # Position within page
    refined: bool = False
    confidence: float = 1.0


@dataclass
class TranslationDocument:
    """A complete document with all translation blocks."""
    blocks: List[TranslationBlock] = field(default_factory=list)
    source_language: str = ""
    target_language: str = ""
    terminology: Dict[str, str] = field(default_factory=dict)  # Extracted term mappings
    
    def get_full_original(self) -> str:
        """Get concatenated original text."""
        return "\n\n".join(b.original for b in self.blocks if b.original)
    
    def get_full_translation(self) -> str:
        """Get concatenated translated text."""
        return "\n\n".join(b.translated for b in self.blocks if b.translated)
    
    def get_context_window(self, block_idx: int, window_size: int = 3) -> str:
        """Get surrounding translations for context."""
        start = max(0, block_idx - window_size)
        end = min(len(self.blocks), block_idx + window_size + 1)
        
        context_parts = []
        for i in range(start, end):
            if i == block_idx:
                context_parts.append(f"[CURRENT BLOCK]")
            elif self.blocks[i].translated:
                context_parts.append(self.blocks[i].translated[:200])
        
        return "\n---\n".join(context_parts)


# =============================================================================
# TERMINOLOGY EXTRACTION
# =============================================================================

def extract_technical_terms(text: str) -> List[str]:
    """
    Extract likely technical terms from text.
    
    Looks for:
    - Capitalized multi-word phrases
    - Terms with numbers/symbols
    - Hyphenated compounds
    """
    terms = set()
    
    # Capitalized phrases (likely proper nouns or technical terms)
    cap_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
    terms.update(cap_phrases)
    
    # Terms with numbers (e.g., "Type II", "Phase 3")
    num_terms = re.findall(r'\b[A-Za-z]+\s*[IVX0-9]+\b', text)
    terms.update(num_terms)
    
    # Hyphenated compounds
    hyphen_terms = re.findall(r'\b[A-Za-z]+-[A-Za-z]+(?:-[A-Za-z]+)*\b', text)
    terms.update(hyphen_terms)
    
    # All-caps acronyms
    acronyms = re.findall(r'\b[A-Z]{2,6}\b', text)
    terms.update(acronyms)
    
    return list(terms)


def find_term_translations(
    original_terms: List[str],
    original_text: str,
    translated_text: str
) -> Dict[str, str]:
    """
    Try to find how terms were translated by comparing texts.
    
    Simple heuristic: look for terms that appear in similar positions.
    """
    mappings = {}
    
    # Split into sentences for alignment
    orig_sentences = re.split(r'[.!?]\s+', original_text)
    trans_sentences = re.split(r'[.!?]\s+', translated_text)
    
    # Only process if similar sentence count
    if len(orig_sentences) != len(trans_sentences):
        return mappings
    
    for term in original_terms:
        term_lower = term.lower()
        
        # Find which sentence contains the term
        for i, orig_sent in enumerate(orig_sentences):
            if term_lower in orig_sent.lower():
                # Look for potential translations in corresponding sentence
                trans_sent = trans_sentences[i] if i < len(trans_sentences) else ""
                
                # For acronyms, they usually stay the same
                if term.isupper() and term in trans_sent:
                    mappings[term] = term
                
                break
    
    return mappings


def check_terminology_consistency(doc: TranslationDocument) -> List[Dict]:
    """
    Check if technical terms are translated consistently.
    
    Returns list of inconsistencies found.
    """
    inconsistencies = []
    
    # Build term usage map
    term_translations: Dict[str, List[str]] = {}
    
    for block in doc.blocks:
        terms = extract_technical_terms(block.original)
        
        for term in terms:
            if term not in term_translations:
                term_translations[term] = []
            
            # Find how this term appears in translation
            # (simplified - just check if term appears unchanged)
            if term in block.translated:
                term_translations[term].append(term)
            else:
                # Term was translated somehow
                term_translations[term].append("[translated]")
    
    # Check for inconsistencies
    for term, usages in term_translations.items():
        unique_usages = set(usages)
        if len(unique_usages) > 1:
            inconsistencies.append({
                "term": term,
                "usages": list(unique_usages),
                "count": len(usages)
            })
    
    return inconsistencies


# =============================================================================
# PASS 1: INITIAL TRANSLATION
# =============================================================================

def pass1_translate(
    blocks: List[str],
    translate_func: Callable[[str], str],
    block_types: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> TranslationDocument:
    """
    Pass 1: Translate all blocks independently.
    
    Args:
        blocks: List of text blocks to translate
        translate_func: Function that translates a single block
        block_types: Optional list of block types
        progress_callback: Optional progress callback
    
    Returns:
        TranslationDocument with initial translations
    """
    doc = TranslationDocument()
    total = len(blocks)
    
    for i, block_text in enumerate(blocks):
        if progress_callback:
            progress_callback(i + 1, total, f"Pass 1: Block {i + 1}/{total}")
        
        block = TranslationBlock(
            original=block_text,
            block_type=block_types[i] if block_types and i < len(block_types) else "text",
            position=i
        )
        
        if block_text.strip():
            block.translated = translate_func(block_text)
        else:
            block.translated = block_text
        
        doc.blocks.append(block)
    
    logger.info(f"Pass 1 complete: {len(doc.blocks)} blocks translated")
    return doc


# =============================================================================
# PASS 2: CONSISTENCY REFINEMENT
# =============================================================================

def pass2_refine(
    doc: TranslationDocument,
    refine_func: Callable[[str, str, str], str],
    target_language: str,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> TranslationDocument:
    """
    Pass 2: Refine translations for consistency.
    
    This pass:
    1. Checks for terminology inconsistencies
    2. Sends problematic blocks for re-translation with context
    3. Updates the document
    
    Args:
        doc: TranslationDocument from pass 1
        refine_func: Function(original, current_translation, context) -> refined_translation
        target_language: Target language name
        progress_callback: Optional progress callback
    
    Returns:
        Refined TranslationDocument
    """
    # Check for inconsistencies
    inconsistencies = check_terminology_consistency(doc)
    
    if inconsistencies:
        logger.info(f"Found {len(inconsistencies)} terminology inconsistencies")
        for inc in inconsistencies[:5]:  # Log first 5
            logger.debug(f"  {inc['term']}: {inc['usages']}")
    
    # Find blocks that need refinement
    blocks_to_refine = []
    
    for i, block in enumerate(doc.blocks):
        needs_refinement = False
        
        # Check if block contains inconsistent terms
        for inc in inconsistencies:
            if inc["term"].lower() in block.original.lower():
                needs_refinement = True
                break
        
        # Also refine headings and captions for consistency
        if block.block_type in ["heading", "caption"]:
            needs_refinement = True
        
        if needs_refinement:
            blocks_to_refine.append(i)
    
    # Refine identified blocks
    total = len(blocks_to_refine)
    
    for idx, block_idx in enumerate(blocks_to_refine):
        if progress_callback:
            progress_callback(idx + 1, total, f"Pass 2: Refining {idx + 1}/{total}")
        
        block = doc.blocks[block_idx]
        context = doc.get_context_window(block_idx, window_size=2)
        
        # Build refinement context
        terminology_hint = ""
        if doc.terminology:
            terminology_hint = "Previous translations used: " + ", ".join(
                f"{k}={v}" for k, v in list(doc.terminology.items())[:10]
            )
        
        full_context = f"{context}\n\n{terminology_hint}"
        
        # Request refinement
        refined = refine_func(block.original, block.translated, full_context)
        
        if refined and refined != block.translated:
            block.translated = refined
            block.refined = True
    
    logger.info(f"Pass 2 complete: {len(blocks_to_refine)} blocks refined")
    return doc


# =============================================================================
# REFINEMENT PROMPTS
# =============================================================================

def create_refinement_prompt(
    original: str,
    current_translation: str,
    context: str,
    target_language: str
) -> str:
    """Create a prompt for the refinement pass."""
    return f"""Review and improve this {target_language} translation for consistency.

ORIGINAL TEXT:
{original}

CURRENT TRANSLATION:
{current_translation}

SURROUNDING CONTEXT:
{context}

TASK:
1. Check if terminology is consistent with the context
2. Fix any awkward phrasing
3. Ensure technical terms are translated consistently
4. Output ONLY the improved {target_language} translation

If the current translation is already good, output it unchanged."""


def refine_with_ollama(
    original: str,
    current_translation: str,
    context: str,
    model: str,
    target_language: str,
    ollama_url: str = "http://localhost:11434"
) -> str:
    """Refine a translation using Ollama."""
    import requests
    
    prompt = create_refinement_prompt(original, current_translation, context, target_language)
    
    try:
        response = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are a translation reviewer. Improve translations for consistency. Output ONLY the {target_language} text."
                    },
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 2048}
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json().get("message", {}).get("content", "")
            return result.strip() if result else current_translation
        
    except Exception as e:
        logger.warning(f"Refinement failed: {e}")
    
    return current_translation


# =============================================================================
# MAIN TWO-PASS FUNCTION
# =============================================================================

def translate_two_pass(
    blocks: List[str],
    model: str,
    target_language: str,
    translate_func: Callable[[str], str],
    block_types: Optional[List[str]] = None,
    enable_pass2: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> List[str]:
    """
    Main two-pass translation function.
    
    Args:
        blocks: List of text blocks to translate
        model: Ollama model name
        target_language: Target language
        translate_func: Function to translate a single block
        block_types: Optional list of block types
        enable_pass2: Whether to run pass 2 (default True)
        progress_callback: Optional progress callback
    
    Returns:
        List of translated blocks
    """
    if not blocks:
        return []
    
    # Pass 1: Initial translation
    doc = pass1_translate(
        blocks,
        translate_func,
        block_types,
        progress_callback
    )
    
    doc.target_language = target_language
    
    # Pass 2: Refinement (optional)
    if enable_pass2 and len(blocks) > 1:
        def refine_func(orig, curr, ctx):
            return refine_with_ollama(orig, curr, ctx, model, target_language)
        
        doc = pass2_refine(doc, refine_func, target_language, progress_callback)
    
    return [b.translated for b in doc.blocks]


# =============================================================================
# BACK-TRANSLATION VALIDATION (Optional)
# =============================================================================

def validate_with_back_translation(
    original: str,
    translated: str,
    back_translate_func: Callable[[str], str],
    similarity_threshold: float = 0.7
) -> Tuple[bool, float, str]:
    """
    Validate a translation by translating it back and comparing.
    
    Args:
        original: Original text
        translated: Translated text
        back_translate_func: Function to translate back to source language
        similarity_threshold: Minimum similarity score to pass
    
    Returns:
        Tuple of (is_valid, similarity_score, back_translation)
    """
    back_translated = back_translate_func(translated)
    
    # Simple word overlap similarity
    orig_words = set(original.lower().split())
    back_words = set(back_translated.lower().split())
    
    if not orig_words:
        return True, 1.0, back_translated
    
    overlap = len(orig_words & back_words)
    similarity = overlap / len(orig_words)
    
    is_valid = similarity >= similarity_threshold
    
    if not is_valid:
        logger.warning(f"Back-translation validation failed: {similarity:.2f} < {similarity_threshold}")
    
    return is_valid, similarity, back_translated


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Two-Pass Translator Test ===\n")
    
    # Test blocks
    test_blocks = [
        "The black hole has an event horizon.",
        "Near the event horizon, time dilation becomes extreme.",
        "The Schwarzschild radius defines the size of the event horizon.",
        "Black holes can merge and emit gravitational waves.",
    ]
    
    print("Test blocks:")
    for i, block in enumerate(test_blocks):
        print(f"  {i+1}. {block}")
    
    # Extract terms
    all_terms = []
    for block in test_blocks:
        all_terms.extend(extract_technical_terms(block))
    
    print(f"\nExtracted terms: {list(set(all_terms))}")
    
    # Simulate pass 1
    def mock_translate(text):
        # Simple mock that replaces some words
        return text.replace("black hole", "Schwarzes Loch").replace("event horizon", "Ereignishorizont")
    
    doc = pass1_translate(test_blocks, mock_translate)
    
    print("\nPass 1 translations:")
    for block in doc.blocks:
        print(f"  {block.translated[:60]}...")
    
    # Check consistency
    inconsistencies = check_terminology_consistency(doc)
    print(f"\nInconsistencies found: {len(inconsistencies)}")
    
    print("\n✅ Two-pass translator module ready")
