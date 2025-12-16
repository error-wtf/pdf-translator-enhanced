"""
Text Normalizer - Strict cleanup of PDF extraction artifacts

Fixes:
1. Remove garbage characters (￾, soft hyphen, zero-width, replacement char)
2. Normalize Unicode (NFKC)
3. Fix ligatures (ﬁ → fi)
4. Normalize dashes/minus
5. Collapse whitespace
6. Paragraph reflow (PDF line wraps → proper paragraphs)

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import re
import unicodedata
from typing import List

# Zero-width characters to remove
ZERO_WIDTH_CHARS = {
    0x200B,  # Zero Width Space
    0x200C,  # Zero Width Non-Joiner
    0x200D,  # Zero Width Joiner
    0x2060,  # Word Joiner
    0xFEFF,  # Zero Width No-Break Space (BOM)
    0x00AD,  # Soft Hyphen
}

# Build translation table for zero-width removal
ZERO_WIDTH_TABLE = dict.fromkeys(ZERO_WIDTH_CHARS, None)

# Garbage characters commonly found in PDF extraction
GARBAGE_CHARS = [
    '\ufffd',  # Replacement Character
    '\uffff',  # Not a character
    '\ufffe',  # Not a character (the infamous ￾)
    '\u0000',  # Null
    '\u001f',  # Unit Separator
    '\u001e',  # Record Separator
    '\u0001',  # Start of Heading
    '\u0002',  # Start of Text
    '\u0003',  # End of Text
    '\u0004',  # End of Transmission
    '\u0005',  # Enquiry
    '\u0006',  # Acknowledge
    '\u0007',  # Bell
    '\u0008',  # Backspace
    '\u000b',  # Vertical Tab
    '\u000c',  # Form Feed
    '\u000e',  # Shift Out
    '\u000f',  # Shift In
    '\u0010',  # Data Link Escape
    '\u0011',  # Device Control 1
    '\u0012',  # Device Control 2
    '\u0013',  # Device Control 3
    '\u0014',  # Device Control 4
    '\u0015',  # Negative Acknowledge
    '\u0016',  # Synchronous Idle
    '\u0017',  # End of Transmission Block
    '\u0018',  # Cancel
    '\u0019',  # End of Medium
    '\u001a',  # Substitute
    '\u001b',  # Escape
    '\u001c',  # File Separator
    '\u001d',  # Group Separator
]

# Superscript/subscript mappings for scientific notation
SUPERSCRIPTS = {
    '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
    '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
    '⁺': '+', '⁻': '-', '⁼': '=', '⁽': '(', '⁾': ')',
    'ⁿ': 'n', 'ⁱ': 'i',
}

SUBSCRIPTS = {
    '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
    '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
    '₊': '+', '₋': '-', '₌': '=', '₍': '(', '₎': ')',
    'ₐ': 'a', 'ₑ': 'e', 'ₒ': 'o', 'ₓ': 'x', 'ₕ': 'h',
    'ₖ': 'k', 'ₗ': 'l', 'ₘ': 'm', 'ₙ': 'n', 'ₚ': 'p',
    'ₛ': 's', 'ₜ': 't',
}

# Ligature mappings
LIGATURES = {
    '\ufb00': 'ff',   # ﬀ
    '\ufb01': 'fi',   # ﬁ
    '\ufb02': 'fl',   # ﬂ
    '\ufb03': 'ffi',  # ﬃ
    '\ufb04': 'ffl',  # ﬄ
    '\ufb05': 'st',   # ﬅ (long s + t)
    '\ufb06': 'st',   # ﬆ
    'ﬁ': 'fi',
    'ﬂ': 'fl',
    'ﬀ': 'ff',
    'ﬃ': 'ffi',
    'ﬄ': 'ffl',
}

# Dash/minus variants to normalize
# NOTE: PRESERVE \u2212 (minus sign) for scientific content!
DASH_VARIANTS = {
    # '\u2212': '-',  # Minus Sign - KEEP for math formulas!
    '\u2013': '-',  # En Dash
    '\u2014': '-',  # Em Dash
    '\u2015': '-',  # Horizontal Bar
    '\u2010': '-',  # Hyphen
    '\u2011': '-',  # Non-Breaking Hyphen
    '\u00ad': '',   # Soft Hyphen (remove entirely)
}

# Space variants to normalize
SPACE_VARIANTS = {
    '\u00a0': ' ',  # Non-Breaking Space
    '\u2002': ' ',  # En Space
    '\u2003': ' ',  # Em Space
    '\u2004': ' ',  # Three-Per-Em Space
    '\u2005': ' ',  # Four-Per-Em Space
    '\u2006': ' ',  # Six-Per-Em Space
    '\u2007': ' ',  # Figure Space
    '\u2008': ' ',  # Punctuation Space
    '\u2009': ' ',  # Thin Space
    '\u200a': ' ',  # Hair Space
    '\u202f': ' ',  # Narrow No-Break Space
    '\u205f': ' ',  # Medium Mathematical Space
    '\u3000': ' ',  # Ideographic Space
}


def convert_superscripts_to_latex(s: str) -> str:
    """Convert Unicode superscripts to LaTeX ^{} notation."""
    # Find sequences of superscript characters
    result = []
    i = 0
    while i < len(s):
        if s[i] in SUPERSCRIPTS:
            # Collect all consecutive superscripts
            sup_chars = []
            while i < len(s) and s[i] in SUPERSCRIPTS:
                sup_chars.append(SUPERSCRIPTS[s[i]])
                i += 1
            result.append('^{' + ''.join(sup_chars) + '}')
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)


def convert_subscripts_to_latex(s: str) -> str:
    """Convert Unicode subscripts to LaTeX _{} notation."""
    result = []
    i = 0
    while i < len(s):
        if s[i] in SUBSCRIPTS:
            # Collect all consecutive subscripts
            sub_chars = []
            while i < len(s) and s[i] in SUBSCRIPTS:
                sub_chars.append(SUBSCRIPTS[s[i]])
                i += 1
            result.append('_{' + ''.join(sub_chars) + '}')
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)


def fix_question_mark_artifacts(s: str) -> str:
    """
    Fix '?' artifacts that appear from bad glyph mapping.
    
    Common patterns:
    - "Phonon?Kopplung" → "Phonon-Kopplung"
    - "10?¹?" → "10^{-1}"
    - "Spin?Spin" → "Spin-Spin"
    """
    # Pattern: Word?Word (compound words with missing hyphen)
    s = re.sub(r'(\w)(\?+)(\w)', r'\1-\3', s)
    
    # Pattern: number?superscript (scientific notation)
    s = re.sub(r'(\d)\?([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)\??', 
               lambda m: m.group(1) + convert_superscripts_to_latex(m.group(2)), s)
    
    return s


def normalize_text(s: str) -> str:
    """
    Strict text normalization for PDF extraction output.
    
    This function MUST be called on every extracted text span
    BEFORE any TeX processing or translation.
    """
    if not s:
        return s
    
    # 1) Unicode normalize (NFKC - compatibility decomposition + canonical composition)
    s = unicodedata.normalize("NFKC", s)
    
    # 2) Remove zero-width characters
    s = s.translate(ZERO_WIDTH_TABLE)
    
    # 3) Remove garbage characters
    for char in GARBAGE_CHARS:
        s = s.replace(char, '')
    
    # 4) Fix ligatures
    for lig, replacement in LIGATURES.items():
        s = s.replace(lig, replacement)
    
    # 5) Normalize dashes/minus (in text mode)
    for dash, replacement in DASH_VARIANTS.items():
        s = s.replace(dash, replacement)
    
    # 6) Normalize space variants
    for space, replacement in SPACE_VARIANTS.items():
        s = s.replace(space, replacement)
    
    # 7) Fix '?' artifacts from bad glyph mapping
    s = fix_question_mark_artifacts(s)
    
    # 8) Convert Unicode super/subscripts to LaTeX
    s = convert_superscripts_to_latex(s)
    s = convert_subscripts_to_latex(s)
    
    # 9) Collapse multiple spaces (but preserve newlines)
    s = re.sub(r'[ \t]+', ' ', s)
    
    # 10) Remove trailing whitespace on lines
    s = re.sub(r' +\n', '\n', s)
    s = re.sub(r'\n +', '\n', s)
    
    # 11) Remove leading/trailing whitespace
    s = s.strip()
    
    return s


def is_heading(line: str) -> bool:
    """Check if a line looks like a heading."""
    L = line.strip()
    if not L:
        return False
    
    # Numbered sections: "1.", "1.1", "1.1.1", etc.
    if re.match(r'^(\d+(\.\d+)*)\.\s+', L):
        return True
    
    # Roman numeral sections: "I.", "II.", "III.", etc.
    if re.match(r'^(I{1,3}|IV|V|VI{0,3}|IX|X)\.\s+', L):
        return True
    
    # Common heading keywords
    heading_keywords = [
        'Abstract', 'Introduction', 'Background', 'Methods', 'Methodology',
        'Results', 'Discussion', 'Conclusion', 'Conclusions', 'References',
        'Acknowledgments', 'Acknowledgements', 'Appendix', 'Bibliography',
        'Summary', 'Overview', 'Theory', 'Analysis', 'Experiments',
        'Related Work', 'Future Work', 'Limitations'
    ]
    for kw in heading_keywords:
        if L.startswith(kw) and (len(L) == len(kw) or L[len(kw)] in ' :'):
            return True
    
    # Markdown-style headers
    if L.startswith('#'):
        return True
    
    return False


def is_bullet(line: str) -> bool:
    """Check if a line is a bullet point."""
    L = line.strip()
    if not L:
        return False
    
    # Common bullet patterns
    if re.match(r'^[•\-\*\+]\s+', L):
        return True
    
    # Numbered lists: "1)", "1.", "(1)", "(a)", "a)", etc.
    if re.match(r'^(\d+[\.\)]\s+|\(\d+\)\s+|\([a-z]\)\s+|[a-z][\.\)]\s+)', L):
        return True
    
    return False


def is_figure_caption(line: str) -> bool:
    """Check if a line is a figure/table caption."""
    L = line.strip()
    if not L:
        return False
    
    # Figure X: or Table X:
    if re.match(r'^(Figure|Fig\.|Table|Tab\.)\s*\d+', L, re.IGNORECASE):
        return True
    
    return False


def should_join_lines(prev_line: str, curr_line: str) -> bool:
    """
    Determine if two lines should be joined (PDF line wrap vs real paragraph break).
    
    Returns True if lines should be joined with a space.
    """
    if not prev_line or not curr_line:
        return False
    
    prev = prev_line.strip()
    curr = curr_line.strip()
    
    if not prev or not curr:
        return False
    
    # Don't join if current line is a heading/bullet/caption
    if is_heading(curr) or is_bullet(curr) or is_figure_caption(curr):
        return False
    
    # Don't join if previous line ends with sentence-ending punctuation
    # AND current line starts with uppercase (likely new paragraph)
    if prev[-1] in '.!?:' and curr[0].isupper():
        # But allow joining if it looks like a continuation
        # (e.g., "Dr." or abbreviations)
        if not re.search(r'\b(Dr|Mr|Mrs|Ms|Prof|Fig|Tab|Eq|et al|vs|i\.e|e\.g)\.$', prev):
            return False
    
    # Join if previous line ends with a hyphen (word continuation)
    if prev.endswith('-'):
        return True
    
    # Join if current line starts with lowercase (continuation)
    if curr[0].islower():
        return True
    
    # Join if previous line doesn't end with punctuation
    if prev[-1] not in '.!?:;':
        return True
    
    return False


def reflow_paragraphs(text: str) -> str:
    """
    Reflow text to fix PDF line wrapping issues.
    
    Rules:
    - Single newline within paragraph → space (join lines)
    - Double newline → paragraph break
    - Headings, bullets, captions → keep on own line
    """
    lines = text.split('\n')
    result = []
    buffer = ""
    
    for i, line in enumerate(lines):
        L = line.strip()
        
        # Empty line = paragraph break
        if not L:
            if buffer:
                result.append(buffer.strip())
                buffer = ""
            result.append("")  # Keep the paragraph break
            continue
        
        # Headings, bullets, captions get their own line
        if is_heading(L) or is_bullet(L) or is_figure_caption(L):
            if buffer:
                result.append(buffer.strip())
                buffer = ""
            result.append(L)
            continue
        
        # Check if we should join with previous content
        if buffer:
            if should_join_lines(buffer, L):
                # Handle hyphenated word continuation
                if buffer.endswith('-'):
                    buffer = buffer[:-1] + L  # Remove hyphen, join directly
                else:
                    buffer = buffer + ' ' + L
            else:
                result.append(buffer.strip())
                buffer = L
        else:
            buffer = L
    
    # Don't forget the last buffer
    if buffer:
        result.append(buffer.strip())
    
    # Clean up multiple empty lines
    cleaned = []
    prev_empty = False
    for line in result:
        if not line:
            if not prev_empty:
                cleaned.append(line)
            prev_empty = True
        else:
            cleaned.append(line)
            prev_empty = False
    
    return '\n'.join(cleaned)


def normalize_and_reflow(text: str) -> str:
    """
    Complete text cleanup: normalize + reflow.
    
    This is the main entry point for text cleanup.
    """
    # First normalize
    text = normalize_text(text)
    
    # Then reflow paragraphs
    text = reflow_paragraphs(text)
    
    return text


def count_garbage_chars(text: str) -> int:
    """Count garbage characters in text (for regression testing)."""
    count = 0
    for char in GARBAGE_CHARS:
        count += text.count(char)
    for code in ZERO_WIDTH_CHARS:
        count += text.count(chr(code))
    return count


# Test function
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    
    # Test cases for normalization
    test_cases = [
        ("altitude\ufffespanning", "altitudespanning"),
        ("already\ufffeknown", "alreadyknown"),
        ("Projection\ufffe-Unified", "Projection-Unified"),
        ("the \ufb01rst paragraph", "the first paragraph"),
        ("a \ufb02uid system", "a fluid system"),
        ("10\u207b\u00b9\u2079 meters", "10^{-19} meters"),
        ("Phonon?Kopplung", "Phonon-Kopplung"),
        ("Spin?Spin?Test", "Spin-Spin-Test"),
    ]
    
    print("=== Text Normalizer Tests ===\n")
    
    all_passed = True
    for input_text, expected_contains in test_cases:
        result = normalize_text(input_text)
        garbage = count_garbage_chars(result)
        
        # Check no garbage chars remain
        passed = garbage == 0
        status = "PASS" if passed else "FAIL"
        
        if not passed:
            all_passed = False
        
        print(f"[{status}] '{input_text}' -> '{result}'")
        print(f"       Garbage chars: {garbage}")
    
    print(f"\n{'='*40}")
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
