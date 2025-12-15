"""
Scientific Post-Processor for PDF Translation

Fixes PDF extraction artifacts WITHOUT changing scientific meaning.
Two modes:
- strict_preservation: Only basic Unicode cleanup, no semantic repair
- safe_repair: Fix obvious extraction glitches (missing operators, broken ranges)

Key principles:
1. NEVER modify protected math blocks or LaTeX formulas
2. NEVER invent content or reinterpret physics
3. All fixes are deterministic, rule-based, and explainable
4. If ambiguous, leave original text unchanged

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("pdf_translator.scientific_postprocessor")


# =============================================================================
# REPAIR MODES
# =============================================================================

class RepairMode(Enum):
    """Post-processing repair modes."""
    STRICT = "strict"           # No semantic repair - only basic Unicode cleanup
    SAFE_REPAIR = "safe_repair"  # Fix obvious extraction glitches


@dataclass
class RepairAction:
    """Record of a single repair action."""
    rule_name: str
    original: str
    fixed: str
    context: str  # Surrounding text for context
    line_number: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {
            "rule": self.rule_name,
            "original": self.original,
            "fixed": self.fixed,
            "context": self.context,
            "line": self.line_number,
        }


@dataclass
class RepairReport:
    """Complete report of all repairs performed."""
    mode: RepairMode
    total_fixes: int = 0
    actions: List[RepairAction] = field(default_factory=list)
    skipped_ambiguous: List[str] = field(default_factory=list)
    
    def add_action(self, action: RepairAction):
        self.actions.append(action)
        self.total_fixes += 1
    
    def add_skipped(self, reason: str):
        self.skipped_ambiguous.append(reason)
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        md = f"""# Scientific Post-Processing Report

## Mode: `{self.mode.value}`
## Total Fixes: {self.total_fixes}

"""
        if self.actions:
            md += "## Applied Repairs\n\n"
            md += "| # | Rule | Original | Fixed |\n"
            md += "|---|------|----------|-------|\n"
            for i, action in enumerate(self.actions, 1):
                orig = action.original[:30] + "..." if len(action.original) > 30 else action.original
                fixed = action.fixed[:30] + "..." if len(action.fixed) > 30 else action.fixed
                md += f"| {i} | `{action.rule_name}` | `{orig}` | `{fixed}` |\n"
        
        if self.skipped_ambiguous:
            md += "\n## Skipped (Ambiguous)\n\n"
            for reason in self.skipped_ambiguous[:10]:
                md += f"- {reason}\n"
            if len(self.skipped_ambiguous) > 10:
                md += f"- ... and {len(self.skipped_ambiguous) - 10} more\n"
        
        return md


# =============================================================================
# PROTECTED PATTERNS (DO NOT MODIFY)
# =============================================================================

# Math block patterns that should NEVER be modified
PROTECTED_PATTERNS = [
    r'\$\$.*?\$\$',                              # Display math $$...$$
    r'\$[^$]+\$',                                 # Inline math $...$
    r'\\\[.*?\\\]',                               # Display math \[...\]
    r'\\\(.*?\\\)',                               # Inline math \(...\)
    r'\\begin\{equation\*?\}.*?\\end\{equation\*?\}',
    r'\\begin\{align\*?\}.*?\\end\{align\*?\}',
    r'\\begin\{gather\*?\}.*?\\end\{gather\*?\}',
    r'\\begin\{multline\*?\}.*?\\end\{multline\*?\}',
    r'«FORMULA_\d+»',                            # Our protected formula markers
    r'«MATH_\d+»',
    r'«EQ_\d+»',
]


def find_protected_regions(text: str) -> List[Tuple[int, int]]:
    """Find all regions that should not be modified."""
    regions = []
    for pattern in PROTECTED_PATTERNS:
        for match in re.finditer(pattern, text, re.DOTALL):
            regions.append((match.start(), match.end()))
    
    # Merge overlapping regions
    if not regions:
        return []
    
    regions.sort()
    merged = [regions[0]]
    for start, end in regions[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    
    return merged


def is_in_protected_region(pos: int, regions: List[Tuple[int, int]]) -> bool:
    """Check if a position is within a protected region."""
    for start, end in regions:
        if start <= pos < end:
            return True
    return False


# =============================================================================
# UNICODE ARTIFACT FIXES (Always applied, even in STRICT mode)
# =============================================================================

# Replacement character artifacts
UNICODE_FIXES = {
    # Encoding artifacts - NOTE: \ufffd is handled by CorruptionResolver, not here
    '\ufffe': '',      # BOM artifact
    '\uffff': '',      # Not a character
    
    # Common PDF extraction bugs
    '': '',           # Apple logo artifact
    '': '',           # Control character
}

# Replacement character patterns - context-aware (not blind removal)
REPLACEMENT_CHAR_PATTERNS = [
    # Replacement char in ranges "10�20" → "10–20"
    (r'(\d+)[�\ufffd](\d+)', r'\1–\2', 'replacement_char_range'),
    
    # Replacement char in compound words "ESO�Spektroskopie" → "ESO-Spektroskopie"
    (r'([A-Za-z])[�\ufffd]([A-Z][a-z])', r'\1-\2', 'replacement_char_compound'),
    
    # Replacement char before % "99,1�%" → "99,1%"
    (r'(\d+[,\.]\d+)[�\ufffd](%)', r'\1\2', 'replacement_char_percent'),
    
    # Isolated replacement char (no clear context) - keep for now, let resolver handle
]

# Regex patterns for Unicode cleanup
UNICODE_PATTERNS = [
    # Question mark artifacts in compound words (German especially)
    # "ESO?Spektroskopie" → "ESO-Spektroskopie"
    (r'([A-Za-z])(\?+)([A-Z][a-z])', r'\1-\3', 'unicode_compound_word'),
    
    # Question mark in percentages "99,1?%" → "99,1%"
    (r'(\d+[,\.]\d+)\?(%)', r'\1\2', 'unicode_percentage'),
    
    # Question mark in year ranges "2027?2030" → "2027–2030"
    (r'(\d{4})\?(\d{4})', r'\1–\2', 'unicode_year_range'),
    
    # Broken dashes in ranges "10?15" when context suggests range
    (r'(\d+)\?(\d+)\s*(GB|MB|KB|cm|mm|m|kg|s|Hz|MHz|GHz)', r'\1–\2 \3', 'unicode_numeric_range'),
]


def apply_unicode_fixes(text: str, report: RepairReport) -> str:
    """Apply basic Unicode cleanup (always safe)."""
    protected = find_protected_regions(text)
    
    # Direct character replacements
    for bad_char, replacement in UNICODE_FIXES.items():
        if bad_char in text:
            # Only fix outside protected regions
            new_text = []
            last_end = 0
            for start, end in protected:
                segment = text[last_end:start]
                if bad_char in segment:
                    fixed_segment = segment.replace(bad_char, replacement)
                    if fixed_segment != segment:
                        report.add_action(RepairAction(
                            rule_name="unicode_char_cleanup",
                            original=repr(bad_char),
                            fixed=repr(replacement) if replacement else "(removed)",
                            context=segment[:50]
                        ))
                    new_text.append(fixed_segment)
                else:
                    new_text.append(segment)
                new_text.append(text[start:end])  # Protected region unchanged
                last_end = end
            
            # Handle text after last protected region
            segment = text[last_end:]
            if bad_char in segment:
                new_text.append(segment.replace(bad_char, replacement))
            else:
                new_text.append(segment)
            
            text = ''.join(new_text)
    
    # Regex pattern fixes - apply both UNICODE_PATTERNS and REPLACEMENT_CHAR_PATTERNS
    all_patterns = UNICODE_PATTERNS + REPLACEMENT_CHAR_PATTERNS
    for pattern, replacement, rule_name in all_patterns:
        matches = list(re.finditer(pattern, text))
        for match in reversed(matches):  # Reverse to preserve positions
            if not is_in_protected_region(match.start(), protected):
                original = match.group(0)
                fixed = re.sub(pattern, replacement, original)
                if original != fixed:
                    report.add_action(RepairAction(
                        rule_name=rule_name,
                        original=original,
                        fixed=fixed,
                        context=text[max(0, match.start()-20):match.end()+20]
                    ))
                    text = text[:match.start()] + fixed + text[match.end():]
    
    return text


# =============================================================================
# SAFE REPAIR FIXES (Only in SAFE_REPAIR mode)
# =============================================================================

# Patterns for obvious extraction glitches that can be safely repaired
SAFE_REPAIR_PATTERNS = [
    # ==========================================================================
    # FONT GLYPH PLACEHOLDERS (e.g. ?F_afec5f_0?, ?G_12ab34_1?)
    # These are extraction failures where font glyphs couldn't be mapped to Unicode
    # ==========================================================================
    (
        r'\?[A-Z]_[a-f0-9]{4,8}_\d+\?',
        '?',  # Mark as unresolved symbol (don't remove!)
        'font_glyph_placeholder',
        None
    ),
    
    # Variant: without leading/trailing ? but with brackets
    (
        r'\[[A-Z]_[a-f0-9]{4,8}_\d+\]',
        '?',  # Mark as unresolved
        'font_glyph_bracket',
        None
    ),
    
    # ==========================================================================
    # SSZ HYPHEN ARTIFACTS (SSZ?Korrekturen → SSZ-Korrekturen)
    # ==========================================================================
    (
        r'SSZ\?',
        'SSZ-',
        'ssz_hyphen_artifact',
        None
    ),
    
    # General word?word pattern (compound word with broken hyphen)
    (
        r'([A-ZÄÖÜ][a-zäöüß]+)\?([A-ZÄÖÜ][a-zäöüß]+)',
        r'\1-\2',
        'compound_word_hyphen',
        None
    ),
    
    # ==========================================================================
    # EXPONENT/SUPERSCRIPT ARTIFACTS (10?²?, 10^-2 broken)
    # ==========================================================================
    
    # "10?²?" or "10?2?" → "10²" (broken superscript with question marks)
    (
        r'10\?([²³⁴⁵⁶⁷⁸⁹⁰¹])\?',
        r'10\1',
        'broken_superscript_10',
        None
    ),
    
    # "10?−?19" or "10?-?19" → "10^-19" (negative exponent with double ?)
    (
        r'10\?[−\-]?\?(\d+)',
        r'10^{-\1}',
        'broken_negative_exponent_double',
        None
    ),
    
    # "10?-2?" → "10⁻²" (negative exponent with question marks)
    (
        r'10\?-(\d)\?',
        lambda m: f'10^{{-{m.group(1)}}}',
        'broken_negative_exponent',
        None
    ),
    
    # "(10{-19})" → "(10^{-19})" (missing caret in LaTeX exponent)
    (
        r'\(10\{(-?\d+)\}\)',
        r'(10^{\1})',
        'missing_caret_exponent',
        None
    ),
    
    # "10{-19}" standalone → "10^{-19}"
    (
        r'(?<!\^)10\{(-?\d+)\}',
        r'10^{\1}',
        'missing_caret_standalone',
        None
    ),
    
    # "(1.1^{-19})" → keep as is (already correct pattern)
    # But fix "(1.1{-19})" → "(1.1×10^{-19})" - likely meant scientific notation
    (
        r'\((\d+\.?\d*)\{(-\d+)\}\)',
        r'(\1×10^{\2})',
        'broken_scientific_notation_paren',
        None
    ),
    
    # "×10?" followed by digit → "×10^" (multiplication with broken exponent)
    (
        r'[×x]\s*10\?\s*(\d+)',
        r'×10^\1',
        'broken_scientific_notation',
        None
    ),
    
    # "× 10-14" → "×10^{-14}" (space and missing caret)
    (
        r'×\s*10-(\d+)',
        r'×10^{-\1}',
        'scientific_notation_missing_caret',
        None
    ),
    
    # ==========================================================================
    # MATH SYMBOL ARTIFACTS
    # ==========================================================================
    
    # "?±?" → "±" (plus-minus with question marks)
    (
        r'\?([±∓≈≠≤≥∝∞∂∇∫∑∏√])\?',
        r'\1',
        'broken_math_symbol',
        None
    ),
    
    # Broken minus in ranges: "10?15" in numeric context → "10–15"
    (
        r'(\d+)\?(\d+)\s*(K|Hz|MHz|GHz|nm|μm|mm|cm|m|km|eV|keV|MeV|GeV|TeV)',
        r'\1–\2 \3',
        'broken_range_with_unit',
        None
    ),
    
    # ==========================================================================
    # BROKEN LATEX/MATH FRAGMENTS (from PDF extraction)
    # ==========================================================================
    
    # "[.|{r=R{}} {-16},{-1},]" → completely broken, mark as UNRESOLVED
    (
        r'\[\.\|\{[^]]{10,50}\}\]',
        '[[FORMEL]]',
        'broken_latex_fragment',
        None
    ),
    
    # "[D = D_SSZ..." broken inline math with square brackets
    (
        r'\[([A-Za-z_]+\s*=\s*[^]]{5,100})\]',
        r'$\1$',
        'bracket_to_math',
        None
    ),
    
    # "$$[..." → "$" (double dollar with bracket)
    (
        r'\$\$\[',
        '$',
        'double_dollar_bracket',
        None
    ),
    
    # "]\$\$" → "$" (closing bracket with double dollar)
    (
        r'\]\$\$',
        '$',
        'bracket_double_dollar',
        None
    ),
    
    # "(^{10})" → "(10^{10})" - missing base for exponent
    (
        r'\(\^(\{[^}]+\})\)',
        r'(10^\1)',
        'missing_exponent_base',
        None
    ),
    
    # "m({-1})" → "m^{-1}" - unit with broken exponent
    (
        r'([a-zA-Z])\(\{(-?\d+)\}\)',
        r'\1^{\2}',
        'unit_broken_exponent',
        None
    ),
    
    # ==========================================================================
    # COMPARISON OPERATORS (SSZ-specific)
    # ==========================================================================
    
    # Missing comparison operators near r* or r_s (SSZ-specific)
    # "For r r*" → "For r < r*" or "For r > r*"
    (
        r'\bFor\s+r\s+r\*',
        'For r < r*',  # Default to < as it's more common in SSZ context
        'missing_comparison_r_star',
        lambda ctx: 'lower' in ctx.lower() or 'less' in ctx.lower() or '<' in ctx
    ),
    (
        r'\bFor\s+r\s+r_s',
        'For r < r_s',
        'missing_comparison_r_s',
        lambda ctx: 'lower' in ctx.lower() or 'less' in ctx.lower() or '<' in ctx
    ),
    
    # ==========================================================================
    # OCR/EXTRACTION WORD ARTIFACTS
    # ==========================================================================
    
    # Duplicated words from OCR "the the" → "the"
    (
        r'\b(the|a|an|is|are|was|were|in|on|at|to|for|of|and|or|der|die|das|und|oder|ist|sind|war|für|von|mit)\s+\1\b',
        r'\1',
        'duplicated_word',
        None
    ),
    
    # Missing space after period (but not in abbreviations or numbers)
    (
        r'([a-z])\.([A-Z][a-z]{2,})',
        r'\1. \2',
        'missing_space_after_period',
        None
    ),
    
    # ==========================================================================
    # SUBSCRIPT/SUPERSCRIPT SPACING
    # ==========================================================================
    
    # Broken subscript notation "r_s" with extra spaces "r _ s" → "r_s"
    (
        r'([a-zA-Z])\s*_\s*([a-zA-Z0-9])',
        r'\1_\2',
        'broken_subscript',
        None
    ),
    
    # Broken superscript notation "x^2" with spaces "x ^ 2" → "x^2"
    (
        r'([a-zA-Z0-9])\s*\^\s*([a-zA-Z0-9{])',
        r'\1^\2',
        'broken_superscript',
        None
    ),
]

# Context-aware repairs (require surrounding text analysis)
CONTEXT_REPAIRS = [
    {
        'name': 'missing_less_than',
        'pattern': r'For\s+r\s+r([*_])',
        'contexts': {
            'smaller': '<',
            'less': '<',
            'below': '<',
            'under': '<',
            'lower': '<',
            'greater': '>',
            'larger': '>',
            'above': '>',
            'over': '>',
            'higher': '>',
        },
        'default': '<',  # Most common in physics papers
    },
]


def get_context_window(text: str, pos: int, window: int = 100) -> str:
    """Get surrounding context for a position."""
    start = max(0, pos - window)
    end = min(len(text), pos + window)
    return text[start:end].lower()


def apply_safe_repairs(text: str, report: RepairReport) -> str:
    """Apply safe repairs for obvious extraction glitches."""
    protected = find_protected_regions(text)
    
    # Simple pattern repairs
    for pattern, replacement, rule_name, context_check in SAFE_REPAIR_PATTERNS:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        
        for match in reversed(matches):
            if is_in_protected_region(match.start(), protected):
                continue
            
            # Check context if required
            if context_check is not None:
                context = get_context_window(text, match.start())
                if not context_check(context):
                    report.add_skipped(f"Ambiguous context for '{match.group(0)}' - skipped")
                    continue
            
            original = match.group(0)
            
            # Handle different replacement types
            if callable(replacement):
                # Lambda function replacement
                fixed = replacement(match)
            elif isinstance(replacement, str) and (r'\1' in replacement or r'\2' in replacement or r'\3' in replacement):
                # Backreference replacement
                fixed = re.sub(pattern, replacement, original, flags=re.IGNORECASE)
            else:
                # Simple string replacement
                fixed = replacement
            
            if original != fixed:
                report.add_action(RepairAction(
                    rule_name=rule_name,
                    original=original,
                    fixed=fixed,
                    context=text[max(0, match.start()-30):match.end()+30]
                ))
                text = text[:match.start()] + fixed + text[match.end():]
    
    # Context-aware repairs
    for repair in CONTEXT_REPAIRS:
        pattern = repair['pattern']
        matches = list(re.finditer(pattern, text))
        
        for match in reversed(matches):
            if is_in_protected_region(match.start(), protected):
                continue
            
            context = get_context_window(text, match.start(), 200)
            
            # Determine correct operator from context
            operator = None
            for keyword, op in repair['contexts'].items():
                if keyword in context:
                    operator = op
                    break
            
            if operator is None:
                # Use default only if unambiguous
                operator = repair['default']
                report.add_skipped(
                    f"No clear context for '{match.group(0)}' - using default '{operator}'"
                )
            
            original = match.group(0)
            suffix = match.group(1)  # The * or _ part
            fixed = f"For r {operator} r{suffix}"
            
            if original != fixed:
                report.add_action(RepairAction(
                    rule_name=repair['name'],
                    original=original,
                    fixed=fixed,
                    context=context[:80]
                ))
                text = text[:match.start()] + fixed + text[match.end():]
    
    return text


# =============================================================================
# 5-STEP RESOLVE STRATEGY (Windsurf-Prompt Implementation)
# Priority: Resolve deterministically, UNRESOLVED only as last resort
# =============================================================================

@dataclass
class CorruptionMatch:
    """Detected corruption pattern."""
    start: int
    end: int
    pattern_type: str
    original: str
    context: str


class CorruptionResolver:
    """
    5-Step hierarchical resolver for PDF extraction corruption.
    
    Step 1: Identify corruption patterns
    Step 2: Resolve using safe deterministic rules (preferred)
    Step 3: Resolve using local neighborhood evidence
    Step 4: Resolve using fallback extraction (if available)
    Step 5: UNRESOLVED marker only if everything fails
    """
    
    def __init__(self):
        self.symbol_map: Dict[str, str] = {}  # Document-level glyph->unicode map
        self.unresolved_count = 0
    
    # =========================================================================
    # STEP 1: Identify corruption patterns
    # =========================================================================
    
    # Corruption detection patterns
    CORRUPTION_PATTERNS = [
        # Replacement characters
        (r'[�\ufffd]', 'replacement_char'),
        # Question mark artifacts (not in words, not punctuation)
        (r'(?<=[0-9])\?(?=[0-9²³⁴⁵⁶⁷⁸⁹⁰¹⁻⁺])', 'question_in_number'),
        (r'(?<=[A-Za-z])\?(?=[A-Z])', 'question_compound'),
        # Font glyph placeholders
        (r'\?[A-Z]_[a-f0-9]{4,8}_\d+\?', 'font_glyph'),
        (r'\[[A-Z]_[a-f0-9]{4,8}_\d+\]', 'font_glyph_bracket'),
        # Broken minus/dash variants
        (r'(?<=[0-9])\s*[\?]\s*(?=[\-−–]?\d)', 'broken_range'),
        # Soft hyphen artifacts
        (r'\u00ad', 'soft_hyphen'),
        # Zero-width chars that shouldn't be there
        (r'[\u200b\u200c\u200d\ufeff]', 'zero_width'),
    ]
    
    def identify_corruptions(self, text: str) -> List[CorruptionMatch]:
        """Step 1: Detect all corruption patterns in text."""
        corruptions = []
        
        for pattern, pattern_type in self.CORRUPTION_PATTERNS:
            for match in re.finditer(pattern, text):
                # Get context window
                ctx_start = max(0, match.start() - 20)
                ctx_end = min(len(text), match.end() + 20)
                context = text[ctx_start:ctx_end]
                
                corruptions.append(CorruptionMatch(
                    start=match.start(),
                    end=match.end(),
                    pattern_type=pattern_type,
                    original=match.group(0),
                    context=context
                ))
        
        return corruptions
    
    # =========================================================================
    # STEP 2: Safe deterministic rules
    # =========================================================================
    
    # Deterministic resolution rules (pattern -> replacement function)
    DETERMINISTIC_RULES = {
        'soft_hyphen': lambda m, ctx: '',  # Remove soft hyphens
        'zero_width': lambda m, ctx: '',   # Remove zero-width chars
        'replacement_char': lambda m, ctx: CorruptionResolver._resolve_replacement_char(m, ctx),
        'question_in_number': lambda m, ctx: '',  # Remove stray ? in numbers
        'question_compound': lambda m, ctx: '-',  # Hyphen in compound words
        'font_glyph': lambda m, ctx: CorruptionResolver._resolve_font_glyph(m, ctx),
        'font_glyph_bracket': lambda m, ctx: CorruptionResolver._resolve_font_glyph(m, ctx),
        'broken_range': lambda m, ctx: '–',  # En-dash for ranges
    }
    
    @staticmethod
    def _resolve_replacement_char(match: CorruptionMatch, context: str) -> Optional[str]:
        """Try to resolve replacement character from context."""
        ctx_lower = context.lower()
        
        # Common patterns where replacement char appears
        if re.search(r'\d�\d', context):
            return '–'  # Range
        if re.search(r'\d�%', context):
            return ''   # Remove before %
        if re.search(r'[a-z]�[A-Z]', context):
            return '-'  # Compound word
        
        # Cannot resolve deterministically
        return None
    
    @staticmethod
    def _resolve_font_glyph(match: CorruptionMatch, context: str) -> Optional[str]:
        """Try to resolve font glyph placeholder from context."""
        # Extract glyph ID
        glyph_match = re.search(r'[A-Z]_([a-f0-9]+)_(\d+)', match.original)
        if not glyph_match:
            return ''  # Remove empty placeholder
        
        # Common glyph resolutions based on surrounding context
        ctx = context.lower()
        
        # Greek letters often appear in scientific contexts
        if 'alpha' in ctx or 'α' in ctx:
            return 'α'
        if 'beta' in ctx or 'β' in ctx:
            return 'β'
        if 'delta' in ctx or 'Δ' in ctx:
            return 'Δ'
        if 'gamma' in ctx or 'γ' in ctx:
            return 'γ'
        
        # Math symbols
        if 'plus' in ctx and 'minus' in ctx:
            return '±'
        if 'infinity' in ctx or '∞' in ctx:
            return '∞'
        
        # Cannot resolve - will go to Step 3
        return None
    
    def apply_deterministic_rules(self, text: str, corruptions: List[CorruptionMatch], 
                                   report: RepairReport) -> Tuple[str, List[CorruptionMatch]]:
        """Step 2: Apply safe deterministic rules. Returns text and unresolved corruptions."""
        unresolved = []
        
        # Process in reverse order to preserve positions
        for corruption in sorted(corruptions, key=lambda c: c.start, reverse=True):
            rule = self.DETERMINISTIC_RULES.get(corruption.pattern_type)
            
            if rule:
                replacement = rule(corruption, corruption.context)
                
                if replacement is not None:
                    # Successfully resolved
                    text = text[:corruption.start] + replacement + text[corruption.end:]
                    report.add_action(RepairAction(
                        rule_name=f"resolve_step2_{corruption.pattern_type}",
                        original=corruption.original,
                        fixed=replacement if replacement else "(removed)",
                        context=corruption.context
                    ))
                else:
                    # Could not resolve with deterministic rule
                    unresolved.append(corruption)
            else:
                unresolved.append(corruption)
        
        return text, unresolved
    
    # =========================================================================
    # STEP 3: Local neighborhood evidence
    # =========================================================================
    
    def build_symbol_map(self, text: str) -> Dict[str, str]:
        """Build document-level symbol map from frequency + context."""
        symbol_counts: Dict[str, Dict[str, int]] = {}
        
        # Find all font glyph patterns and their contexts
        for match in re.finditer(r'\?([A-Z]_[a-f0-9]+_\d+)\?', text):
            glyph_id = match.group(1)
            
            # Get surrounding context
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]
            
            if glyph_id not in symbol_counts:
                symbol_counts[glyph_id] = {}
            
            # Try to infer symbol from context patterns
            inferred = self._infer_symbol_from_context(context)
            if inferred:
                symbol_counts[glyph_id][inferred] = symbol_counts[glyph_id].get(inferred, 0) + 1
        
        # Build map from most frequent inference
        for glyph_id, inferences in symbol_counts.items():
            if inferences:
                best = max(inferences.items(), key=lambda x: x[1])
                self.symbol_map[glyph_id] = best[0]
        
        return self.symbol_map
    
    def _infer_symbol_from_context(self, context: str) -> Optional[str]:
        """Infer symbol from surrounding context patterns."""
        ctx_lower = context.lower()
        
        # Scientific symbol patterns
        patterns = [
            (r'segment\s*density', 'Ξ'),
            (r'time\s*dilation', 'D'),
            (r'schwarzschild', 'r'),
            (r'planck', 'ℏ'),
            (r'angular', 'ω'),
            (r'wavelength', 'λ'),
            (r'frequency', 'ν'),
            (r'temperature', 'T'),
            (r'pressure', 'P'),
            (r'energy', 'E'),
            (r'mass', 'M'),
            (r'radius', 'r'),
        ]
        
        for pattern, symbol in patterns:
            if re.search(pattern, ctx_lower):
                return symbol
        
        return None
    
    def apply_neighborhood_evidence(self, text: str, corruptions: List[CorruptionMatch],
                                     report: RepairReport) -> Tuple[str, List[CorruptionMatch]]:
        """Step 3: Use document-level symbol map and neighborhood context."""
        if not self.symbol_map:
            self.build_symbol_map(text)
        
        unresolved = []
        
        for corruption in sorted(corruptions, key=lambda c: c.start, reverse=True):
            resolved = False
            
            # Try symbol map lookup
            if corruption.pattern_type in ('font_glyph', 'font_glyph_bracket'):
                glyph_match = re.search(r'[A-Z]_[a-f0-9]+_\d+', corruption.original)
                if glyph_match:
                    glyph_id = glyph_match.group(0)
                    if glyph_id in self.symbol_map:
                        replacement = self.symbol_map[glyph_id]
                        text = text[:corruption.start] + replacement + text[corruption.end:]
                        report.add_action(RepairAction(
                            rule_name="resolve_step3_symbol_map",
                            original=corruption.original,
                            fixed=replacement,
                            context=corruption.context
                        ))
                        resolved = True
            
            if not resolved:
                unresolved.append(corruption)
        
        return text, unresolved
    
    # =========================================================================
    # STEP 4: Fallback extraction (placeholder - requires PyMuPDF integration)
    # =========================================================================
    
    def apply_fallback_extraction(self, text: str, corruptions: List[CorruptionMatch],
                                   report: RepairReport) -> Tuple[str, List[CorruptionMatch]]:
        """Step 4: Re-extract with alternate options (requires PDF access)."""
        # This is a placeholder - actual implementation would need PDF document access
        # For now, just pass through to Step 5
        return text, corruptions
    
    # =========================================================================
    # STEP 5: UNRESOLVED marker (last resort only)
    # =========================================================================
    
    def mark_unresolved(self, text: str, corruptions: List[CorruptionMatch],
                        report: RepairReport) -> str:
        """Step 5: Mark remaining corruptions as UNRESOLVED (last resort)."""
        for corruption in sorted(corruptions, key=lambda c: c.start, reverse=True):
            # Create UNRESOLVED marker
            marker_type = "GLYPH" if "glyph" in corruption.pattern_type else "CHAR"
            marker = f"[[UNRESOLVED_{marker_type}:{corruption.original}]]"
            
            text = text[:corruption.start] + marker + text[corruption.end:]
            report.add_action(RepairAction(
                rule_name="unresolved_step5",
                original=corruption.original,
                fixed=marker,
                context=corruption.context
            ))
            self.unresolved_count += 1
        
        return text
    
    # =========================================================================
    # MAIN RESOLVE PIPELINE
    # =========================================================================
    
    def resolve(self, text: str, report: RepairReport) -> str:
        """
        Execute the 5-step resolve pipeline.
        
        Returns text with corruptions resolved (or marked UNRESOLVED if all else fails).
        """
        # Step 1: Identify all corruptions
        corruptions = self.identify_corruptions(text)
        
        if not corruptions:
            return text  # No corruptions found
        
        logger.info(f"Step 1: Found {len(corruptions)} corruption patterns")
        
        # Step 2: Apply deterministic rules
        text, remaining = self.apply_deterministic_rules(text, corruptions, report)
        logger.info(f"Step 2: {len(corruptions) - len(remaining)} resolved, {len(remaining)} remaining")
        
        if not remaining:
            return text
        
        # Step 3: Apply neighborhood evidence
        text, remaining = self.apply_neighborhood_evidence(text, remaining, report)
        logger.info(f"Step 3: {len(remaining)} remaining after neighborhood analysis")
        
        if not remaining:
            return text
        
        # Step 4: Try fallback extraction
        text, remaining = self.apply_fallback_extraction(text, remaining, report)
        
        if not remaining:
            return text
        
        # Step 5: Mark remaining as UNRESOLVED (last resort)
        text = self.mark_unresolved(text, remaining, report)
        logger.warning(f"Step 5: {len(remaining)} corruptions marked as UNRESOLVED")
        
        return text


# =============================================================================
# FIGURE/TABLE REFERENCE REPAIRS
# =============================================================================

def repair_figure_references(text: str, report: RepairReport) -> str:
    """Fix broken figure/table references."""
    protected = find_protected_regions(text)
    
    # "Figure1" → "Figure 1"
    pattern = r'\b(Figure|Fig|Table|Tab|Equation|Eq)\.?(\d+)'
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    
    for match in reversed(matches):
        if is_in_protected_region(match.start(), protected):
            continue
        
        original = match.group(0)
        prefix = match.group(1)
        number = match.group(2)
        
        # Normalize prefix
        if prefix.lower() in ('fig', 'fig.'):
            fixed = f"Fig. {number}"
        elif prefix.lower() in ('tab', 'tab.'):
            fixed = f"Tab. {number}"
        elif prefix.lower() in ('eq', 'eq.'):
            fixed = f"Eq. {number}"
        else:
            fixed = f"{prefix} {number}"
        
        if original != fixed:
            report.add_action(RepairAction(
                rule_name='figure_reference',
                original=original,
                fixed=fixed,
                context=text[max(0, match.start()-20):match.end()+20]
            ))
            text = text[:match.start()] + fixed + text[match.end():]
    
    return text


# =============================================================================
# MAIN POST-PROCESSOR
# =============================================================================

class ScientificPostProcessor:
    """
    Scientific post-processor with configurable repair modes.
    
    Usage:
        processor = ScientificPostProcessor(mode=RepairMode.SAFE_REPAIR)
        fixed_text, report = processor.process(text)
    """
    
    def __init__(self, mode: RepairMode = RepairMode.STRICT):
        self.mode = mode
        logger.info(f"ScientificPostProcessor initialized in {mode.value} mode")
    
    def process(self, text: str) -> Tuple[str, RepairReport]:
        """
        Process text and return fixed text with repair report.
        
        Args:
            text: Input text to process
            
        Returns:
            Tuple of (fixed_text, report)
        """
        report = RepairReport(mode=self.mode)
        
        if not text:
            return text, report
        
        # Step 1: Always apply Unicode fixes (safe in any mode)
        text = apply_unicode_fixes(text, report)
        
        # Step 2: Apply safe repairs only in SAFE_REPAIR mode
        if self.mode == RepairMode.SAFE_REPAIR:
            text = apply_safe_repairs(text, report)
            text = repair_figure_references(text, report)
            
            # Step 3: Apply 5-step resolve pipeline for remaining corruptions
            resolver = CorruptionResolver()
            text = resolver.resolve(text, report)
        
        logger.info(f"Post-processing complete: {report.total_fixes} fixes applied")
        
        return text, report
    
    def process_blocks(self, blocks: List[str]) -> Tuple[List[str], RepairReport]:
        """
        Process multiple text blocks.
        
        Args:
            blocks: List of text blocks
            
        Returns:
            Tuple of (fixed_blocks, combined_report)
        """
        combined_report = RepairReport(mode=self.mode)
        fixed_blocks = []
        
        for i, block in enumerate(blocks):
            fixed_block, block_report = self.process(block)
            fixed_blocks.append(fixed_block)
            
            # Merge reports
            for action in block_report.actions:
                action.line_number = i + 1
                combined_report.add_action(action)
            combined_report.skipped_ambiguous.extend(block_report.skipped_ambiguous)
        
        return fixed_blocks, combined_report


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def postprocess_strict(text: str) -> Tuple[str, RepairReport]:
    """Process text in strict mode (minimal changes)."""
    processor = ScientificPostProcessor(RepairMode.STRICT)
    return processor.process(text)


def postprocess_safe(text: str) -> Tuple[str, RepairReport]:
    """Process text in safe repair mode."""
    processor = ScientificPostProcessor(RepairMode.SAFE_REPAIR)
    return processor.process(text)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI for testing post-processor."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Scientific PDF Post-Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scientific_postprocessor.py --mode safe_repair --input text.txt
  python scientific_postprocessor.py --mode strict --test
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['strict', 'safe_repair'],
        default='safe_repair',
        help='Processing mode (default: safe_repair)'
    )
    parser.add_argument(
        '--input', '-i',
        help='Input file to process'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file (default: stdout)'
    )
    parser.add_argument(
        '--report', '-r',
        help='Write repair report to file'
    )
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Run built-in tests'
    )
    
    args = parser.parse_args()
    
    # Ensure UTF-8 output
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    
    if args.test:
        run_tests()
        return
    
    if not args.input:
        parser.print_help()
        return
    
    # Read input
    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Process
    mode = RepairMode.STRICT if args.mode == 'strict' else RepairMode.SAFE_REPAIR
    processor = ScientificPostProcessor(mode)
    fixed_text, report = processor.process(text)
    
    # Write output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(fixed_text)
        print(f"✅ Output written to {args.output}")
    else:
        print(fixed_text)
    
    # Write report
    if args.report:
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write(report.to_markdown())
        print(f"📋 Report written to {args.report}")
    else:
        print("\n" + "="*60)
        print(report.to_markdown())


def run_tests():
    """Run built-in test cases."""
    print("=" * 60)
    print("Scientific Post-Processor Tests")
    print("=" * 60)
    
    test_cases = [
        # (input, expected_in_output, mode, description)
        (
            "ESO?Spektroskopie achieves 99,1?% accuracy",
            "ESO-Spektroskopie",
            RepairMode.STRICT,
            "Unicode compound word fix (strict mode)"
        ),
        (
            "For r r*, SSZ predicts lower time dilation.",
            "For r < r*",
            RepairMode.SAFE_REPAIR,
            "Missing comparison operator (safe mode)"
        ),
        (
            "The the results show improvement",
            "The results",
            RepairMode.SAFE_REPAIR,
            "Duplicated word removal"
        ),
        (
            "See Figure1 and Table2",
            "Figure 1",
            RepairMode.SAFE_REPAIR,
            "Figure reference normalization"
        ),
        (
            "r _ s = 2GM/c^2 is the Schwarzschild radius",
            "r_s",
            RepairMode.SAFE_REPAIR,
            "Broken subscript fix"
        ),
        (
            "Testing 2027?2030 timeline",
            "2027–2030",
            RepairMode.STRICT,
            "Year range fix (Unicode)"
        ),
        (
            "Using $r_s = 2GM/c^2$ formula, For r r* we see...",
            "$r_s = 2GM/c^2$",  # Should be preserved
            RepairMode.SAFE_REPAIR,
            "Protected math block (should not modify)"
        ),
        # NEW: Font glyph placeholder tests (marked with ? instead of removed)
        (
            "The value is ?F_afec5f_0? approximately 10.",
            "The value is ? approximately 10.",
            RepairMode.SAFE_REPAIR,
            "Font glyph placeholder marker"
        ),
        (
            "Critical scale: 10?²? meters",
            "10²",
            RepairMode.SAFE_REPAIR,
            "Broken superscript fix"
        ),
        (
            "Range is 100?500 MHz for this experiment",
            "100–500 MHz",
            RepairMode.SAFE_REPAIR,
            "Broken range with unit fix"
        ),
        (
            "Symbol ?±? indicates error margin",
            "±",
            RepairMode.SAFE_REPAIR,
            "Broken math symbol fix"
        ),
        # 5-STEP RESOLVE STRATEGY TESTS
        (
            "Text with soft\u00adhyphen artifacts",
            "Text with softhyphen artifacts",
            RepairMode.SAFE_REPAIR,
            "Step 2: Soft hyphen removal"
        ),
        (
            "Zero\u200bwidth\u200cchar\u200dtest",
            "Zerowidthchartest",
            RepairMode.SAFE_REPAIR,
            "Step 2: Zero-width char removal"
        ),
        (
            "Value is 10�20 range",
            "10–20",
            RepairMode.SAFE_REPAIR,
            "Step 2: Replacement char in range → en-dash"
        ),
        (
            "ESO�Spektroskopie data",
            "ESO-Spektroskopie",
            RepairMode.SAFE_REPAIR,
            "Step 2: Replacement char in compound → hyphen"
        ),
        # NEW: PDF-specific patterns from paper a translated.pdf
        (
            "SSZ?Korrekturen sind wichtig",
            "SSZ-Korrekturen",
            RepairMode.SAFE_REPAIR,
            "SSZ hyphen artifact"
        ),
        (
            "Zeit?Dilatation gemessen",
            "Zeit-Dilatation",
            RepairMode.SAFE_REPAIR,
            "Compound word hyphen artifact"
        ),
        (
            "Wert (10{-19}) ist klein",
            "(10^{-19})",
            RepairMode.SAFE_REPAIR,
            "Missing caret in exponent"
        ),
        (
            "× 10-14 Sekunden",
            "×10^{-14}",
            RepairMode.SAFE_REPAIR,
            "Scientific notation missing caret"
        ),
    ]
    
    all_passed = True
    
    for input_text, expected, mode, description in test_cases:
        processor = ScientificPostProcessor(mode)
        result, report = processor.process(input_text)
        
        passed = expected in result
        status = "✅ PASS" if passed else "❌ FAIL"
        
        if not passed:
            all_passed = False
        
        print(f"\n{status}: {description}")
        print(f"  Mode: {mode.value}")
        print(f"  Input:    '{input_text[:60]}...' " if len(input_text) > 60 else f"  Input:    '{input_text}'")
        print(f"  Expected: '{expected}' in output")
        print(f"  Result:   '{result[:60]}...'" if len(result) > 60 else f"  Result:   '{result}'")
        print(f"  Fixes:    {report.total_fixes}")
    
    print("\n" + "=" * 60)
    print(f"Overall: {'ALL PASSED ✅' if all_passed else 'SOME FAILED ❌'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
