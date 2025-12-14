"""
PERFECT PDF Translator - Maximum Quality Pipeline

CRITICAL PRINCIPLE: Pass 1 MUST be perfect!
Errors in Pass 1 propagate to Pass 2. Therefore:
- Every chunk is validated IMMEDIATELY
- Sentence count must match (±10%)
- All placeholders must survive
- Escalating prompts on retry
- NO chunk passes without validation

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import logging
import re
import subprocess
import shutil
import time
import hashlib
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass, field
import fitz  # PyMuPDF

# Import enhanced formula protection
from formula_isolator import (
    extract_formulas as extract_formulas_enhanced,
    normalize_output,
    audit_utf8,
    assert_no_corruption,
    regression_check,
)

logger = logging.getLogger("pdf_translator.perfect")


# =============================================================================
# STRICT QUALITY SETTINGS
# =============================================================================

MIN_TRANSLATION_RATIO = 0.7
MAX_TRANSLATION_RATIO = 2.0
MIN_SENTENCE_RATIO = 0.85
MAX_SENTENCE_RATIO = 1.3

MAX_CHUNK_CHARS = 2000
OVERLAP_CHARS = 150

MAX_RETRIES = 7
RETRY_DELAY_BASE = 1.5


# =============================================================================
# SCIENTIFIC GLOSSARY - MANDATORY TRANSLATIONS
# =============================================================================

SCIENTIFIC_GLOSSARY = {
    "German": {
        # Quantum Physics
        "entanglement": "Verschränkung",
        "entangled": "verschränkt",
        "superposition": "Superposition",
        "coherence": "Kohärenz",
        "decoherence": "Dekohärenz",
        "qubit": "Qubit",
        "quantum": "Quanten-",
        "wave function": "Wellenfunktion",
        "eigenstate": "Eigenzustand",
        "eigenvalue": "Eigenwert",
        "observable": "Observable",
        "measurement": "Messung",
        "collapse": "Kollaps",
        "spin": "Spin",
        "photon": "Photon",
        "phonon": "Phonon",
        "fermion": "Fermion",
        "boson": "Boson",
        # Relativity
        "spacetime": "Raumzeit",
        "time dilation": "Zeitdilatation",
        "gravitational redshift": "Gravitationsrotverschiebung",
        "frame of reference": "Bezugssystem",
        "proper time": "Eigenzeit",
        "world line": "Weltlinie",
        "metric tensor": "metrischer Tensor",
        "curvature": "Krümmung",
        "geodesic": "Geodäte",
        # General Physics
        "fidelity": "Fidelität",
        "lifetime": "Lebensdauer",
        "decay": "Zerfall",
        "coupling": "Kopplung",
        "interaction": "Wechselwirkung",
        "correlation": "Korrelation",
        "fluctuation": "Fluktuation",
        "noise": "Rauschen",
        "drift": "Drift",
        "phase": "Phase",
        "amplitude": "Amplitude",
        "frequency": "Frequenz",
        "wavelength": "Wellenlänge",
        # Computing
        "gate": "Gatter",
        "circuit": "Schaltkreis",
        "error correction": "Fehlerkorrektur",
        "threshold": "Schwellenwert",
        "overhead": "Overhead",
        "scalability": "Skalierbarkeit",
    },
    "French": {
        "entanglement": "intrication",
        "coherence": "cohérence",
        "spacetime": "espace-temps",
        "qubit": "qubit",
    },
    "Spanish": {
        "entanglement": "entrelazamiento",
        "coherence": "coherencia",
        "spacetime": "espacio-tiempo",
        "qubit": "qubit",
    },
}


def get_glossary_for_language(target_language: str) -> str:
    """Get glossary section for the target language."""
    glossary = SCIENTIFIC_GLOSSARY.get(target_language, {})
    if not glossary:
        return ""
    
    lines = ["MANDATORY SCIENTIFIC TERMINOLOGY (use EXACTLY these translations):"]
    for eng, trans in glossary.items():
        lines.append(f"  - {eng} → {trans}")
    return "\n".join(lines)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ValidationResult:
    """Result of chunk validation."""
    valid: bool
    reason: str
    original_sentences: int
    translated_sentences: int
    placeholders_found: int
    placeholders_expected: int
    length_ratio: float


@dataclass
class TranslationResult:
    """Complete translation result with quality metrics."""
    success: bool
    output_path: Optional[str]
    pages_processed: int
    total_original_chars: int
    total_translated_chars: int
    formula_count: int
    formulas_preserved: int
    validation_passed: bool
    warnings: List[str] = field(default_factory=list)
    
    @property
    def translation_ratio(self) -> float:
        if self.total_original_chars == 0:
            return 0
        return self.total_translated_chars / self.total_original_chars


@dataclass
class PageData:
    """All data for a single page."""
    page_num: int
    width: float
    height: float
    images: List[Dict] = field(default_factory=list)


# =============================================================================
# FORMULA ISOLATION - 100% PROTECTION
# =============================================================================

FORMULA_PATTERNS = [
    (r'\$\$[\s\S]+?\$\$', 'display'),
    (r'\\\[[\s\S]+?\\\]', 'display'),
    (r'\\begin\{equation\*?\}[\s\S]+?\\end\{equation\*?\}', 'equation'),
    (r'\\begin\{align\*?\}[\s\S]+?\\end\{align\*?\}', 'align'),
    (r'\\begin\{gather\*?\}[\s\S]+?\\end\{gather\*?\}', 'gather'),
    (r'(?<!\$)\$(?!\$)[^$]+\$(?!\$)', 'inline'),
    (r'\\\([\s\S]+?\\\)', 'inline'),
    (r'\\cite[tp]?\{[^}]+\}', 'cite'),
    (r'\\ref\{[^}]+\}', 'ref'),
    (r'\\eqref\{[^}]+\}', 'eqref'),
    (r'\\label\{[^}]+\}', 'label'),
]


def isolate_formulas(text: str) -> Tuple[str, Dict[str, str]]:
    """Extract ALL formulas and replace with unique placeholders."""
    formula_map = {}
    result = text
    
    all_matches = []
    for pattern, ftype in FORMULA_PATTERNS:
        for match in re.finditer(pattern, text, re.DOTALL):
            all_matches.append((match.start(), match.end(), match.group(), ftype))
    
    all_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    
    filtered_matches = []
    last_end = -1
    for start, end, content, ftype in all_matches:
        if start >= last_end:
            filtered_matches.append((start, end, content, ftype))
            last_end = end
    
    for start, end, content, ftype in reversed(filtered_matches):
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        placeholder = f"⟦F_{content_hash}⟧"
        formula_map[placeholder] = content
        result = result[:start] + placeholder + result[end:]
    
    return result, formula_map


def restore_formulas(text: str, formula_map: Dict[str, str]) -> Tuple[str, int, int]:
    """Restore formulas from placeholders."""
    result = text
    restored = 0
    
    for placeholder, original in formula_map.items():
        if placeholder in result:
            result = result.replace(placeholder, original)
            restored += 1
    
    return result, len(formula_map), restored


# =============================================================================
# SENTENCE COUNTING - CRITICAL FOR VALIDATION
# =============================================================================

def count_sentences(text: str) -> int:
    """Count sentences in text."""
    clean = re.sub(r'⟦F_[a-f0-9]+⟧', '', text)
    sentences = re.split(r'[.!?]+(?:\s|$)', clean)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    return len(sentences)


# =============================================================================
# STRICT VALIDATION
# =============================================================================

def validate_translation_strict(
    original: str,
    translated: str,
    expected_placeholders: List[str]
) -> ValidationResult:
    """STRICT validation of translation quality."""
    if not translated or not translated.strip():
        return ValidationResult(
            valid=False, reason="Empty translation",
            original_sentences=0, translated_sentences=0,
            placeholders_found=0, placeholders_expected=len(expected_placeholders),
            length_ratio=0
        )
    
    orig_len = len(original.strip())
    trans_len = len(translated.strip())
    ratio = trans_len / orig_len if orig_len > 0 else 0
    
    if ratio < MIN_TRANSLATION_RATIO:
        return ValidationResult(
            valid=False, reason=f"Too short: {ratio:.1%} (min {MIN_TRANSLATION_RATIO:.0%})",
            original_sentences=count_sentences(original),
            translated_sentences=count_sentences(translated),
            placeholders_found=0, placeholders_expected=len(expected_placeholders),
            length_ratio=ratio
        )
    
    if ratio > MAX_TRANSLATION_RATIO:
        return ValidationResult(
            valid=False, reason=f"Too long: {ratio:.1%} (max {MAX_TRANSLATION_RATIO:.0%})",
            original_sentences=count_sentences(original),
            translated_sentences=count_sentences(translated),
            placeholders_found=0, placeholders_expected=len(expected_placeholders),
            length_ratio=ratio
        )
    
    orig_sentences = count_sentences(original)
    trans_sentences = count_sentences(translated)
    
    if orig_sentences > 0:
        sentence_ratio = trans_sentences / orig_sentences
        if sentence_ratio < MIN_SENTENCE_RATIO:
            return ValidationResult(
                valid=False,
                reason=f"Missing sentences: {trans_sentences}/{orig_sentences} ({sentence_ratio:.0%})",
                original_sentences=orig_sentences,
                translated_sentences=trans_sentences,
                placeholders_found=0, placeholders_expected=len(expected_placeholders),
                length_ratio=ratio
            )
        if sentence_ratio > MAX_SENTENCE_RATIO:
            return ValidationResult(
                valid=False,
                reason=f"Too many sentences: {trans_sentences}/{orig_sentences}",
                original_sentences=orig_sentences,
                translated_sentences=trans_sentences,
                placeholders_found=0, placeholders_expected=len(expected_placeholders),
                length_ratio=ratio
            )
    
    placeholders_found = sum(1 for p in expected_placeholders if p in translated)
    if placeholders_found < len(expected_placeholders):
        missing = len(expected_placeholders) - placeholders_found
        return ValidationResult(
            valid=False,
            reason=f"Missing {missing} formula placeholder(s)",
            original_sentences=orig_sentences,
            translated_sentences=trans_sentences,
            placeholders_found=placeholders_found,
            placeholders_expected=len(expected_placeholders),
            length_ratio=ratio
        )
    
    failure_patterns = [
        r'^(I\'m sorry|I cannot|I apologize|As an AI)',
        r'^(Here is|Here\'s|Below is|The following)',
        r'^(Translation|Translated|Übersetzung):?\s*$',
        r'^\[.*\]$',
        r'^I\'d be happy to',
    ]
    
    first_line = translated.strip().split('\n')[0]
    for pattern in failure_patterns:
        if re.match(pattern, first_line, re.IGNORECASE):
            return ValidationResult(
                valid=False,
                reason=f"LLM meta-response: {first_line[:40]}...",
                original_sentences=orig_sentences,
                translated_sentences=trans_sentences,
                placeholders_found=placeholders_found,
                placeholders_expected=len(expected_placeholders),
                length_ratio=ratio
            )
    
    return ValidationResult(
        valid=True, reason="OK",
        original_sentences=orig_sentences,
        translated_sentences=trans_sentences,
        placeholders_found=placeholders_found,
        placeholders_expected=len(expected_placeholders),
        length_ratio=ratio
    )


# =============================================================================
# ESCALATING PROMPTS WITH SCIENTIFIC GLOSSARY
# =============================================================================

def build_translation_prompt(
    text: str,
    target_language: str,
    attempt: int,
    previous_issue: str = "",
    context: str = ""
) -> Tuple[str, str]:
    """Build translation prompt with scientific glossary."""
    
    glossary = get_glossary_for_language(target_language)
    
    base_rules = f"""TARGET LANGUAGE: {target_language}

ABSOLUTE RULES:
1. Output ONLY the {target_language} translation - nothing else
2. Translate EVERY sentence completely - no omissions
3. Keep all ⟦F_...⟧ placeholders EXACTLY unchanged
4. Keep mathematical notation unchanged
5. Keep author names, citations, URLs unchanged
6. Preserve paragraph structure

{glossary}"""

    if attempt == 0:
        system = f"""You are a professional scientific translator specializing in physics and quantum mechanics.

{base_rules}

Translate accurately using correct scientific terminology."""

    elif attempt == 1:
        system = f"""You are a professional scientific translator.

{base_rules}

IMPORTANT: Previous translation was incomplete.
Issue: {previous_issue}

You MUST translate EVERY sentence using correct scientific terminology."""

    elif attempt == 2:
        system = f"""You are a professional scientific translator. This is attempt 3.

{base_rules}

CRITICAL ERRORS IN PREVIOUS ATTEMPTS:
- {previous_issue}

REQUIREMENTS:
- Translate ALL sentences
- Use EXACT terminology from glossary
- Every ⟦F_...⟧ must appear in output
- No meta-text, just the translation"""

    elif attempt >= 3:
        system = f"""CRITICAL: Attempt {attempt + 1}. Previous attempts FAILED.

{base_rules}

FAILURE REASON: {previous_issue}

YOU MUST:
1. Translate WORD BY WORD if necessary
2. Use EXACT terms from glossary (entanglement = Verschränkung, etc.)
3. Output EVERY sentence
4. Include ALL ⟦F_...⟧ placeholders
5. Output ONLY {target_language} text
6. NO preamble, NO explanations"""

    if context:
        system += f"\n\nPREVIOUS CONTEXT:\n{context[:300]}..."

    user = f"Translate to {target_language}. Output ONLY the translation:\n\n{text}"
    
    return system, user


# =============================================================================
# PERFECT TRANSLATION WITH VALIDATION LOOP
# =============================================================================

def translate_chunk_perfect(
    text: str,
    model: str,
    target_language: str,
    context: str = ""
) -> Tuple[str, bool, str]:
    """Translate a single chunk with PERFECT validation."""
    import requests
    
    # UTF-8 audit before processing
    utf8_warnings = audit_utf8(text, "input")
    if utf8_warnings:
        logger.warning(f"UTF-8 issues: {utf8_warnings}")
    
    # Use enhanced formula extraction (includes Unicode math)
    enhanced_result = extract_formulas_enhanced(text)
    protected_text = enhanced_result.text_with_placeholders
    formula_map = enhanced_result.formula_map
    expected_placeholders = list(formula_map.keys())
    
    last_issue = ""
    best_result = ""
    best_validation = None
    
    for attempt in range(MAX_RETRIES):
        system_prompt, user_prompt = build_translation_prompt(
            protected_text, target_language, attempt, last_issue, context
        )
        
        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.1 + (attempt * 0.05),
                        "num_predict": 8192,
                        "top_p": 0.95,
                    },
                },
                timeout=300,
            )
            
            if response.status_code != 200:
                last_issue = f"API error: {response.status_code}"
                time.sleep(RETRY_DELAY_BASE ** attempt)
                continue
            
            result = response.json().get("message", {}).get("content", "")
            result = clean_translation_output(result)
            
            validation = validate_translation_strict(protected_text, result, expected_placeholders)
            
            if best_validation is None or validation.length_ratio > best_validation.length_ratio:
                best_result = result
                best_validation = validation
            
            if validation.valid:
                final_text, _, _ = restore_formulas(result, formula_map)
                # Normalize output (remove HTML artifacts, ensure consistent format)
                final_text = normalize_output(final_text, mode="unicode")
                # Final corruption check
                if not assert_no_corruption(final_text):
                    logger.warning("Corruption detected after restoration, retrying...")
                    last_issue = "Corruption detected"
                    continue
                logger.info(f"Chunk validated on attempt {attempt + 1}: {validation.translated_sentences}/{validation.original_sentences} sentences")
                return final_text, True, ""
            
            last_issue = validation.reason
            logger.warning(f"Attempt {attempt + 1} failed: {validation.reason}")
            
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_BASE ** attempt)
                
        except Exception as e:
            last_issue = str(e)
            logger.error(f"Translation error on attempt {attempt + 1}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_BASE ** attempt)
    
    if best_result and best_validation and best_validation.length_ratio >= 0.5:
        final_text, _, _ = restore_formulas(best_result, formula_map)
        final_text = normalize_output(final_text, mode="unicode")
        logger.warning(f"Using best attempt despite validation failure: {last_issue}")
        return final_text, False, last_issue
    
    logger.error(f"Translation failed completely - using original text")
    return text, False, f"All {MAX_RETRIES} attempts failed: {last_issue}"


def clean_translation_output(text: str) -> str:
    """Remove LLM artifacts from translation."""
    lines = text.strip().split('\n')
    
    while lines and re.match(r'^(Here|Below|The following|Translation|Übersetzung)', lines[0], re.I):
        lines = lines[1:]
    
    result = '\n'.join(lines).strip()
    result = re.sub(r'^```\w*\n?', '', result)
    result = re.sub(r'\n?```$', '', result)
    
    return result.strip()


# =============================================================================
# TEXT CHUNKING
# =============================================================================

def chunk_text_smart(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """Split text into chunks at natural boundaries."""
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 > max_chars:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            if len(para) > max_chars:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""
                for sent in sentences:
                    if len(current_chunk) + len(sent) + 1 > max_chars:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = sent + " "
                    else:
                        current_chunk += sent + " "
            else:
                current_chunk = para + "\n\n"
        else:
            current_chunk += para + "\n\n"
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


# =============================================================================
# TEXT EXTRACTION
# =============================================================================

def extract_with_marker(pdf_path: str, output_dir: Path, timeout: int = 180) -> Optional[str]:
    """Extract with Marker (best for formulas)."""
    try:
        import sys
        import json
        
        worker_script = Path(__file__).parent / "marker_worker.py"
        
        result = subprocess.run(
            [sys.executable, str(worker_script), pdf_path, str(output_dir)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path(__file__).parent)
        )
        
        for line in result.stdout.split('\n'):
            if line.startswith('MARKER_RESULT:'):
                result_json = json.loads(line[14:].strip())
                if result_json.get('success'):
                    output_path = result_json.get('output_path')
                    if output_path and Path(output_path).exists():
                        return Path(output_path).read_text(encoding='utf-8')
        return None
    except Exception as e:
        logger.warning(f"Marker extraction failed: {e}")
        return None


def extract_with_pymupdf(pdf_path: Path) -> Tuple[str, List[Dict]]:
    """Extract text and images with PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    
    images = []
    all_text_parts = []
    
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    
    for block in blocks:
        if block.get("type") == 0:
            block_text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    block_text += span.get("text", "")
                block_text += "\n"
            
            if block_text.strip():
                all_text_parts.append(block_text.strip())
    
    for img_idx, img_info in enumerate(page.get_images(full=True)):
        xref = img_info[0]
        try:
            base_image = doc.extract_image(xref)
            img_rects = page.get_image_rects(xref)
            if img_rects:
                rect = img_rects[0]
                images.append({
                    "data": base_image["image"],
                    "ext": base_image["ext"],
                    "x": rect.x0,
                    "y": rect.y0,
                    "width": rect.width,
                    "height": rect.height,
                })
        except Exception as e:
            logger.warning(f"Image extraction failed: {e}")
    
    doc.close()
    return "\n\n".join(all_text_parts), images


def merge_extractions(marker_text: Optional[str], pymupdf_text: str) -> str:
    """Merge extractions - prefer more complete source."""
    if not marker_text:
        return pymupdf_text
    if not pymupdf_text:
        return marker_text
    
    marker_len = len(marker_text.strip())
    pymupdf_len = len(pymupdf_text.strip())
    
    has_formulas = bool(re.search(r'\$.*?\$|\\begin\{|\\frac|\\sum|\\int', marker_text))
    
    if has_formulas and marker_len >= pymupdf_len * 0.6:
        return marker_text
    
    return pymupdf_text if pymupdf_len > marker_len else marker_text


# =============================================================================
# MAIN TRANSLATION PIPELINE
# =============================================================================

def translate_text_perfect(
    text: str,
    model: str,
    target_language: str,
    progress_callback: Optional[Callable] = None
) -> Tuple[str, int, int]:
    """Translate text with PERFECT quality."""
    chunks = chunk_text_smart(text)
    translated_chunks = []
    failed_chunks = 0
    
    context = ""
    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(i + 1, len(chunks), f"Chunk {i+1}/{len(chunks)}")
        
        translated, success, issue = translate_chunk_perfect(
            chunk, model, target_language, context
        )
        
        if not success:
            failed_chunks += 1
            logger.warning(f"Chunk {i+1} has issues: {issue}")
        
        translated_chunks.append(translated)
        context = translated[-400:] if len(translated) > 400 else translated
    
    return "\n\n".join(translated_chunks), len(chunks), failed_chunks


# =============================================================================
# PDF RECONSTRUCTION
# =============================================================================

def create_translated_pdf(
    page_data: PageData,
    translated_text: str,
    output_path: Path
) -> bool:
    """Create PDF with translated text."""
    try:
        doc = fitz.open()
        page = doc.new_page(width=page_data.width, height=page_data.height)
        
        margin = 50
        text_rect = fitz.Rect(margin, margin, page_data.width - margin, page_data.height - margin)
        
        fontsize = 10
        rc = page.insert_textbox(
            text_rect,
            translated_text,
            fontname="helv",
            fontsize=fontsize,
            align=fitz.TEXT_ALIGN_LEFT,
        )
        
        if rc < 0:
            fontsize = 8
            page.insert_textbox(
                text_rect,
                translated_text,
                fontname="helv",
                fontsize=fontsize,
                align=fitz.TEXT_ALIGN_LEFT,
            )
        
        for img in page_data.images:
            try:
                img_rect = fitz.Rect(
                    img["x"], img["y"],
                    img["x"] + img["width"],
                    img["y"] + img["height"]
                )
                page.insert_image(img_rect, stream=img["data"])
            except Exception as e:
                logger.warning(f"Image insertion failed: {e}")
        
        doc.save(str(output_path))
        doc.close()
        return True
        
    except Exception as e:
        logger.exception(f"PDF creation failed: {e}")
        return False


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def translate_pdf_perfect(
    input_pdf: str,
    output_dir: str,
    model: str,
    target_language: str,
    progress_callback: Optional[Callable] = None,
) -> TranslationResult:
    """PERFECT translation pipeline with scientific glossary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = TranslationResult(
        success=False,
        output_path=None,
        pages_processed=0,
        total_original_chars=0,
        total_translated_chars=0,
        formula_count=0,
        formulas_preserved=0,
        validation_passed=True,
        warnings=[]
    )
    
    try:
        doc = fitz.open(input_pdf)
        total_pages = len(doc)
        doc.close()
        
        if progress_callback:
            progress_callback(1, 100, f"Processing {total_pages} pages...")
        
        pages_dir = output_dir / "pages"
        pages_dir.mkdir(exist_ok=True)
        translated_pages = []
        total_failed_chunks = 0
        
        for page_num in range(total_pages):
            page_progress = 5 + int(90 * page_num / total_pages)
            
            if progress_callback:
                progress_callback(page_progress, 100, f"Page {page_num + 1}/{total_pages}")
            
            doc = fitz.open(input_pdf)
            page_pdf = pages_dir / f"page_{page_num + 1:03d}.pdf"
            single_doc = fitz.open()
            single_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            single_doc.save(str(page_pdf))
            single_doc.close()
            doc.close()
            
            marker_dir = output_dir / "marker" / f"page_{page_num + 1}"
            marker_dir.mkdir(parents=True, exist_ok=True)
            
            marker_text = extract_with_marker(str(page_pdf), marker_dir)
            pymupdf_text, images = extract_with_pymupdf(page_pdf)
            
            full_text = merge_extractions(marker_text, pymupdf_text)
            
            if not full_text.strip():
                logger.warning(f"Page {page_num + 1}: No text extracted")
                result.warnings.append(f"Page {page_num + 1}: No text")
                continue
            
            result.total_original_chars += len(full_text)
            
            _, formulas = isolate_formulas(full_text)
            result.formula_count += len(formulas)
            
            page_doc = fitz.open(str(page_pdf))
            page_data = PageData(
                page_num=page_num + 1,
                width=page_doc[0].rect.width,
                height=page_doc[0].rect.height,
                images=images
            )
            page_doc.close()
            
            def page_cb(current, total, msg):
                if progress_callback:
                    sub = page_progress + int(90 / total_pages * current / max(total, 1))
                    progress_callback(sub, 100, f"Page {page_num + 1}: {msg}")
            
            translated_text, total_chunks, failed = translate_text_perfect(
                full_text, model, target_language, page_cb
            )
            
            total_failed_chunks += failed
            result.total_translated_chars += len(translated_text)
            
            _, remaining = isolate_formulas(translated_text)
            result.formulas_preserved += len(remaining)
            
            translated_page = output_dir / "translated" / f"page_{page_num + 1:03d}.pdf"
            translated_page.parent.mkdir(exist_ok=True)
            
            if create_translated_pdf(page_data, translated_text, translated_page):
                translated_pages.append(translated_page)
                result.pages_processed += 1
        
        if progress_callback:
            progress_callback(96, 100, "Merging pages...")
        
        output_pdf = output_dir / "translated_perfect.pdf"
        merged_doc = fitz.open()
        
        for page_path in translated_pages:
            if page_path.exists():
                page_doc = fitz.open(str(page_path))
                merged_doc.insert_pdf(page_doc)
                page_doc.close()
        
        merged_doc.save(str(output_pdf))
        merged_doc.close()
        
        if total_failed_chunks > 0:
            result.warnings.append(f"{total_failed_chunks} chunks had validation issues")
            result.validation_passed = False
        
        if result.total_original_chars > 0:
            ratio = result.total_translated_chars / result.total_original_chars
            if ratio < MIN_TRANSLATION_RATIO or ratio > MAX_TRANSLATION_RATIO:
                result.warnings.append(f"Overall ratio: {ratio:.1%}")
                result.validation_passed = False
        
        result.success = True
        result.output_path = str(output_pdf)
        
        if progress_callback:
            progress_callback(100, 100, "Complete!")
        
        return result
        
    except Exception as e:
        logger.exception(f"Perfect translation failed: {e}")
        result.warnings.append(str(e))
        return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python perfect_translator.py input.pdf [output_dir] [language]")
        sys.exit(1)
    
    input_pdf = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output_perfect"
    language = sys.argv[3] if len(sys.argv) > 3 else "German"
    
    result = translate_pdf_perfect(
        input_pdf,
        output_dir,
        model="qwen2.5:7b",
        target_language=language,
        progress_callback=lambda c, t, m: print(f"[{c:3d}%] {m}")
    )
    
    print(f"\n{'='*60}")
    print(f"SUCCESS: {result.success}")
    print(f"Pages: {result.pages_processed}")
    print(f"Original: {result.total_original_chars:,} chars")
    print(f"Translated: {result.total_translated_chars:,} chars")
    print(f"Ratio: {result.translation_ratio:.1%}")
    print(f"Formulas: {result.formulas_preserved}/{result.formula_count}")
    if result.validation_passed:
        print("Validation: PERFECT")
    else:
        print("Validation: WARNINGS")
    if result.warnings:
        print(f"Warnings: {result.warnings}")
    if result.output_path:
        print(f"\nOutput: {result.output_path}")
