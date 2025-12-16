"""
PDF Translation with Marker - Perfect Scientific PDF Translation

Uses Marker to convert PDF to Markdown with preserved formulas,
then translates and converts back to PDF.

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List
import requests
from text_normalizer import normalize_text, normalize_and_reflow

logger = logging.getLogger("pdf_translator.marker")


def unload_ollama_model():
    """Unload Ollama models to free VRAM before loading Marker."""
    try:
        # Get loaded models
        response = requests.get("http://localhost:11434/api/ps", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            if not models:
                logger.info("No Ollama models loaded in VRAM")
                return
            
            for model in models:
                model_name = model.get("name", "")
                if model_name:
                    logger.info(f"Unloading Ollama model: {model_name}")
                    # Send request with keep_alive=0 to unload immediately
                    try:
                        resp = requests.post(
                            "http://localhost:11434/api/generate",
                            json={"model": model_name, "keep_alive": 0, "prompt": ""},
                            timeout=30
                        )
                        logger.info(f"Unload request sent for {model_name}: {resp.status_code}")
                    except Exception as e:
                        logger.warning(f"Failed to unload {model_name}: {e}")
            
            # Verify unload
            import time
            time.sleep(3)
            verify = requests.get("http://localhost:11434/api/ps", timeout=5)
            if verify.status_code == 200:
                remaining = verify.json().get("models", [])
                if remaining:
                    logger.warning(f"Still {len(remaining)} models in VRAM after unload attempt")
                else:
                    logger.info("All Ollama models unloaded - VRAM free for Marker")
    except Exception as e:
        logger.debug(f"Could not unload Ollama model: {e}")


def check_marker_models_available() -> bool:
    """Check if Marker models are already downloaded."""
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        cached_repos = [repo.repo_id for repo in cache_info.repos]
        # Check for typical Marker model repos (surya models)
        marker_repos = ["vikp/surya_det3", "vikp/surya_rec2", "vikp/texify"]
        found = sum(1 for repo in marker_repos if repo in cached_repos)
        logger.debug(f"Marker models check: found {found}/3 required models in cache")
        return found >= 2  # At least 2 of 3 core models
    except Exception as e:
        logger.debug(f"Could not check Marker models: {e}")
        return True  # Assume available, let Marker handle errors


def pdf_to_markdown_with_marker(pdf_path: str, output_dir: str, timeout_seconds: int = 300) -> Optional[str]:
    """
    Converts PDF to Markdown using Marker.
    Marker preserves formulas as LaTeX and maintains structure.
    
    Returns path to markdown file or None on failure.
    
    NOTE: Marker has known issues on Windows where it can hang during
    "Recognizing Layout". If this happens, the function will timeout
    and fall back to pdfplumber.
    """
    import concurrent.futures
    import time
    
    # Free VRAM by unloading Ollama models before loading Marker
    logger.info("Unloading Ollama models to free VRAM...")
    unload_ollama_model()
    
    # Give Ollama time to release VRAM
    time.sleep(2)
    
    # Check if models are available
    if not check_marker_models_available():
        logger.warning("Marker models not downloaded yet.")
        logger.info("Falling back to pdfplumber extraction...")
        return None
    
    logger.info(f"Starting Marker conversion (timeout: {timeout_seconds}s)...")
    
    # Run Marker in subprocess to avoid Gradio threading issues
    try:
        import json
        import sys
        worker_script = Path(__file__).parent / "marker_worker.py"
        
        # Use the same Python interpreter as the current process
        python_exe = sys.executable
        
        logger.info(f"Starting Marker worker subprocess with {python_exe}...")
        result = subprocess.run(
            [python_exe, str(worker_script), pdf_path, output_dir],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(Path(__file__).parent)
        )
        
        # Parse result from stdout
        for line in result.stdout.split('\n'):
            if line.startswith('MARKER_STATUS:'):
                logger.info(f"Marker: {line[14:].strip()}")
            elif line.startswith('MARKER_RESULT:'):
                result_json = json.loads(line[14:].strip())
                if result_json.get('success'):
                    logger.info(f"Marker extracted {result_json.get('text_length', 0)} chars, {result_json.get('image_count', 0)} images")
                    return result_json.get('output_path')
                else:
                    logger.error(f"Marker failed: {result_json.get('error')}")
                    return None
        
        # If no result found, check stderr
        if result.returncode != 0:
            logger.error(f"Marker worker failed: {result.stderr}")
            return None
            
        logger.warning("Marker worker returned no result")
        return None
        
    except subprocess.TimeoutExpired:
        logger.error(f"Marker timed out after {timeout_seconds}s")
        return None
    except Exception as e:
        logger.exception(f"Marker conversion failed: {e}")
        return None


def pdf_to_markdown_fallback(pdf_path: str, output_dir: str) -> str:
    """
    Fallback: Uses pdfplumber for extraction.
    """
    import pdfplumber
    
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(layout=True) or ""
            text_parts.append(text)
    
    output_path = Path(output_dir) / "extracted.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n\n---\n\n".join(text_parts))
    
    return str(output_path)


def fix_encoding_errors(content: str) -> str:
    """
    Fix common encoding errors from PDF extraction.
    These occur when PDF text is extracted with wrong encoding or ligatures.
    """
    # Common PDF extraction errors - MUST fix ligatures first
    replacements = [
        # Unicode ligatures (these are the main culprits)
        ('\ufb01', 'fi'),  # ﬁ ligature U+FB01
        ('\ufb02', 'fl'),  # ﬂ ligature U+FB02
        ('\ufb00', 'ff'),  # ﬀ ligature U+FB00
        ('\ufb03', 'ffi'), # ﬃ ligature U+FB03
        ('\ufb04', 'ffl'), # ﬄ ligature U+FB04
        ('ﬁ', 'fi'),
        ('ﬂ', 'fl'),
        ('ﬀ', 'ff'),
        ('ﬃ', 'ffi'),
        ('ﬄ', 'ffl'),
        # Hyphenation artifacts (word broken across lines)
        ('- \n', ''),
        ('-\n', ''),
        # Common OCR/extraction errors
        ('amplication', 'amplification'),
        ('instabili ties', 'instabilities'),
        ('reecting', 'reflecting'),
        ('eective', 'effective'),
        ('dierential', 'differential'),
        ('coecient', 'coefficient'),
        # Unicode replacement character
        ('�', ''),
        # Double spaces
        ('  ', ' '),
        # Broken section numbers
        ('0.0.1 ', ''),  # Remove broken numbering
        ('1 2.', '2.'),  # Fix "1 2. Methodology" -> "2. Methodology"
    ]
    
    result = content
    for old, new in replacements:
        result = result.replace(old, new)
    
    # Fix broken LaTeX commands (common extraction errors)
    # These are literal string replacements, not regex
    latex_fixes = [
        # Broken \exp\bigg patterns
        ('\\expigg', '\\exp\\bigg'),
        ('\\exp igg', '\\exp\\bigg'),
        # Broken \text patterns
        ('\\ ext{', '\\text{'),
        ('\ ext{', '\\text{'),
        (' ext{', '\\text{'),
        # Broken \frac patterns
        ('\\rac{', '\\frac{'),
        (' rac{', '\\frac{'),
        # Broken angle brackets
        ('\nangle', '\\rangle'),
        # Broken \theta
        (' heta', '\\theta'),
        ('( heta', '(\\theta'),
    ]
    
    for old, new in latex_fixes:
        result = result.replace(old, new)
    
    return result


def translate_markdown(
    markdown_path: str,
    output_path: str,
    model: str,
    target_language: str,
    progress_callback=None,
    use_openai: bool = False,
    openai_api_key: str = None
) -> bool:
    """
    Translates markdown while preserving:
    - LaTeX formulas ($$...$$ and $...$)
    - Code blocks
    - Image references
    - Headers and structure
    """
    import requests
    
    with open(markdown_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    logger.debug(f"translate_markdown: input file has {len(content)} chars")
    
    # Pre-process: Strict text normalization (removes ￾, soft hyphen, zero-width, etc.)
    content = normalize_and_reflow(content)
    
    # Additional legacy encoding fixes
    content = fix_encoding_errors(content)
    
    logger.debug(f"translate_markdown: after fix_encoding_errors has {len(content)} chars")
    
    # Protect formulas and code
    protected = {}
    counter = [0]
    
    def protect(match):
        key = f"__PROTECTED_{counter[0]}__"
        protected[key] = match.group(0)
        counter[0] += 1
        return key
    
    # Protect patterns - order matters! More specific patterns first
    patterns = [
        # LaTeX environments (all common math environments)
        (r'\\begin\{equation\*?\}[\s\S]*?\\end\{equation\*?\}', 'latex_equation'),
        (r'\\begin\{align\*?\}[\s\S]*?\\end\{align\*?\}', 'latex_align'),
        (r'\\begin\{eqnarray\*?\}[\s\S]*?\\end\{eqnarray\*?\}', 'latex_eqnarray'),
        (r'\\begin\{gather\*?\}[\s\S]*?\\end\{gather\*?\}', 'latex_gather'),
        (r'\\begin\{multline\*?\}[\s\S]*?\\end\{multline\*?\}', 'latex_multline'),
        (r'\\begin\{split\}[\s\S]*?\\end\{split\}', 'latex_split'),
        (r'\\begin\{cases\}[\s\S]*?\\end\{cases\}', 'latex_cases'),
        (r'\\begin\{matrix\}[\s\S]*?\\end\{matrix\}', 'latex_matrix'),
        (r'\\begin\{pmatrix\}[\s\S]*?\\end\{pmatrix\}', 'latex_pmatrix'),
        (r'\\begin\{bmatrix\}[\s\S]*?\\end\{bmatrix\}', 'latex_bmatrix'),
        (r'\\begin\{array\}[\s\S]*?\\end\{array\}', 'latex_array'),
        # Display math ($$...$$, \[...\])
        (r'\$\$[\s\S]*?\$\$', 'display_math'),
        (r'\\\[[\s\S]*?\\\]', 'display_math_bracket'),
        # Inline math ($...$, \(...\)) - improved to handle nested braces
        (r'\$(?:[^$\\]|\\.)+\$', 'inline_math'),
        (r'\\\((?:[^\\]|\\.)*?\\\)', 'inline_math_paren'),
        # Greek letters (Unicode and LaTeX)
        (r'[αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]+', 'greek_unicode'),
        # Standalone LaTeX commands with arguments
        (r'\\(?:frac|sqrt|vec|hat|bar|tilde|dot|ddot|overline|underline)\{[^}]*\}\{[^}]*\}', 'latex_frac'),
        (r'\\(?:frac|sqrt|vec|hat|bar|tilde|dot|ddot|overline|underline)\{[^}]*\}', 'latex_command_arg'),
        (r'\\(?:alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega|infty|partial|nabla|hbar|ell|Re|Im|wp|aleph|forall|exists|neg|emptyset|in|notin|subset|supset|cup|cap|vee|wedge|oplus|otimes|perp|parallel|sim|simeq|approx|cong|equiv|propto|neq|leq|geq|ll|gg|pm|mp|times|div|cdot|ast|star|circ|bullet|diamond|dagger|ddagger|sum|prod|int|oint|bigcup|bigcap|lim|sup|inf|max|min|arg|det|dim|exp|ln|log|sin|cos|tan|cot|sec|csc|arcsin|arccos|arctan|sinh|cosh|tanh)\b', 'latex_symbol'),
        (r'\\[a-zA-Z]+\{[^}]*\}', 'latex_command_arg'),
        (r'\\[a-zA-Z]+', 'latex_command'),
        # Subscripts and superscripts
        (r'_\{[^}]+\}', 'subscript'),
        (r'\^\{[^}]+\}', 'superscript'),
        (r'_[a-zA-Z0-9]', 'subscript_simple'),
        (r'\^[a-zA-Z0-9]', 'superscript_simple'),
        # Code blocks
        (r'```[\s\S]*?```', 'code_block'),
        (r'`[^`]+`', 'inline_code'),
        # Images and links
        (r'!\[.*?\]\(.*?\)', 'image'),
        (r'\[.*?\]\(.*?\)', 'link'),
        # Scientific notation and numbers with units
        (r'\d+\.?\d*\s*×\s*10\^[−\-]?\{?[−\-]?\d+\}?', 'scientific_notation'),
        (r'\d+\.?\d*\s*[eE][−\-+]?\d+', 'exponential'),
        (r'\d+\.?\d*\s*(?:Hz|kHz|MHz|GHz|THz|nm|μm|mm|cm|m|km|ns|μs|ms|s|eV|keV|MeV|GeV|TeV|K|°C|°F|Pa|kPa|MPa|GPa|J|kJ|MJ|W|kW|MW|GW|V|mV|kV|A|mA|μA|Ω|kΩ|MΩ|F|pF|nF|μF|H|mH|μH|T|mT|μT|G|mol|M|mM|μM|nM|L|mL|μL|g|mg|μg|kg)\b', 'number_with_unit'),
        # Variable names (single letters with optional subscript)
        (r'\b[A-Za-z]_[A-Za-z0-9]+\b', 'variable_subscript'),
        # Common physics symbols
        (r'[ℏℓ∇∂∫∑∏√∞±∓≈≠≤≥≪≫∝∈∉⊂⊃∪∩∧∨⊕⊗⊥∥]', 'math_symbol'),
    ]
    
    protected_content = content
    for pattern, name in patterns:
        protected_content = re.sub(pattern, protect, protected_content)
    
    logger.debug(f"translate_markdown: after protection has {len(protected_content)} chars, {len(protected)} protected items")
    
    # Split into chunks for translation
    chunks = re.split(r'(\n#{1,6}\s+[^\n]+\n)', protected_content)
    
    logger.debug(f"translate_markdown: split into {len(chunks)} chunks, total chars: {sum(len(c) for c in chunks)}")
    
    total_chunks = len([c for c in chunks if c.strip() and not c.startswith('#')])
    translated_chunks = []
    chunk_count = 0
    
    for chunk in chunks:
        if not chunk.strip():
            translated_chunks.append(chunk)
            continue
        
        # Keep headers but translate their text
        if chunk.strip().startswith('#'):
            # Extract header level and text
            match = re.match(r'(\n?#{1,6}\s+)(.+?)(\n)', chunk)
            if match:
                prefix, text, suffix = match.groups()
                translated_text = translate_text_chunk(text, model, target_language, use_openai, openai_api_key)
                translated_chunks.append(f"{prefix}{translated_text}{suffix}")
            else:
                translated_chunks.append(chunk)
            continue
        
        chunk_count += 1
        if progress_callback:
            progress_callback(chunk_count, total_chunks, f"Translating chunk {chunk_count}/{total_chunks}")
        
        # Split very large chunks into smaller pieces (max 2000 chars per piece)
        MAX_CHUNK_SIZE = 2000
        if len(chunk) > MAX_CHUNK_SIZE:
            # Split by paragraphs first
            paragraphs = re.split(r'(\n\n+)', chunk)
            current_piece = ""
            pieces = []
            
            for para in paragraphs:
                if len(current_piece) + len(para) > MAX_CHUNK_SIZE and current_piece:
                    pieces.append(current_piece)
                    current_piece = para
                else:
                    current_piece += para
            if current_piece:
                pieces.append(current_piece)
            
            # Translate each piece
            translated_pieces = []
            for i, piece in enumerate(pieces):
                if progress_callback:
                    progress_callback(chunk_count, total_chunks, f"Chunk {chunk_count}/{total_chunks} - part {i+1}/{len(pieces)}")
                translated_piece = translate_text_chunk(piece, model, target_language, use_openai, openai_api_key)
                translated_pieces.append(translated_piece)
            translated_chunks.append(''.join(translated_pieces))
        else:
            translated = translate_text_chunk(chunk, model, target_language, use_openai, openai_api_key)
            translated_chunks.append(translated)
    
    # Reconstruct
    result = ''.join(translated_chunks)
    
    # CRITICAL: Check for garbage BEFORE restoring protected content
    # Detect repetitive garbage patterns like "flfflifl"
    words = result.split()
    if len(words) > 20:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.15:  # Less than 15% unique words = garbage
            logger.error(f"Translation produced garbage (unique ratio: {unique_ratio:.2f}). Using original content.")
            # Return original content instead
            result = content  # Use the original pre-processed content
    
    # Check for ligature garbage patterns
    if 'flfl' in result or 'flffli' in result or 'fifi' in result:
        logger.error("Translation produced ligature garbage. Using original content.")
        result = content
    
    # Restore protected content
    for key, value in protected.items():
        result = result.replace(key, value)
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result)
    
    return True


def translate_text_chunk(text: str, model: str, target_language: str, use_openai: bool = False, openai_api_key: str = None) -> str:
    """Translates a single text chunk using Ollama or OpenAI with scientific accuracy."""
    if not text.strip():
        return text
    
    # PASSTHROUGH MODE: If target is "PASSTHROUGH" or "Original", return text unchanged
    # This is for testing the pipeline without translation
    if target_language.upper() in ["PASSTHROUGH", "ORIGINAL", "NONE", "ENGLISH"]:
        logger.debug(f"Passthrough mode: returning original text")
        return text
    
    # Skip if mostly placeholders
    placeholder_count = text.count('__PROTECTED_')
    if placeholder_count > len(text.split()) / 2:
        return text
    
    # Scientific translation prompt - preserves technical accuracy
    system_prompt = f"""You are a scientific translator. Translate the following text to {target_language}.

RULES:
1. Output ONLY the translated text - NO explanations, NO comments, NO questions
2. Keep __PROTECTED_X__ placeholders unchanged (these are formulas)
3. Keep markdown formatting (#, **, *, -)
4. Keep technical terms and variable names
5. This is a physics paper - the text IS readable, just translate it
6. NEVER say "I cannot translate" or "the text is encrypted" - just translate it

If you see LaTeX-like symbols ($, \\, etc.), they are math formulas - keep them as-is."""

    if use_openai and openai_api_key:
        return translate_text_chunk_openai(text, openai_api_key, target_language, system_prompt)
    else:
        return translate_text_chunk_ollama(text, model, target_language, system_prompt)


def translate_text_chunk_openai(text: str, api_key: str, target_language: str, system_prompt: str) -> str:
    """Translates text using OpenAI with gpt-4o for scientific accuracy."""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        # Use gpt-4o for better scientific translation quality
        # Fall back to gpt-4o-mini if gpt-4o fails (e.g., rate limits)
        models_to_try = ["gpt-4o", "gpt-4o-mini"]
        
        for model in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Translate to {target_language}:\n{text}"}
                    ],
                    temperature=0.1,
                    max_tokens=4096
                )
                result = response.choices[0].message.content.strip()
                if result:
                    return result
            except Exception as model_error:
                logger.debug(f"Model {model} failed: {model_error}")
                continue
        
    except Exception as e:
        logger.warning(f"OpenAI translation error: {e}")
    return text


def translate_text_chunk_ollama(text: str, model: str, target_language: str, system_prompt: str) -> str:
    """Translates text using Ollama with validation."""
    import requests
    
    logger.debug(f"Ollama translation: model={model}, text_len={len(text)}")
    
    try:
        # First check if Ollama is running
        try:
            health = requests.get("http://localhost:11434/api/tags", timeout=5)
            if health.status_code != 200:
                logger.error("Ollama service not responding")
                return text
        except Exception as e:
            logger.error(f"Ollama not running: {e}")
            return text
        
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Translate to {target_language}:\n{text}"}
                ],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 4096}
            },
            timeout=300  # 5 minutes timeout for large scientific chunks
        )
        
        if response.status_code == 200:
            result = response.json().get("message", {}).get("content", text)
            result = result.strip()
            
            # Validate the translation - detect garbage responses
            garbage_indicators = [
                "lo siento", "i'm sorry", "i cannot", "no puedo",
                "encrypted", "encriptado", "codificado", "encoded",
                "not clear", "no está claro", "could you please",
                "podrías proporcionar", "provide more context"
            ]
            
            result_lower = result.lower()
            for indicator in garbage_indicators:
                if indicator in result_lower:
                    logger.warning(f"Ollama returned garbage response (detected: '{indicator}'). Keeping original.")
                    return text
            
            # Detect repetitive garbage patterns like "flfflifl flfflifl"
            # Check if result has too many repeated short patterns
            words = result.split()
            if len(words) > 10:
                # Count unique words vs total words
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.1:  # Less than 10% unique words = garbage
                    logger.warning(f"Ollama returned repetitive garbage (unique ratio: {unique_ratio:.2f}). Keeping original.")
                    return text
                
                # Check for ligature-like garbage patterns
                if any(pattern in result_lower for pattern in ['flfl', 'fifi', 'ffff', 'flffli']):
                    logger.warning(f"Ollama returned ligature garbage. Keeping original.")
                    return text
            
            # Check if result is too short compared to input (likely failed)
            if len(result) < len(text) * 0.3 and len(text) > 100:
                logger.warning(f"Ollama response too short ({len(result)} vs {len(text)}). Keeping original.")
                return text
            
            logger.debug(f"Ollama translation successful: {len(result)} chars")
            return result
        else:
            # Handle error responses
            error_msg = response.json().get("error", "Unknown error")
            logger.error(f"Ollama API error ({response.status_code}): {error_msg}")
            return text
    except requests.exceptions.Timeout:
        logger.error(f"Ollama timeout after 300s - model too slow for this chunk. Keeping original.")
        return text
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Ollama connection error: {e}")
        return text
    except Exception as e:
        logger.error(f"Ollama translation error: {e}")
    
    return text


def markdown_to_pdf(markdown_path: str, output_pdf: str) -> bool:
    """
    Converts Markdown to PDF using pandoc with pdflatex.
    """
    # First, fix any encoding errors in the markdown file
    try:
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply encoding fixes
        fixed_content = fix_encoding_errors(content)
        
        # Write back if changed
        if fixed_content != content:
            logger.info(f"Fixed encoding errors in markdown before PDF conversion")
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
    except Exception as e:
        logger.warning(f"Could not fix encoding in markdown: {e}")
    
    # Find pdflatex
    pdflatex_paths = [
        r"C:\Users\linoc\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe",
        r"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe",
        "pdflatex"
    ]
    pdflatex_cmd = None
    for p in pdflatex_paths:
        if Path(p).exists() or p == "pdflatex":
            pdflatex_cmd = p
            break
    
    try:
        # Try pandoc with pdflatex
        result = subprocess.run(
            [
                "pandoc",
                markdown_path,
                "-o", output_pdf,
                f"--pdf-engine={pdflatex_cmd}",
                "-V", "geometry:margin=2cm",
                "--standalone"
            ],
            capture_output=True,
            timeout=120
        )
        
        if Path(output_pdf).exists():
            return True
        else:
            logger.warning(f"pandoc stderr: {result.stderr.decode('utf-8', errors='ignore')}")
            
    except FileNotFoundError:
        logger.warning("pandoc not found")
    except Exception as e:
        logger.warning(f"pandoc failed: {e}")
    
    # Fallback: Create simple LaTeX and compile
    try:
        return markdown_to_latex_to_pdf(markdown_path, output_pdf)
    except Exception as e:
        logger.warning(f"LaTeX fallback failed: {e}")
    
    return False


def markdown_to_latex_to_pdf(markdown_path: str, output_pdf: str) -> bool:
    """
    Converts Markdown to LaTeX, then compiles to PDF.
    """
    # Find pdflatex
    pdflatex_paths = [
        r"C:\Users\linoc\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe",
        r"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe",
        "pdflatex"
    ]
    pdflatex_cmd = "pdflatex"
    for p in pdflatex_paths:
        if Path(p).exists():
            pdflatex_cmd = p
            break
    
    with open(markdown_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to LaTeX
    latex_content = markdown_to_latex(md_content)
    
    # Write LaTeX
    tex_path = markdown_path.replace('.md', '.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    # Compile
    output_dir = Path(markdown_path).parent
    tex_filename = Path(tex_path).name
    
    logger.info(f"Compiling LaTeX with: {pdflatex_cmd}")
    result = subprocess.run(
        [pdflatex_cmd, "-interaction=nonstopmode", tex_filename],
        cwd=str(output_dir),
        capture_output=True,
        timeout=120
    )
    
    # Check for output
    generated_pdf = str(output_dir / tex_filename.replace('.tex', '.pdf'))
    if Path(generated_pdf).exists():
        if generated_pdf != output_pdf:
            import shutil
            shutil.copy(generated_pdf, output_pdf)
        return True
    
    logger.warning(f"pdflatex failed: {result.stderr.decode('utf-8', errors='ignore')[:500]}")
    return False


def normalize_markdown(md_content: str) -> str:
    """
    Pre-normalize Markdown before LaTeX conversion.
    Fixes common extraction issues:
    1. Paragraph reflow (remove PDF line wraps)
    2. Clean up inline markdown headers
    3. Math token recomposition
    """
    content = md_content
    
    # === FIX 4: Paragraph Reflow ===
    # Remove single line breaks within paragraphs (keep double line breaks)
    # But preserve line breaks before/after headers, lists, math blocks
    lines = content.split('\n')
    reflowed = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Keep empty lines as paragraph separators
        if not stripped:
            reflowed.append('')
            i += 1
            continue
        
        # Keep headers, lists, math blocks on their own lines
        if (stripped.startswith('#') or 
            stripped.startswith('- ') or 
            stripped.startswith('* ') or
            stripped.startswith('$$') or
            stripped.startswith('\\[') or
            stripped.startswith('\\begin{')):
            reflowed.append(line)
            i += 1
            continue
        
        # Collect paragraph lines and join them
        para_lines = [stripped]
        i += 1
        while i < len(lines):
            next_line = lines[i].strip()
            # Stop at empty line, header, list, or math
            if (not next_line or 
                next_line.startswith('#') or 
                next_line.startswith('- ') or
                next_line.startswith('* ') or
                next_line.startswith('$$') or
                next_line.startswith('\\[') or
                next_line.startswith('\\begin{')):
                break
            para_lines.append(next_line)
            i += 1
        
        # Join paragraph with spaces
        reflowed.append(' '.join(para_lines))
    
    content = '\n'.join(reflowed)
    
    # === FIX 2: Clean inline markdown headers ===
    # Remove ### or ## that appear mid-line (not at start)
    content = re.sub(r'(\S)\s*###\s+', r'\1\n\n### ', content)
    content = re.sub(r'(\S)\s*##\s+', r'\1\n\n## ', content)
    
    # === FIX 3: Math token recomposition ===
    # Join fragmented math expressions (iℏ on one line, ∂ on next, etc.)
    # Look for lines that are just math symbols and join them
    math_symbols = r'^[\s\d\+\-\=\*\/\(\)\[\]\{\}\^\\_\$\\α-ωΑ-Ωℏ∂∇∫∑∏√∞±×÷≤≥≠≈→←↔⇒⇐⇔]+$'
    lines = content.split('\n')
    joined = []
    buffer = []
    for line in lines:
        stripped = line.strip()
        # If line looks like fragmented math, buffer it
        if stripped and len(stripped) < 20 and re.match(math_symbols, stripped):
            buffer.append(stripped)
        else:
            if buffer:
                # Join buffered math fragments
                joined.append(' '.join(buffer))
                buffer = []
            joined.append(line)
    if buffer:
        joined.append(' '.join(buffer))
    content = '\n'.join(joined)
    
    return content


def markdown_to_latex(md_content: str) -> str:
    """
    Converts Markdown to LaTeX.
    """
    # Pre-normalize the markdown
    md_content = normalize_markdown(md_content)
    
    # Preamble
    latex = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{geometry}
\geometry{margin=2cm}
\usepackage{hyperref}
\usepackage{graphicx}

\begin{document}

"""
    
    content = md_content
    
    # Replace Greek letters with LaTeX commands
    greek_map = {
        'α': r'$\alpha$', 'β': r'$\beta$', 'γ': r'$\gamma$', 'δ': r'$\delta$',
        'ε': r'$\epsilon$', 'ζ': r'$\zeta$', 'η': r'$\eta$', 'θ': r'$\theta$',
        'ι': r'$\iota$', 'κ': r'$\kappa$', 'λ': r'$\lambda$', 'μ': r'$\mu$',
        'ν': r'$\nu$', 'ξ': r'$\xi$', 'π': r'$\pi$', 'ρ': r'$\rho$',
        'σ': r'$\sigma$', 'τ': r'$\tau$', 'υ': r'$\upsilon$', 'φ': r'$\phi$',
        'χ': r'$\chi$', 'ψ': r'$\psi$', 'ω': r'$\omega$',
        'Α': r'$A$', 'Β': r'$B$', 'Γ': r'$\Gamma$', 'Δ': r'$\Delta$',
        'Ε': r'$E$', 'Ζ': r'$Z$', 'Η': r'$H$', 'Θ': r'$\Theta$',
        'Ι': r'$I$', 'Κ': r'$K$', 'Λ': r'$\Lambda$', 'Μ': r'$M$',
        'Ν': r'$N$', 'Ξ': r'$\Xi$', 'Π': r'$\Pi$', 'Ρ': r'$P$',
        'Σ': r'$\Sigma$', 'Τ': r'$T$', 'Υ': r'$\Upsilon$', 'Φ': r'$\Phi$',
        'Χ': r'$X$', 'Ψ': r'$\Psi$', 'Ω': r'$\Omega$',
        '∞': r'$\infty$', '∂': r'$\partial$', '∇': r'$\nabla$',
        '±': r'$\pm$', '×': r'$\times$', '÷': r'$\div$',
        '≤': r'$\leq$', '≥': r'$\geq$', '≠': r'$\neq$', '≈': r'$\approx$',
        '→': r'$\rightarrow$', '←': r'$\leftarrow$', '↔': r'$\leftrightarrow$',
        '⇒': r'$\Rightarrow$', '⇐': r'$\Leftarrow$', '⇔': r'$\Leftrightarrow$',
    }
    for char, latex_cmd in greek_map.items():
        content = content.replace(char, latex_cmd)
    
    # === FIX 1: Abstract handling ===
    # Convert "Abstract" or "### Abstract" to proper LaTeX abstract environment
    # This keeps abstract in document flow without page break
    abstract_pattern = r'^(?:###?\s*)?(?:Abstract|ABSTRACT)\s*\n(.*?)(?=\n(?:###?\s*)?(?:Introduction|INTRODUCTION|1\.|I\.)|\Z)'
    abstract_match = re.search(abstract_pattern, content, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
    if abstract_match:
        abstract_text = abstract_match.group(1).strip()
        # Replace the abstract section with proper LaTeX abstract
        content = re.sub(abstract_pattern, '', content, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
        # We'll insert the abstract after \begin{document} later
        abstract_latex = f'\\begin{{abstract}}\n{abstract_text}\n\\end{{abstract}}\n\n'
    else:
        abstract_latex = ''
    
    # Headers
    content = re.sub(r'^######\s+(.+)$', r'\\subparagraph{\1}', content, flags=re.MULTILINE)
    content = re.sub(r'^#####\s+(.+)$', r'\\paragraph{\1}', content, flags=re.MULTILINE)
    content = re.sub(r'^####\s+(.+)$', r'\\subsubsection{\1}', content, flags=re.MULTILINE)
    content = re.sub(r'^###\s+(.+)$', r'\\subsection{\1}', content, flags=re.MULTILINE)
    content = re.sub(r'^##\s+(.+)$', r'\\section{\1}', content, flags=re.MULTILINE)
    content = re.sub(r'^#\s+(.+)$', r'\\section*{\1}', content, flags=re.MULTILINE)
    
    # Bold and italic
    content = re.sub(r'\*\*\*(.+?)\*\*\*', r'\\textbf{\\textit{\1}}', content)
    content = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', content)
    content = re.sub(r'\*(.+?)\*', r'\\textit{\1}', content)
    
    # Lists
    lines = content.split('\n')
    in_list = False
    new_lines = []
    for line in lines:
        if line.strip().startswith('- '):
            if not in_list:
                new_lines.append('\\begin{itemize}')
                in_list = True
            new_lines.append('\\item ' + line.strip()[2:])
        else:
            if in_list:
                new_lines.append('\\end{itemize}')
                in_list = False
            new_lines.append(line)
    if in_list:
        new_lines.append('\\end{itemize}')
    content = '\n'.join(new_lines)
    
    # Escape special characters (but not in math mode)
    # This is simplified - a full implementation would be more careful
    content = content.replace('%', '\\%')
    content = content.replace('&', '\\&')
    content = content.replace('#', '\\#')
    
    # Insert abstract right after \begin{document}
    latex += abstract_latex
    latex += content
    latex += "\n\n\\end{document}\n"
    
    return latex


def translate_pdf_with_marker(
    input_pdf: str,
    output_dir: str,
    model: str,
    target_language: str,
    progress_callback=None,
    use_openai: bool = False,
    openai_api_key: str = None
) -> Tuple[Optional[str], str]:
    """
    Main function: Translates PDF using Marker pipeline.
    
    1. PDF → Markdown (with Marker, preserving formulas)
    2. Markdown → Translated Markdown
    3. Markdown → PDF
    
    Returns (output_path, status_message)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if Marker models are available first
    if not check_marker_models_available():
        return None, "❌ Marker models not downloaded!\n\nRun install.bat and select 'y' for Marker models,\nor use 'Standard' mode instead."
    
    # Step 1: PDF to Markdown
    if progress_callback:
        progress_callback(5, 100, "Extracting PDF with Marker...")
    
    md_path = pdf_to_markdown_with_marker(input_pdf, str(output_dir))
    if not md_path:
        if progress_callback:
            progress_callback(10, 100, "Marker failed, using fallback...")
        md_path = pdf_to_markdown_fallback(input_pdf, str(output_dir))
    
    # Step 2: Translate Markdown
    if progress_callback:
        progress_callback(20, 100, "Translating...")
    
    translated_md = str(output_dir / "translated.md")
    success = translate_markdown(
        md_path, translated_md, model, target_language,
        progress_callback=lambda c, t, s: progress_callback(20 + int(60 * c / max(t, 1)), 100, s) if progress_callback else None,
        use_openai=use_openai,
        openai_api_key=openai_api_key
    )
    
    if not success:
        return None, "Translation failed"
    
    # Step 3: Markdown to PDF
    if progress_callback:
        progress_callback(85, 100, "Generating PDF...")
    
    output_pdf = str(output_dir / "translated.pdf")
    pdf_success = markdown_to_pdf(translated_md, output_pdf)
    
    if pdf_success and Path(output_pdf).exists():
        if progress_callback:
            progress_callback(100, 100, "Complete!")
        # Also copy to a stable location for easy access
        import shutil
        stable_output = Path(__file__).parent / "output" / "translated.pdf"
        stable_output.parent.mkdir(exist_ok=True)
        shutil.copy2(output_pdf, stable_output)
        logger.info(f"PDF also saved to: {stable_output}")
        # Return the original path (like .tex files do) - Gradio handles this better
        return output_pdf, f"✅ Translation complete with Marker pipeline!\n\n📁 Also saved to: {stable_output}"
    
    # Return markdown if PDF failed
    if progress_callback:
        progress_callback(100, 100, "Complete (PDF generation failed)")
    # Copy markdown to stable location
    import shutil
    stable_md = Path(__file__).parent / "output" / "translated.md"
    stable_md.parent.mkdir(exist_ok=True)
    shutil.copy2(translated_md, stable_md)
    return str(stable_md), "✅ Translation complete! (PDF generation failed - returning Markdown)"


# ===================================================================
# LaTeX Translation Mode - TEX → translated TEX → PDF
# ===================================================================

def translate_latex(
    latex_path: str,
    output_path: str,
    model: str = "qwen2.5:7b",
    target_language: str = "German",
    progress_callback=None,
    use_openai: bool = False,
    openai_api_key: str = None
) -> bool:
    """
    Translate a LaTeX file while preserving all LaTeX commands and formulas.
    
    This is the BEST approach for scientific papers:
    - Formulas stay intact (no extraction errors)
    - Structure preserved (sections, labels, refs)
    - Consistent output via pdflatex
    """
    import requests
    
    with open(latex_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    logger.debug(f"translate_latex: input file has {len(content)} chars")
    
    # Protect LaTeX constructs that should NOT be translated
    protected = {}
    counter = [0]
    
    def protect(match):
        key = f"__LATEX_PROTECTED_{counter[0]}__"
        protected[key] = match.group(0)
        counter[0] += 1
        return key
    
    # Patterns to protect (order matters - more specific first)
    patterns = [
        # Document structure
        (r'\\documentclass\[.*?\]\{.*?\}', 'documentclass'),
        (r'\\documentclass\{.*?\}', 'documentclass'),
        (r'\\usepackage\[.*?\]\{.*?\}', 'usepackage'),
        (r'\\usepackage\{.*?\}', 'usepackage'),
        (r'\\begin\{document\}', 'begin_doc'),
        (r'\\end\{document\}', 'end_doc'),
        # Preamble commands
        (r'\\newcommand\{.*?\}.*?(?=\\newcommand|\\begin\{document\}|\n\n)', 'newcommand'),
        (r'\\renewcommand\{.*?\}.*', 'renewcommand'),
        (r'\\geometry\{.*?\}', 'geometry'),
        (r'\\bibliographystyle\{.*?\}', 'bibstyle'),
        # Math environments (display)
        (r'\\begin\{equation\*?\}[\s\S]*?\\end\{equation\*?\}', 'equation'),
        (r'\\begin\{align\*?\}[\s\S]*?\\end\{align\*?\}', 'align'),
        (r'\\begin\{gather\*?\}[\s\S]*?\\end\{gather\*?\}', 'gather'),
        (r'\\begin\{multline\*?\}[\s\S]*?\\end\{multline\*?\}', 'multline'),
        (r'\\\[[\s\S]*?\\\]', 'display_math'),
        (r'\$\$[\s\S]*?\$\$', 'display_math2'),
        # Inline math
        (r'\$[^\$\n]+\$', 'inline_math'),
        (r'\\([^a-zA-Z])', 'escaped_char'),  # \%, \$, etc.
        # Tables and figures
        (r'\\begin\{table\}[\s\S]*?\\end\{table\}', 'table'),
        (r'\\begin\{tabular\}[\s\S]*?\\end\{tabular\}', 'tabular'),
        (r'\\begin\{figure\}[\s\S]*?\\end\{figure\}', 'figure'),
        # Lists (protect structure, not content)
        (r'\\begin\{itemize\}', 'begin_itemize'),
        (r'\\end\{itemize\}', 'end_itemize'),
        (r'\\begin\{enumerate\}', 'begin_enumerate'),
        (r'\\end\{enumerate\}', 'end_enumerate'),
        (r'\\item\s*', 'item'),
        # References and citations
        (r'\\label\{[^}]*\}', 'label'),
        (r'\\ref\{[^}]*\}', 'ref'),
        (r'\\cref\{[^}]*\}', 'cref'),
        (r'\\eqref\{[^}]*\}', 'eqref'),
        (r'\\cite[pt]?\{[^}]*\}', 'cite'),
        (r'\\citep\{[^}]*\}', 'citep'),
        (r'\\citet\{[^}]*\}', 'citet'),
        (r'\\bibliography\{[^}]*\}', 'bibliography'),
        # Section commands (protect command, translate title separately)
        (r'\\section\*?\{', 'section_start'),
        (r'\\subsection\*?\{', 'subsection_start'),
        (r'\\subsubsection\*?\{', 'subsubsection_start'),
        (r'\\paragraph\{', 'paragraph_start'),
        # Text formatting commands - protect the ENTIRE command including content
        (r'\\textbf\{[^}]*\}', 'textbf'),
        (r'\\textit\{[^}]*\}', 'textit'),
        (r'\\emph\{[^}]*\}', 'emph'),
        (r'\\textrm\{[^}]*\}', 'textrm'),
        (r'\\text\{[^}]*\}', 'text'),
        (r'\\maketitle', 'maketitle'),
        (r'\\date\{[^}]*\}', 'date'),
        (r'\\author\{[^}]*\}', 'author'),
        (r'\\title\{[^}]*\}', 'title'),
        (r'\\tableofcontents', 'toc'),
        (r'\\newpage', 'newpage'),
        (r'\\clearpage', 'clearpage'),
        # Bibliography entries
        (r'\\bibitem\[.*?\]\{.*?\}', 'bibitem'),
        (r'\\bibitem\{.*?\}', 'bibitem'),
        # Comments
        (r'%.*$', 'comment'),
    ]
    
    protected_content = content
    for pattern, name in patterns:
        protected_content = re.sub(pattern, protect, protected_content, flags=re.MULTILINE)
    
    logger.debug(f"translate_latex: protected {len(protected)} items")
    
    # Split into translatable chunks (by paragraphs)
    paragraphs = re.split(r'(\n\n+)', protected_content)
    
    translated_paragraphs = []
    total = len([p for p in paragraphs if p.strip() and not p.startswith('__LATEX_PROTECTED_')])
    count = 0
    
    for para in paragraphs:
        # Skip empty or whitespace-only
        if not para.strip():
            translated_paragraphs.append(para)
            continue
        
        # Skip if mostly protected content
        if para.count('__LATEX_PROTECTED_') > len(para.split()) / 2:
            translated_paragraphs.append(para)
            continue
        
        # Skip preamble (before \begin{document})
        if '\\documentclass' in para or '\\usepackage' in para:
            translated_paragraphs.append(para)
            continue
        
        count += 1
        if progress_callback:
            progress_callback(count, total, f"Translating paragraph {count}/{total}")
        
        # Translate this paragraph
        translated = translate_text_chunk(para, model, target_language, use_openai, openai_api_key)
        translated_paragraphs.append(translated)
    
    # Reconstruct
    result = ''.join(translated_paragraphs)
    
    # Restore protected content
    for key, value in protected.items():
        result = result.replace(key, value)
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result)
    
    logger.info(f"translate_latex: wrote {len(result)} chars to {output_path}")
    return True


def translate_latex_to_pdf(
    latex_path: str,
    output_dir: str,
    model: str = "qwen2.5:7b",
    target_language: str = "German",
    progress_callback=None,
    use_openai: bool = False,
    openai_api_key: str = None
) -> Tuple[Optional[str], str]:
    """
    Complete LaTeX translation pipeline:
    1. Translate LaTeX file (preserving formulas)
    2. Compile to PDF with pdflatex
    
    Returns: (output_path, status_message)
    """
    from pathlib import Path
    import shutil
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_name = Path(latex_path).stem
    translated_tex = output_dir / f"{input_name}_translated.tex"
    output_pdf = output_dir / f"{input_name}_translated.pdf"
    
    # Step 1: Translate LaTeX
    if progress_callback:
        progress_callback(10, 100, "Translating LaTeX...")
    
    success = translate_latex(
        latex_path,
        str(translated_tex),
        model,
        target_language,
        progress_callback=lambda c, t, s: progress_callback(10 + int(c/t*70), 100, s) if progress_callback else None,
        use_openai=use_openai,
        openai_api_key=openai_api_key
    )
    
    if not success:
        return None, "LaTeX translation failed"
    
    # Step 2: Compile with pdflatex
    if progress_callback:
        progress_callback(85, 100, "Compiling PDF with pdflatex...")
    
    # Find pdflatex
    pdflatex_paths = [
        r"C:\Users\linoc\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe",
        "pdflatex",
    ]
    
    pdflatex = None
    for p in pdflatex_paths:
        if Path(p).exists() or shutil.which(p):
            pdflatex = p
            break
    
    if not pdflatex:
        return str(translated_tex), "✅ LaTeX translated! (pdflatex not found - returning .tex file)"
    
    # Run pdflatex twice for references
    for run in range(2):
        try:
            result = subprocess.run(
                [pdflatex, "-interaction=nonstopmode", str(translated_tex)],
                cwd=str(output_dir),
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode != 0:
                logger.warning(f"pdflatex run {run+1} warnings: {result.stderr[:500]}")
        except Exception as e:
            logger.error(f"pdflatex failed: {e}")
            return str(translated_tex), f"✅ LaTeX translated! (pdflatex error: {e})"
    
    if output_pdf.exists():
        if progress_callback:
            progress_callback(100, 100, "Complete!")
        
        # Copy to stable location
        stable_output = Path(__file__).parent / "output" / f"{input_name}_translated.pdf"
        stable_output.parent.mkdir(exist_ok=True)
        shutil.copy2(output_pdf, stable_output)
        logger.info(f"PDF saved to: {stable_output}")
        
        return str(output_pdf), f"✅ LaTeX translation complete!\n\n📁 PDF: {stable_output}\n📄 TEX: {translated_tex}"
    
    return str(translated_tex), "✅ LaTeX translated! (PDF compilation failed - returning .tex file)"


# For testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python pdf_marker_translator.py input.pdf output_dir [model] [target_lang]")
        print("       python pdf_marker_translator.py input.tex output_dir [model] [target_lang]  # LaTeX mode")
        sys.exit(1)
    
    input_pdf = sys.argv[1]
    output_dir = sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "qwen2.5:7b"
    target = sys.argv[4] if len(sys.argv) > 4 else "German"
    
    print(f"Translating {input_pdf} to {target} using {model}...")
    
    output, status = translate_pdf_with_marker(
        input_pdf, output_dir, model, target,
        progress_callback=lambda c, t, s: print(f"  [{c}%] {s}")
    )
    
    print(status)
    if output:
        print(f"Output: {output}")
