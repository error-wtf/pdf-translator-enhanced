"""
LaTeX File Translator

Translates .tex files directly while preserving all LaTeX commands and formulas.
This provides perfect 1:1 output for scientific papers.

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger("pdf_translator.latex_translator")


def extract_translatable_segments(latex_content: str) -> List[Tuple[int, int, str, bool]]:
    """
    Extracts segments from LaTeX that should/shouldn't be translated.
    
    Returns list of (start, end, content, should_translate)
    """
    segments = []
    
    # Patterns for content that should NOT be translated
    no_translate_patterns = [
        # Math environments
        (r'\$\$.*?\$\$', 'display_math'),
        (r'\$[^$]+\$', 'inline_math'),
        (r'\\begin\{equation\*?\}.*?\\end\{equation\*?\}', 'equation'),
        (r'\\begin\{align\*?\}.*?\\end\{align\*?\}', 'align'),
        (r'\\begin\{eqnarray\*?\}.*?\\end\{eqnarray\*?\}', 'eqnarray'),
        (r'\\begin\{gather\*?\}.*?\\end\{gather\*?\}', 'gather'),
        (r'\\begin\{multline\*?\}.*?\\end\{multline\*?\}', 'multline'),
        (r'\\\[.*?\\\]', 'display_math_bracket'),
        (r'\\\(.*?\\\)', 'inline_math_paren'),
        # LaTeX commands with arguments that shouldn't be translated
        (r'\\documentclass\{[^}]*\}', 'documentclass'),
        (r'\\usepackage(\[[^\]]*\])?\{[^}]*\}', 'usepackage'),
        (r'\\bibliography\{[^}]*\}', 'bibliography'),
        (r'\\bibliographystyle\{[^}]*\}', 'bibliographystyle'),
        (r'\\cite\{[^}]*\}', 'cite'),
        (r'\\ref\{[^}]*\}', 'ref'),
        (r'\\label\{[^}]*\}', 'label'),
        (r'\\includegraphics(\[[^\]]*\])?\{[^}]*\}', 'includegraphics'),
        (r'\\input\{[^}]*\}', 'input'),
        (r'\\include\{[^}]*\}', 'include'),
        # Comments
        (r'%.*$', 'comment'),
    ]
    
    # Find all non-translatable regions
    protected_regions = []
    for pattern, name in no_translate_patterns:
        for match in re.finditer(pattern, latex_content, re.DOTALL | re.MULTILINE):
            protected_regions.append((match.start(), match.end(), name))
    
    # Sort by start position
    protected_regions.sort(key=lambda x: x[0])
    
    # Merge overlapping regions
    merged = []
    for start, end, name in protected_regions:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]), merged[-1][2])
        else:
            merged.append((start, end, name))
    
    # Build segments list
    pos = 0
    for start, end, name in merged:
        if pos < start:
            # Translatable text before this protected region
            text = latex_content[pos:start]
            if text.strip():
                segments.append((pos, start, text, True))
        # Protected region
        segments.append((start, end, latex_content[start:end], False))
        pos = end
    
    # Remaining text after last protected region
    if pos < len(latex_content):
        text = latex_content[pos:]
        if text.strip():
            segments.append((pos, len(latex_content), text, True))
    
    return segments


def translate_latex_segment_openai(text: str, api_key: str, target_language: str) -> str:
    """Translate a LaTeX segment using OpenAI."""
    import openai
    
    if not text.strip():
        return text
    
    # Protect LaTeX commands
    placeholders = {}
    protected = text
    
    patterns = [
        r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})*',
        r'\$[^$]+\$',
        r'\\\[.*?\\\]',
        r'\\\(.*?\\\)',
    ]
    
    counter = 0
    for pattern in patterns:
        for match in re.finditer(pattern, protected, re.DOTALL):
            placeholder = f"__LATEX_{counter}__"
            placeholders[placeholder] = match.group()
            protected = protected.replace(match.group(), placeholder, 1)
            counter += 1
    
    system_prompt = f"""You are a scientific translator. Translate the text to {target_language}.
RULES:
- Keep all __LATEX_N__ placeholders exactly as they are
- Translate ONLY the natural language text
- Output ONLY the translation, no explanations"""

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": protected}
            ],
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        
        # Restore placeholders
        for key, value in placeholders.items():
            result = result.replace(key, value)
        
        return result
    except Exception as e:
        logger.warning(f"OpenAI translation error: {e}")
        return text


def translate_latex_segment(text: str, model: str, target_language: str, ollama_url: str = "http://localhost:11434") -> str:
    """
    Translates a LaTeX text segment while preserving inline commands.
    """
    import requests
    
    if not text or not text.strip():
        return text
    
    # Protect inline LaTeX commands
    placeholders = {}
    counter = [0]
    
    def protect_command(match):
        key = f"__LATEX_CMD_{counter[0]}__"
        placeholders[key] = match.group(0)
        counter[0] += 1
        return key
    
    # Protect common inline commands
    protected_text = text
    inline_patterns = [
        r'\\textbf\{[^}]*\}',
        r'\\textit\{[^}]*\}',
        r'\\emph\{[^}]*\}',
        r'\\underline\{[^}]*\}',
        r'\\footnote\{[^}]*\}',
        r'\\[a-zA-Z]+',  # Any command without braces
        r'\{[^}]*\}',  # Braced content (be careful)
    ]
    
    for pattern in inline_patterns[:5]:  # Only protect specific commands
        protected_text = re.sub(pattern, protect_command, protected_text)
    
    system_prompt = f"""You are a LaTeX document translator. Translate text to {target_language}.

CRITICAL RULES:
- Output ONLY the {target_language} translation
- Keep ALL placeholders like __LATEX_CMD_0__ exactly as they are
- Keep LaTeX commands (\\section, \\textbf, etc.) unchanged
- Keep author names unchanged
- Do NOT add any comments or explanations"""

    user_prompt = f"Translate to {target_language}:\n{protected_text}"
    
    try:
        response = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 4096}
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json().get("message", {}).get("content", text)
            
            # Restore placeholders
            for key, value in placeholders.items():
                result = result.replace(key, value)
            
            return result
    except Exception as e:
        logger.warning(f"Translation error: {e}")
    
    return text


def translate_latex_file(
    input_path: str,
    output_path: str,
    model: str,
    target_language: str,
    progress_callback=None,
    use_openai: bool = False,
    openai_api_key: str = None
) -> bool:
    """
    Translates a .tex file while preserving all LaTeX structure.
    
    Args:
        input_path: Path to input .tex file
        output_path: Path to output .tex file
        model: Ollama model name (ignored if use_openai=True)
        target_language: Target language
        progress_callback: Optional callback(current, total, status)
        use_openai: If True, use OpenAI instead of Ollama
        openai_api_key: OpenAI API key (required if use_openai=True)
    
    Returns:
        True if successful
    """
    try:
        # Read input file
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract segments
        segments = extract_translatable_segments(content)
        total_translatable = sum(1 for s in segments if s[3])
        
        if progress_callback:
            progress_callback(0, total_translatable, "Analyzing LaTeX structure...")
        
        # Translate segments
        translated_parts = []
        translate_count = 0
        
        for start, end, text, should_translate in segments:
            if should_translate:
                translate_count += 1
                if progress_callback:
                    progress_callback(translate_count, total_translatable, 
                                    f"Translating segment {translate_count}/{total_translatable}")
                
                if use_openai and openai_api_key:
                    translated = translate_latex_segment_openai(text, openai_api_key, target_language)
                else:
                    translated = translate_latex_segment(text, model, target_language)
                translated_parts.append(translated)
            else:
                translated_parts.append(text)
        
        # Reconstruct document
        result = ''.join(translated_parts)
        
        # Update babel language
        babel_map = {
            'german': 'ngerman', 'deutsch': 'ngerman',
            'italian': 'italian', 'italiano': 'italian',
            'french': 'french', 'français': 'french',
            'spanish': 'spanish', 'español': 'spanish',
            'english': 'english',
        }
        babel_lang = babel_map.get(target_language.lower(), 'english')
        result = re.sub(
            r'\\usepackage\[([^\]]*)\]\{babel\}',
            f'\\\\usepackage[{babel_lang}]{{babel}}',
            result
        )
        
        # Write output
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)
        
        if progress_callback:
            progress_callback(total_translatable, total_translatable, "Complete!")
        
        logger.info(f"Translated LaTeX file: {input_path} -> {output_path}")
        return True
        
    except Exception as e:
        logger.exception(f"LaTeX translation error: {e}")
        return False


# For testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python latex_translator.py input.tex output.tex [model] [target_lang]")
        sys.exit(1)
    
    input_tex = sys.argv[1]
    output_tex = sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "qwen2.5:7b"
    target = sys.argv[4] if len(sys.argv) > 4 else "German"
    
    print(f"Translating {input_tex} to {target} using {model}...")
    
    success = translate_latex_file(
        input_tex, output_tex, model, target,
        progress_callback=lambda c, t, s: print(f"  [{c}/{t}] {s}")
    )
    
    if success:
        print(f"Done! Output: {output_tex}")
    else:
        print("Translation failed!")
