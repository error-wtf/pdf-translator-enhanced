#!/usr/bin/env python3
"""
OpenAI API Backend for PDF Translator
© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import os
import logging
from typing import Optional, Tuple, List, Dict, Any

logger = logging.getLogger("pdf_translator.openai")

# Try to import OpenAI
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("openai package not installed - pip install openai")

# Available OpenAI models
OPENAI_MODELS = {
    "gpt-4o": {"name": "GPT-4o", "context": 128000, "price": "$$$$"},
    "gpt-4o-mini": {"name": "GPT-4o Mini", "context": 128000, "price": "$$"},
    "gpt-4-turbo": {"name": "GPT-4 Turbo", "context": 128000, "price": "$$$"},
    "gpt-3.5-turbo": {"name": "GPT-3.5 Turbo", "context": 16385, "price": "$"},
}

def check_openai_available() -> bool:
    """Check if OpenAI is available and API key is set."""
    if not OPENAI_AVAILABLE:
        return False
    api_key = os.environ.get("OPENAI_API_KEY", "")
    return len(api_key) > 10

def get_openai_client() -> Optional["OpenAI"]:
    """Get OpenAI client instance."""
    if not OPENAI_AVAILABLE:
        return None
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

def get_openai_models() -> List[Tuple[str, str]]:
    """Get list of OpenAI models for dropdown."""
    choices = []
    for model_id, info in OPENAI_MODELS.items():
        label = f"🔑 {info['name']} ({info['price']})"
        choices.append((label, model_id))
    return choices

def translate_text_openai(
    text: str,
    target_language: str,
    model: str = "gpt-4o-mini",
    source_language: str = "auto"
) -> Tuple[bool, str]:
    """
    Translate text using OpenAI API.
    
    Args:
        text: Text to translate
        target_language: Target language (e.g., "German", "French")
        model: OpenAI model to use
        source_language: Source language or "auto"
        
    Returns:
        Tuple of (success, translated_text_or_error)
    """
    if not OPENAI_AVAILABLE:
        return False, "OpenAI package not installed. Run: pip install openai"
    
    client = get_openai_client()
    if not client:
        return False, "OpenAI API key not set. Set OPENAI_API_KEY environment variable."
    
    # Build system prompt
    system_prompt = f"""You are a professional translator. Translate the following text to {target_language}.

IMPORTANT RULES:
1. Keep ALL mathematical formulas, equations, and symbols UNCHANGED
2. Keep ALL numbers, units, and scientific notation UNCHANGED
3. Keep ALL placeholders like {{{{FORMULA_1}}}} or {{{{REF_1}}}} UNCHANGED
4. Keep ALL references like [1], [2-5], Fig. 1, Eq. (3) UNCHANGED
5. Keep ALL Greek letters (α, β, γ, etc.) and special symbols UNCHANGED
6. Translate ONLY the natural language text
7. Do NOT add any explanations or notes
8. Return ONLY the translated text"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            max_tokens=4096
        )
        
        translated = response.choices[0].message.content.strip()
        return True, translated
        
    except Exception as e:
        logger.error(f"OpenAI translation error: {e}")
        return False, f"OpenAI error: {str(e)}"

def translate_batch_openai(
    texts: List[str],
    target_language: str,
    model: str = "gpt-4o-mini"
) -> List[Tuple[bool, str]]:
    """
    Translate multiple texts using OpenAI API.
    
    Args:
        texts: List of texts to translate
        target_language: Target language
        model: OpenAI model to use
        
    Returns:
        List of (success, result) tuples
    """
    results = []
    for text in texts:
        result = translate_text_openai(text, target_language, model)
        results.append(result)
    return results

def check_api_key_valid(api_key: str) -> Tuple[bool, str]:
    """
    Check if an API key is valid by making a test request.
    
    Args:
        api_key: OpenAI API key to test
        
    Returns:
        Tuple of (valid, message)
    """
    if not OPENAI_AVAILABLE:
        return False, "OpenAI package not installed"
    
    if not api_key or len(api_key) < 10:
        return False, "Invalid API key format"
    
    try:
        client = OpenAI(api_key=api_key)
        # Make a minimal request to check validity
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        return True, "API key is valid"
    except Exception as e:
        return False, f"Invalid API key: {str(e)}"

def set_api_key(api_key: str) -> Tuple[bool, str]:
    """
    Set the OpenAI API key as environment variable.
    
    Args:
        api_key: OpenAI API key
        
    Returns:
        Tuple of (success, message)
    """
    if not api_key:
        return False, "No API key provided"
    
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Verify it works
    valid, msg = check_api_key_valid(api_key)
    if valid:
        return True, "API key set and verified successfully"
    else:
        return False, msg

def get_openai_status() -> str:
    """Get OpenAI status string for UI."""
    if not OPENAI_AVAILABLE:
        return "❌ OpenAI not installed (pip install openai)"
    
    if check_openai_available():
        return "✅ OpenAI API ready"
    else:
        return "⚠️ OpenAI API key not set"
