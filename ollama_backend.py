"""
Ollama Backend for PDF-Translator
Fallback to OpenAI with local LLMs

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import logging
import subprocess
import requests
import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("pdf_translator.ollama")

# Ollama API base URL
OLLAMA_BASE_URL = "http://localhost:11434"

# Default context length for unknown models
DEFAULT_CONTEXT_LENGTH = 8192


# =============================================================================
# TRANSLATION CONTEXT - For consistent translations across document
# =============================================================================

@dataclass
class TranslationContext:
    """
    Maintains context for consistent translation across a document.
    
    Stores previous translations to provide context to the LLM,
    ensuring terminology consistency throughout the document.
    """
    target_language: str
    previous_translations: List[Tuple[str, str]] = field(default_factory=list)
    max_context_items: int = 5  # Keep last N translations for context
    glossary_context: str = ""
    
    def __post_init__(self):
        """Initialize glossary context."""
        try:
            from glossary import get_glossary_context
            self.glossary_context = get_glossary_context(self.target_language)
        except ImportError:
            logger.warning("Glossary module not available")
            self.glossary_context = ""
    
    def add_translation(self, original: str, translated: str):
        """Add a completed translation to context."""
        # Only keep meaningful translations (not too short)
        if len(original) > 20 and len(translated) > 20:
            self.previous_translations.append((
                original[:200],  # Truncate for efficiency
                translated[:200]
            ))
            # Keep only last N items
            if len(self.previous_translations) > self.max_context_items:
                self.previous_translations = self.previous_translations[-self.max_context_items:]
    
    def get_context_prompt(self) -> str:
        """Generate context section for LLM prompt."""
        parts = []
        
        # Add glossary
        if self.glossary_context:
            parts.append(self.glossary_context)
        
        # Add previous translations for consistency
        if self.previous_translations:
            parts.append("\nPREVIOUS TRANSLATIONS (maintain consistency):")
            for orig, trans in self.previous_translations[-3:]:  # Last 3 only
                parts.append(f"  Original: {orig[:100]}...")
                parts.append(f"  Translation: {trans[:100]}...")
                parts.append("")
        
        return "\n".join(parts)


# =============================================================================
# VRAM-BASED MODEL RECOMMENDATIONS
# =============================================================================

# VRAM-based model recommendations with context window sizes
# context_length: Model's native context window in tokens
# ~500 tokens ≈ 1 page of scientific text
OLLAMA_MODELS: Dict[str, Dict[str, Any]] = {
    # ============================================================
    # 4 GB VRAM (Minimal - works on most systems)
    # ============================================================
    "qwen2.5:3b": {
        "vram_min": 2,
        "vram_recommended": 4,
        "size_gb": 1.9,
        "context_length": 32768,
        "description": "Qwen 2.5 3B - Lightweight, good quality",
        "quality": "good",
    },
    # ============================================================
    # 8 GB VRAM
    # ============================================================
    "llama3.2:3b": {
        "vram_min": 4,
        "vram_recommended": 8,
        "size_gb": 2.0,
        "context_length": 131072,  # 128K context
        "description": "Llama 3.2 3B - Fast, 128K context",
        "quality": "good",
    },
    "phi3:mini": {
        "vram_min": 4,
        "vram_recommended": 8,
        "size_gb": 2.3,
        "context_length": 128000,  # 128K context
        "description": "Microsoft Phi-3 Mini - 128K context",
        "quality": "good",
    },
    "gemma2:2b": {
        "vram_min": 4,
        "vram_recommended": 8,
        "size_gb": 1.6,
        "context_length": 8192,  # 8K context
        "description": "Google Gemma 2 2B - Very fast, 8K context",
        "quality": "basic",
    },
    "mistral:7b-instruct-q4_0": {
        "vram_min": 4,
        "vram_recommended": 8,
        "size_gb": 4.1,
        "context_length": 32768,  # 32K context
        "description": "Mistral 7B Q4 - 32K context, compressed",
        "quality": "good",
    },
    # ============================================================
    # 16 GB VRAM
    # ============================================================
    "llama3.1:8b": {
        "vram_min": 8,
        "vram_recommended": 16,
        "size_gb": 4.7,
        "context_length": 131072,  # 128K context
        "description": "Llama 3.1 8B - 128K context, very good",
        "quality": "very good",
    },
    "mistral:7b": {
        "vram_min": 8,
        "vram_recommended": 16,
        "size_gb": 4.1,
        "context_length": 32768,  # 32K context
        "description": "Mistral 7B - 32K context, excellent translations",
        "quality": "very good",
    },
    "mistral-nemo:12b": {
        "vram_min": 8,
        "vram_recommended": 16,
        "size_gb": 7.1,
        "context_length": 131072,  # 128K context!
        "description": "Mistral Nemo 12B - 128K, NOT for translations",
        "quality": "good",  # Downgraded - bad for translations
    },
    "gemma2:9b": {
        "vram_min": 8,
        "vram_recommended": 16,
        "size_gb": 5.4,
        "context_length": 8192,  # 8K context
        "description": "Google Gemma 2 9B - 8K context",
        "quality": "very good",
    },
    "phi3:medium": {
        "vram_min": 10,
        "vram_recommended": 16,
        "size_gb": 7.9,
        "context_length": 128000,  # 128K context
        "description": "Microsoft Phi-3 Medium - 128K context",
        "quality": "excellent",
    },
    "qwen2.5:7b": {
        "vram_min": 8,
        "vram_recommended": 16,
        "size_gb": 4.7,
        "context_length": 131072,  # 128K context
        "description": "Qwen 2.5 7B - 128K, BEST for translations ⭐",
        "quality": "excellent",
    },
    "openchat:7b": {
        "vram_min": 8,
        "vram_recommended": 16,
        "size_gb": 4.1,
        "context_length": 8192,  # 8K context
        "description": "OpenChat 7B - 8K context",
        "quality": "very good",
    },
    "neural-chat:7b": {
        "vram_min": 8,
        "vram_recommended": 16,
        "size_gb": 4.1,
        "context_length": 8192,  # 8K context
        "description": "Intel Neural Chat 7B - 8K context",
        "quality": "very good",
    },
    # ============================================================
    # 24 GB VRAM
    # ============================================================
    "mistral-small:22b": {
        "vram_min": 14,
        "vram_recommended": 24,
        "size_gb": 13,
        "context_length": 32768,  # 32K context
        "description": "Mistral Small 22B - 32K context, very strong",
        "quality": "excellent",
    },
    "codestral:22b": {
        "vram_min": 14,
        "vram_recommended": 24,
        "size_gb": 13,
        "context_length": 32768,  # 32K context
        "description": "Codestral 22B - 32K context, Code & Text",
        "quality": "excellent",
    },
    "qwen2.5:14b": {
        "vram_min": 12,
        "vram_recommended": 24,
        "size_gb": 9.0,
        "context_length": 131072,  # 128K context
        "description": "Qwen 2.5 14B - 128K context, multilingual",
        "quality": "excellent",
    },
    "gpt-oss:20b": {
        "vram_min": 16,
        "vram_recommended": 24,
        "size_gb": 12.0,
        "context_length": 65536,
        "description": "GPT-OSS 20B - strong general model",
        "quality": "excellent",
    },
    # ============================================================
    # 32 GB VRAM
    # ============================================================
    "llama3.1:70b-instruct-q4_0": {
        "vram_min": 24,
        "vram_recommended": 32,
        "size_gb": 40,
        "context_length": 131072,  # 128K context
        "description": "Llama 3.1 70B Q4 - 128K context, near GPT-4",
        "quality": "excellent",
    },
    "mixtral:8x7b": {
        "vram_min": 24,
        "vram_recommended": 32,
        "size_gb": 26,
        "context_length": 32768,  # 32K context
        "description": "Mixtral 8x7B MoE - 32K context",
        "quality": "excellent",
    },
    "qwen2.5:32b": {
        "vram_min": 20,
        "vram_recommended": 32,
        "size_gb": 19,
        "context_length": 131072,  # 128K context
        "description": "Qwen 2.5 32B - 128K context, multilingual",
        "quality": "excellent",
    },
    "command-r:35b": {
        "vram_min": 22,
        "vram_recommended": 32,
        "size_gb": 20,
        "context_length": 131072,  # 128K context
        "description": "Command-R 35B - 128K context",
        "quality": "excellent",
    },
    # ============================================================
    # 48 GB VRAM
    # ============================================================
    "mixtral:8x22b": {
        "vram_min": 32,
        "vram_recommended": 48,
        "size_gb": 80,
        "context_length": 65536,  # 64K context
        "description": "Mixtral 8x22B - 64K context, largest MoE",
        "quality": "premium",
    },
    "mistral-large:123b-q4_0": {
        "vram_min": 40,
        "vram_recommended": 48,
        "size_gb": 70,
        "context_length": 131072,  # 128K context
        "description": "Mistral Large 123B Q4 - 128K context",
        "quality": "premium",
    },
    # ============================================================
    # 64 GB VRAM
    # ============================================================
    "llama3.1:70b": {
        "vram_min": 40,
        "vram_recommended": 64,
        "size_gb": 40,
        "context_length": 131072,  # 128K context
        "description": "Llama 3.1 70B - 128K context, top quality",
        "quality": "premium",
    },
    "qwen2.5:72b": {
        "vram_min": 45,
        "vram_recommended": 64,
        "size_gb": 43,
        "context_length": 131072,  # 128K context
        "description": "Qwen 2.5 72B - 128K context, best translations",
        "quality": "premium",
    },
    "command-r-plus:104b": {
        "vram_min": 55,
        "vram_recommended": 64,
        "size_gb": 60,
        "context_length": 131072,  # 128K context
        "description": "Command-R+ 104B - 128K context, enterprise",
        "quality": "premium",
    },
    # ============================================================
    # 96 GB VRAM
    # ============================================================
    "llama3.1:405b-instruct-q4_0": {
        "vram_min": 80,
        "vram_recommended": 96,
        "size_gb": 230,
        "context_length": 131072,  # 128K context
        "description": "Llama 3.1 405B Q4 - 128K context, largest",
        "quality": "maximum",
    },
    "mistral-large:123b": {
        "vram_min": 80,
        "vram_recommended": 96,
        "size_gb": 70,
        "context_length": 131072,  # 128K context
        "description": "Mistral Large 123B - 128K context, flagship",
        "quality": "maximum",
    },
}


# =============================================================================
# OLLAMA UTILITY FUNCTIONS
# =============================================================================

def check_ollama_installed() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def get_installed_models() -> List[str]:
    """Returns list of installed Ollama models."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            return models
    except Exception as e:
        logger.warning("Could not get installed models: %s", e)
    return []


def is_model_installed(model_name: str) -> bool:
    """Check if a specific model is installed."""
    installed = get_installed_models()
    # Check exact match or base name match
    base_name = model_name.split(":")[0]
    return any(
        m == model_name or m.startswith(base_name + ":")
        for m in installed
    )


def pull_model(model_name: str, progress_callback=None) -> bool:
    """
    Pull/download an Ollama model.
    
    Args:
        model_name: Name of the model to pull
        progress_callback: Optional callback(status, percent)
    
    Returns:
        True if successful
    """
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/pull",
            json={"name": model_name},
            stream=True,
            timeout=3600  # 1 hour timeout for large models
        )
        
        if response.status_code != 200:
            return False
        
        for line in response.iter_lines():
            if line:
                import json
                data = json.loads(line)
                status = data.get("status", "")
                
                # Parse progress
                if "completed" in data and "total" in data:
                    completed = data["completed"]
                    total = data["total"]
                    if total > 0:
                        percent = int(100 * completed / total)
                        if progress_callback:
                            progress_callback(status, percent)
                elif progress_callback:
                    progress_callback(status, 0)
        
        return True
    except Exception as e:
        logger.exception("Error pulling model: %s", e)
        return False


def get_models_for_vram(vram_gb: int) -> List[Dict]:
    """Returns models that fit in the given VRAM."""
    fitting = []
    for name, info in OLLAMA_MODELS.items():
        if info["vram_min"] <= vram_gb:
            fitting.append({
                "name": name,
                "size_gb": info["size_gb"],
                "description": info["description"],
                "fits_comfortably": info["vram_recommended"] <= vram_gb,
                "context_length": info.get("context_length", DEFAULT_CONTEXT_LENGTH),
            })
    return sorted(fitting, key=lambda x: x["size_gb"])


def get_models_for_vram_with_installed(vram_gb: int) -> List[Dict]:
    """Returns models for VRAM, marking which are installed."""
    models = get_models_for_vram(vram_gb)
    installed = get_installed_models()
    
    for model in models:
        base_name = model["name"].split(":")[0]
        model["installed"] = any(
            m == model["name"] or m.startswith(base_name + ":")
            for m in installed
        )
    
    return models


def detect_gpu_vram() -> Optional[int]:
    """
    Detects available GPU VRAM in GB.
    
    Returns:
        VRAM in GB or None if no GPU detected
    """
    # Try nvidia-smi first (NVIDIA GPUs)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # Get first GPU's memory in MB, convert to GB
            memory_mb = int(result.stdout.strip().split("\n")[0])
            vram_gb = memory_mb // 1024
            logger.info("Detected NVIDIA GPU with %d GB VRAM", vram_gb)
            return vram_gb
    except Exception:
        pass
    
    # Try PyTorch for broader GPU support
    try:
        import torch
        if torch.cuda.is_available():
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = vram_bytes // (1024 ** 3)
            logger.info("Detected GPU via PyTorch with %d GB VRAM", vram_gb)
            return vram_gb
    except Exception:
        pass
    
    logger.warning("No GPU detected")
    return None


def get_max_vram_for_system() -> int:
    """Returns maximum VRAM available, or 8 GB default."""
    detected = detect_gpu_vram()
    return detected if detected else 8


def get_model_context_length(model_name: str) -> int:
    """Returns the context length for a model."""
    if model_name in OLLAMA_MODELS:
        return OLLAMA_MODELS[model_name].get("context_length", DEFAULT_CONTEXT_LENGTH)
    
    # Try base name match
    base_name = model_name.split(":")[0]
    for name, info in OLLAMA_MODELS.items():
        if name.startswith(base_name):
            return info.get("context_length", DEFAULT_CONTEXT_LENGTH)
    
    return DEFAULT_CONTEXT_LENGTH


def get_token_limit_for_model(model_name: str, vram_gb: Optional[int] = None) -> int:
    """
    Returns optimal token limit based on model and available VRAM.
    
    Considers both the model's native context window and VRAM constraints.
    """
    if vram_gb is None:
        vram_gb = detect_gpu_vram() or 8
    
    # Get model's native context length
    model_context = get_model_context_length(model_name)
    
    # VRAM-based limits (conservative to avoid OOM)
    VRAM_TOKEN_MAP = {
        4:  4096,
        6:  8192,
        8:  16384,
        12: 32768,
        16: 65536,
        24: 98304,
        32: 131072,
        48: 131072,
        64: 131072,
        96: 131072,
    }
    
    vram_limit = 4096
    for vram_threshold, tokens in sorted(VRAM_TOKEN_MAP.items()):
        if vram_gb >= vram_threshold:
            vram_limit = tokens
    
    # Use minimum of model context and VRAM limit
    token_limit = min(model_context, vram_limit)
    
    logger.info("Model %s: context=%d, VRAM %dGB limit=%d → using %d tokens", 
                model_name, model_context, vram_gb, vram_limit, token_limit)
    return token_limit


def get_token_limit_for_vram(vram_gb: Optional[int] = None) -> int:
    """Returns optimal token limit based on available VRAM only."""
    if vram_gb is None:
        vram_gb = detect_gpu_vram() or 8
    
    VRAM_TOKEN_MAP = {
        4:  4096,
        6:  8192,
        8:  16384,
        12: 32768,
        16: 65536,
        24: 98304,
        32: 131072,
        48: 131072,
        64: 131072,
        96: 131072,
    }
    
    best_tokens = 4096
    for vram_threshold, tokens in sorted(VRAM_TOKEN_MAP.items()):
        if vram_gb >= vram_threshold:
            best_tokens = tokens
    
    return best_tokens


def get_page_estimate_for_model(model_name: str, vram_gb: Optional[int] = None) -> int:
    """Returns estimated pages based on model's context window and VRAM."""
    token_limit = get_token_limit_for_model(model_name, vram_gb)
    pages = token_limit // 500  # ~500 tokens per page
    return max(pages, 1)


def get_page_estimate_for_vram(vram_gb: Optional[int] = None) -> int:
    """Returns estimated pages based on VRAM only."""
    if vram_gb is None:
        vram_gb = detect_gpu_vram() or 8
    
    VRAM_PAGE_MAP = {
        4:  8,
        6:  16,
        8:  32,
        12: 65,
        16: 131,
        24: 196,
        32: 262,
        48: 262,
        64: 262,
        96: 262,
    }
    
    best_pages = 8
    for vram_threshold, pages in sorted(VRAM_PAGE_MAP.items()):
        if vram_gb >= vram_threshold:
            best_pages = pages
    
    return best_pages


# =============================================================================
# TRANSLATION FUNCTIONS
# =============================================================================

def translate_with_context(
    text: str,
    model: str,
    target_language: str,
    context: TranslationContext,
    element_type: str = "text",
    vram_gb: Optional[int] = None,
) -> str:
    """
    Translates text using Ollama WITH context for consistency.
    
    This is the preferred translation function as it maintains
    terminology consistency across the document.
    
    Args:
        text: Text to translate
        model: Ollama model name
        target_language: Target language
        context: TranslationContext for consistency
        element_type: Type of element (text, figure_caption, table_content)
        vram_gb: Available VRAM in GB (auto-detected if None)
    
    Returns:
        Translated text
    """
    if not text.strip():
        return text
    
    # Apply glossary protection
    try:
        from glossary import apply_glossary
        protected_text, restore_glossary = apply_glossary(text, target_language)
    except ImportError:
        protected_text = text
        restore_glossary = lambda x: x
    
    # Get token limit
    token_limit = get_token_limit_for_model(model, vram_gb)
    
    # Build context-aware prompt
    context_section = context.get_context_prompt()
    
    # Build system prompt with context
    system_prompt = f"""You are a scientific translator. Translate the user's text to {target_language}.

{context_section}

RULES:
- Output ONLY the {target_language} translation
- Do NOT include the original text
- Do NOT include these instructions in your output
- Keep all __GLOSS_*__ placeholders exactly as they are
- Keep math symbols, formulas, equations unchanged
- Keep author names unchanged
- Keep abbreviations unchanged
- Maintain consistent terminology with previous translations
- Translate everything else to {target_language}"""

    # Simple user prompt
    user_prompt = f"Translate to {target_language}:\n\n{protected_text}"
    
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": token_limit,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                },
            },
            timeout=300,
        )
        
        if response.status_code == 200:
            data = response.json()
            message = data.get("message", {})
            result = message.get("content", text)
            
            # Clean output
            result = _clean_translation_output(result, target_language)
            
            # Restore glossary placeholders
            result = restore_glossary(result)
            
            # Add to context for future translations
            context.add_translation(text, result)
            
            return result
        else:
            logger.error("Ollama API error: HTTP %d", response.status_code)
            return text
            
    except Exception as e:
        logger.exception("Ollama translation error: %s", e)
        return text


def translate_with_ollama(
    text: str,
    model: str,
    source_language: Optional[str],
    target_language: str,
    element_type: str = "text",
    vram_gb: Optional[int] = None,
) -> str:
    """
    Translates text using Ollama (legacy function without context).
    
    For new code, prefer translate_with_context() for better consistency.
    
    Args:
        text: Text to translate
        model: Ollama model name
        source_language: Source language (optional)
        target_language: Target language
        element_type: Type of element (text, figure_caption, table_content)
        vram_gb: Available VRAM in GB (auto-detected if None)
    
    Returns:
        Translated text
    """
    if not text.strip():
        return text
    
    # Apply glossary protection
    try:
        from glossary import apply_glossary, get_glossary_context
        protected_text, restore_glossary = apply_glossary(text, target_language)
        glossary_context = get_glossary_context(target_language)
    except ImportError:
        protected_text = text
        restore_glossary = lambda x: x
        glossary_context = ""
    
    # Get model-specific token limit
    token_limit = get_token_limit_for_model(model, vram_gb)
    
    # Build element-specific prompt
    if element_type == "figure_caption":
        task_hint = """This is a Figure Caption.
- Preserve "Figure X:" or "Fig. X" exactly
- Only translate the description after the number"""
    elif element_type == "table_content":
        task_hint = """This is Table Content.
- Maintain tabular structure
- Only translate cell text"""
    else:
        task_hint = ""
    
    system_prompt = f"""You are a scientific translator. Translate the user's text to {target_language}.

{glossary_context}

{task_hint}

RULES:
- Output ONLY the {target_language} translation
- Do NOT include the original text
- Do NOT include these instructions in your output
- Keep all __GLOSS_*__ placeholders exactly as they are
- Keep math symbols, formulas, equations unchanged
- Keep author names unchanged
- Keep abbreviations unchanged
- Translate everything else to {target_language}"""

    user_prompt = f"Translate to {target_language}:\n\n{protected_text}"
    
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": token_limit,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                },
            },
            timeout=300,
        )
        
        if response.status_code == 200:
            data = response.json()
            message = data.get("message", {})
            result = message.get("content", text)
            
            # Clean and restore
            result = _clean_translation_output(result, target_language)
            result = restore_glossary(result)
            
            return result
        else:
            logger.error("Ollama API error: HTTP %d", response.status_code)
            return text
            
    except Exception as e:
        logger.exception("Ollama translation error: %s", e)
        return text


def _clean_translation_output(text: str, target_language: str) -> str:
    """Removes any echoed prompts or meta-comments from translation output."""
    # Patterns that indicate echoed instructions
    bad_patterns = [
        r'^(ABSOLUTE RULES?|CRITICAL INSTRUCTIONS?|RULES?:).*?(?=\n\n|\Z)',
        r'^(TRADUZIONE|TRANSLATION|ÜBERSETZUNG|TRADUCTION):?\s*',
        r'^(Here is|Ecco|Hier ist|Voici).*?:\s*',
        r'^(Note|Nota|Hinweis|Remarque):.*?(?=\n\n|\Z)',
        r'^\d+\.\s*(OUTPUT|LANGUAGE|MATH|FORMULA|STRUCTURE|NAMES|NO META).*?(?=\n)',
        r'^(Translate to|Traduci in|Übersetze nach).*?:\s*',
        r'^(TEXT|TESTO|TEXTE):?\s*\n',
        r'^(GLOSSARY|PREVIOUS TRANSLATIONS).*?(?=\n\n)',
    ]
    
    result = text
    for pattern in bad_patterns:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
    # Clean up whitespace
    result = re.sub(r'\n{3,}', '\n\n', result)
    result = result.strip()
    
    return result


def get_vram_recommendations() -> str:
    """Returns formatted VRAM recommendations."""
    return """
## 🎮 VRAM Model Recommendations

| VRAM | Model | Size | Description |
|------|-------|------|-------------|
| **8 GB** | `llama3.2:3b` | 2 GB | Fast, good baseline |
| | `mistral:7b-instruct-q4_0` | 4.1 GB | Mistral compressed |
| | `phi3:mini` | 2.3 GB | Microsoft, efficient |
| **16 GB** | `llama3.1:8b` ⭐ | 4.7 GB | **Recommended!** Balanced |
| | `mistral:7b` | 4.1 GB | Excellent for translations |
| | `qwen2.5:7b` | 4.7 GB | Best for translations |
| | `openchat:7b` | 4.1 GB | ChatGPT alternative |
| **24 GB** | `mistral-small:22b` | 13 GB | Official Mistral Small |
| | `codestral:22b` | 13 GB | Mistral for Code & Text |
| | `gpt-oss:20b` | 12 GB | GPT-OSS 20B |
| **32 GB** | `mixtral:8x7b` | 26 GB | Mistral MoE, very strong |
| | `qwen2.5:32b` | 19 GB | Top for multilingual |
| **48+ GB** | `llama3.1:70b` | 40 GB | Premium quality |
| | `qwen2.5:72b` | 43 GB | State-of-the-art |
"""
