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
from typing import Optional, List, Dict, Any

logger = logging.getLogger("pdf_translator.ollama")

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
        "description": "Mistral Large 123B - 128K context",
        "quality": "maximum",
    },
}

# Default context length for unknown models - use 32K as most modern models support this
DEFAULT_CONTEXT_LENGTH = 32768

OLLAMA_BASE_URL = "http://localhost:11434"


def detect_gpu_vram() -> Optional[int]:
    """
    Automatically detects available GPU VRAM in GB.
    Supports NVIDIA (nvidia-smi), AMD (rocm-smi) and Apple Silicon.
    
    Returns:
        VRAM in GB or None if not detectable
    """
    import subprocess
    import platform
    
    # NVIDIA GPU (Windows/Linux)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Can have multiple GPUs, take the largest
            vram_values = [int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]
            if vram_values:
                vram_mb = max(vram_values)
                vram_gb = vram_mb // 1024
                logger.info("Detected NVIDIA GPU with %d GB VRAM", vram_gb)
                return vram_gb
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    
    # AMD GPU (Linux with ROCm)
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Parse ROCm output
            for line in result.stdout.split("\n"):
                if "Total" in line and "Memory" in line:
                    # Format: "GPU[0] : VRAM Total Memory (B): 17163091968"
                    parts = line.split(":")
                    if len(parts) >= 2:
                        try:
                            vram_bytes = int(parts[-1].strip())
                            vram_gb = vram_bytes // (1024 ** 3)
                            logger.info("Detected AMD GPU with %d GB VRAM", vram_gb)
                            return vram_gb
                        except ValueError:
                            pass
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Apple Silicon (macOS) - Unified Memory
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                total_mem_bytes = int(result.stdout.strip())
                # Apple Silicon shares RAM with GPU, approx. 75% usable for ML
                usable_gb = (total_mem_bytes // (1024 ** 3)) * 3 // 4
                logger.info("Detected Apple Silicon with ~%d GB usable for ML", usable_gb)
                return usable_gb
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
    
    logger.warning("Could not detect GPU VRAM automatically")
    return None


def get_max_vram_for_system() -> int:
    """
    Returns the maximum VRAM for the system.
    If not detectable, 8 GB is used as a safe default.
    """
    detected = detect_gpu_vram()
    if detected:
        return detected
    return 8  # Safe default


def get_models_for_vram(vram_gb: int) -> List[Dict[str, Any]]:
    """Returns suitable models for the specified VRAM size."""
    suitable = []
    for model_name, info in OLLAMA_MODELS.items():
        if info["vram_min"] <= vram_gb:
            suitable.append({
                "name": model_name,
                **info,
                "fits_comfortably": info["vram_recommended"] <= vram_gb,
            })
    # Sort by quality (best first that still fit)
    quality_order = {"maximum": 5, "premium": 4, "excellent": 3, "very good": 2, "good": 1, "basic": 0}
    suitable.sort(key=lambda x: (x["fits_comfortably"], quality_order.get(x["quality"], 0)), reverse=True)
    return suitable


def check_ollama_installed() -> bool:
    """Checks if Ollama is installed and reachable."""
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
            return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        logger.warning("Could not fetch installed models: %s", e)
    return []


def is_model_installed(model_name: str) -> bool:
    """Checks if a specific model is installed."""
    installed = get_installed_models()
    # Exact match or match without tag (e.g. "llama3.1:8b" matches "llama3.1:8b-instruct")
    base_name = model_name.split(":")[0]
    for m in installed:
        if m == model_name or m.startswith(f"{base_name}:"):
            return True
    return False


def get_installed_model_version(model_name: str) -> Optional[str]:
    """Returns the installed version of a model, if present."""
    installed = get_installed_models()
    base_name = model_name.split(":")[0]
    for m in installed:
        if m == model_name:
            return m
        if m.startswith(f"{base_name}:"):
            return m
    return None


def get_models_for_vram_with_installed(vram_gb: int) -> List[Dict[str, Any]]:
    """
    Returns suitable models for the specified VRAM size.
    Already installed models are marked and sorted preferentially.
    """
    installed = get_installed_models()
    suitable = []
    
    for model_name, info in OLLAMA_MODELS.items():
        if info["vram_min"] <= vram_gb:
            # Check if installed (exact or similar version)
            base_name = model_name.split(":")[0]
            is_installed = False
            installed_version = None
            
            for m in installed:
                if m == model_name:
                    is_installed = True
                    installed_version = m
                    break
                if m.startswith(f"{base_name}:"):
                    is_installed = True
                    installed_version = m
                    break
            
            suitable.append({
                "name": model_name,
                **info,
                "fits_comfortably": info["vram_recommended"] <= vram_gb,
                "is_installed": is_installed,
                "installed_version": installed_version,
            })
    
    # Sort: Installed first, then by quality
    quality_order = {"maximum": 5, "premium": 4, "excellent": 3, "very good": 2, "good": 1, "basic": 0}
    suitable.sort(
        key=lambda x: (
            x["is_installed"],  # Installed first
            x["fits_comfortably"],
            quality_order.get(x["quality"], 0)
        ),
        reverse=True
    )
    return suitable


def pull_model(model_name: str, progress_callback=None) -> bool:
    """
    Downloads an Ollama model.
    
    Args:
        model_name: Name of the model (e.g. "llama3.1:8b")
        progress_callback: Optional callback(status: str, percent: int)
    
    Returns:
        True if successful
    """
    logger.info("Pulling Ollama model: %s", model_name)
    
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/pull",
            json={"name": model_name},
            stream=True,
            timeout=3600,  # 1 hour for large models
        )
        
        if response.status_code != 200:
            logger.error("Failed to pull model: HTTP %d", response.status_code)
            return False
        
        for line in response.iter_lines():
            if line:
                import json
                try:
                    data = json.loads(line)
                    status = data.get("status", "")
                    
                    if progress_callback:
                        # Calculate progress if possible
                        completed = data.get("completed", 0)
                        total = data.get("total", 0)
                        percent = int(completed / total * 100) if total > 0 else 0
                        progress_callback(status, percent)
                    
                    if status == "success":
                        logger.info("Model %s pulled successfully", model_name)
                        return True
                        
                except json.JSONDecodeError:
                    pass
        
        return True
        
    except Exception as e:
        logger.exception("Error pulling model %s: %s", model_name, e)
        return False


def get_model_context_length(model_name: str) -> int:
    """
    Returns the context length for a specific model.
    
    Args:
        model_name: Ollama model name (e.g., "mistral-nemo:12b")
    
    Returns:
        Context length in tokens
    """
    # Check exact match first
    if model_name in OLLAMA_MODELS:
        return OLLAMA_MODELS[model_name].get("context_length", DEFAULT_CONTEXT_LENGTH)
    
    # Check base model name (without tag)
    base_name = model_name.split(":")[0].lower()
    for name, info in OLLAMA_MODELS.items():
        if name.lower().startswith(base_name) or base_name.startswith(name.split(":")[0].lower()):
            return info.get("context_length", DEFAULT_CONTEXT_LENGTH)
    
    # Heuristics for unknown models based on name patterns
    name_lower = model_name.lower()
    
    # Most modern large models support 128K
    if any(x in name_lower for x in ['llama3', 'qwen', 'mistral', 'phi3', 'command-r']):
        return 131072  # 128K
    
    # Older or smaller models typically have 32K
    if any(x in name_lower for x in ['llama2', 'codellama', 'vicuna', 'openchat']):
        return 32768  # 32K
    
    # GPT-like models (custom fine-tunes) - assume good context
    if 'gpt' in name_lower:
        return 65536  # 64K
    
    return DEFAULT_CONTEXT_LENGTH


def get_token_limit_for_model(model_name: str, vram_gb: Optional[int] = None) -> int:
    """
    Returns optimal token limit based on model's context window AND available VRAM.
    
    Uses the MINIMUM of:
    - Model's native context length
    - VRAM-based safe limit
    
    Args:
        model_name: Ollama model name
        vram_gb: Available VRAM in GB (auto-detected if None)
    
    Returns:
        num_predict token limit
    """
    if vram_gb is None:
        vram_gb = detect_gpu_vram() or 8
    
    # Get model's native context length
    model_context = get_model_context_length(model_name)
    
    # VRAM-based limits - AGGRESSIVE for maximum context usage
    # Modern models handle large contexts well, so we push limits
    VRAM_TOKEN_MAP = {
        4:  4096,      # 4GB: ~8 pages
        6:  8192,      # 6GB: ~16 pages
        8:  16384,     # 8GB: ~32 pages
        12: 32768,     # 12GB: ~65 pages
        16: 65536,     # 16GB: ~131 pages (128K models)
        24: 98304,     # 24GB: ~196 pages
        32: 131072,    # 32GB: ~262 pages (full 128K)
        48: 131072,    # 48GB: full context
        64: 131072,    # 64GB: full context
        96: 131072,    # 96GB: full context
    }
    
    # Find VRAM-based limit - use detected VRAM directly
    vram_limit = 4096  # Minimum
    for vram_threshold, tokens in sorted(VRAM_TOKEN_MAP.items()):
        if vram_gb >= vram_threshold:
            vram_limit = tokens
    
    # Use minimum of model context and VRAM limit
    token_limit = min(model_context, vram_limit)
    
    logger.info("Model %s: context=%d, VRAM %dGB limit=%d → using %d tokens", 
                model_name, model_context, vram_gb, vram_limit, token_limit)
    return token_limit


def get_token_limit_for_vram(vram_gb: Optional[int] = None) -> int:
    """
    Returns optimal token limit based on available VRAM only.
    DEPRECATED: Use get_token_limit_for_model() for model-specific limits.
    
    Returns:
        num_predict token limit
    """
    if vram_gb is None:
        vram_gb = detect_gpu_vram() or 8
    
    # AGGRESSIVE token limits for maximum context
    VRAM_TOKEN_MAP = {
        4:  4096,      # 4GB: ~8 pages
        6:  8192,      # 6GB: ~16 pages
        8:  16384,     # 8GB: ~32 pages
        12: 32768,     # 12GB: ~65 pages
        16: 65536,     # 16GB: ~131 pages
        24: 98304,     # 24GB: ~196 pages
        32: 131072,    # 32GB: ~262 pages
        48: 131072,    # 48GB: full 128K
        64: 131072,    # 64GB: full 128K
        96: 131072,    # 96GB: full 128K
    }
    
    best_tokens = 4096
    for vram_threshold, tokens in sorted(VRAM_TOKEN_MAP.items()):
        if vram_gb >= vram_threshold:
            best_tokens = tokens
    
    logger.info("VRAM %d GB → token limit %d", vram_gb, best_tokens)
    return best_tokens


def get_page_estimate_for_model(model_name: str, vram_gb: Optional[int] = None) -> int:
    """
    Returns estimated number of pages based on model's context window and VRAM.
    
    Based on ~500 tokens per page average for scientific papers.
    """
    token_limit = get_token_limit_for_model(model_name, vram_gb)
    pages = token_limit // 500  # ~500 tokens per page
    return max(pages, 1)  # At least 1 page


def get_page_estimate_for_vram(vram_gb: Optional[int] = None) -> int:
    """
    Returns estimated number of pages based on VRAM only.
    """
    if vram_gb is None:
        vram_gb = detect_gpu_vram() or 8
    
    # AGGRESSIVE page estimates matching new token limits
    VRAM_PAGE_MAP = {
        4:  8,       # 4GB: ~8 pages
        6:  16,      # 6GB: ~16 pages
        8:  32,      # 8GB: ~32 pages
        12: 65,      # 12GB: ~65 pages
        16: 131,     # 16GB: ~131 pages
        24: 196,     # 24GB: ~196 pages
        32: 262,     # 32GB: ~262 pages
        48: 262,     # 48GB: full context
        64: 262,     # 64GB: full context
        96: 262,     # 96GB: full context
    }
    
    best_pages = 8
    for vram_threshold, pages in sorted(VRAM_PAGE_MAP.items()):
        if vram_gb >= vram_threshold:
            best_pages = pages
    
    return best_pages


def translate_with_ollama(
    text: str,
    model: str,
    source_language: Optional[str],
    target_language: str,
    element_type: str = "text",
    vram_gb: Optional[int] = None,
) -> str:
    """
    Translates text using Ollama with VRAM-optimized token limits.
    
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
    
    # Get model-specific token limit (considers both model context and VRAM)
    token_limit = get_token_limit_for_model(model, vram_gb)
    
    # Prompt basierend auf Element-Typ - DETAILLIERT wie OpenAI-Version
    if element_type == "figure_caption":
        task = f"""You are a professional scientific translator.

Task:
- This is a **Figure Caption**. Translate it from {source_language or "the source language"} to {target_language}.
- **Preserve the leading text** like "Figure 1:" or "Fig. 3.2." exactly as it is, only translating the descriptive text that follows.
- Preserve *all* original LaTeX math segments as they are.
- Do NOT change anything inside math segments ($...$, \\[...\\], \\begin{{equation}}, etc.).
- Output only the translation, no comments or explanations."""

    elif element_type == "table_content":
        task = f"""You are a professional scientific translator.

Task:
- This is **Table Content (potentially with a title/caption)**. Translate it from {source_language or "the source language"} to {target_language}.
- **Strictly maintain the tabular structure and alignment** using Markdown, LaTeX, or other clear text formatting.
- Preserve row/column delineation.
- Only translate the text within the table (headers, cells, notes).
- Preserve *all* original LaTeX math segments as they are.
- Do NOT change anything inside math segments.
- Output only the translation, no comments or explanations."""

    else:
        # Simple, direct prompt - no complex rules that model might echo back
        task = ""

    # Direct prompt without rules that could be echoed
    prompt = f"""Translate to {target_language}:

{text}"""

    # All rules go in system prompt only (not echoed by model)
    system_prompt = f"""You are a scientific translator. Translate the user's text to {target_language}.

Rules:
- Output ONLY the {target_language} translation
- Do NOT include the original text
- Do NOT include these instructions in your output
- Keep math symbols, formulas, equations unchanged
- Keep author names (Press, Teukolsky, Casu, Wrede) unchanged
- Keep abbreviations (SSZ, GR, ApJ) unchanged
- Translate everything else to {target_language}"""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Very low for consistent translations
                    "num_predict": token_limit,  # VRAM-optimized token limit
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,  # Reduce repetition
                },
            },
            timeout=300,  # 5 minutes per block
        )
        
        if response.status_code == 200:
            data = response.json()
            # /api/chat returns message.content instead of response
            message = data.get("message", {})
            result = message.get("content", text)
            
            # Post-process: Remove any echoed instructions/prompts
            result = _clean_translation_output(result, target_language)
            return result
        else:
            logger.error("Ollama API error: HTTP %d", response.status_code)
            return text
            
    except Exception as e:
        logger.exception("Ollama translation error: %s", e)
        return text


def _clean_translation_output(text: str, target_language: str) -> str:
    """
    Removes any echoed prompts or meta-comments from translation output.
    """
    import re
    
    # Patterns that indicate echoed instructions (in various languages)
    bad_patterns = [
        r'^(ABSOLUTE RULES?|CRITICAL INSTRUCTIONS?|RULES?:).*?(?=\n\n|\Z)',
        r'^(TRADUZIONE|TRANSLATION|ÜBERSETZUNG|TRADUCTION):?\s*',
        r'^(Here is|Ecco|Hier ist|Voici).*?:\s*',
        r'^(Note|Nota|Hinweis|Remarque):.*?(?=\n\n|\Z)',
        r'^\d+\.\s*(OUTPUT|LANGUAGE|MATH|FORMULA|STRUCTURE|NAMES|NO META).*?(?=\n)',
        r'^(Translate to|Traduci in|Übersetze nach).*?:\s*',
        r'^(TEXT|TESTO|TEXTE):?\s*\n',
    ]
    
    result = text
    for pattern in bad_patterns:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
    # Remove leading/trailing whitespace and multiple newlines
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
| | `mistral-nemo:12b` | 7.1 GB | Latest Mistral |
| | `openchat:7b` | 4.1 GB | ChatGPT alternative |
| | `neural-chat:7b` | 4.1 GB | Intel, optimized for chat |
| **24 GB** | `mistral-small:22b` | 13 GB | Official Mistral Small |
| | `codestral:22b` | 13 GB | Mistral for Code & Text |
| | `openchat:8b` | 4.9 GB | Improved ChatGPT alternative |
| **32 GB** | `mixtral:8x7b` | 26 GB | Mistral MoE, very strong |
| | `qwen2.5:32b` | 19 GB | Top for multilingual |
| | `command-r:35b` | 20 GB | Cohere, strong for instructions |
| **48 GB** | `mixtral:8x22b` | 80 GB | Largest Mistral MoE |
| | `mistral-large:123b-q4_0` | 70 GB | Mistral Flagship Q4 |
| **64 GB** | `llama3.1:70b` | 40 GB | Premium quality |
| | `qwen2.5:72b` | 43 GB | State-of-the-art |
| | `command-r-plus:104b` | 60 GB | Cohere Enterprise |
| **96 GB** | `llama3.1:405b-q4` | 230 GB | Largest Open-Source |
| | `mistral-large:123b` | 70 GB | Mistral Flagship Full |
"""
