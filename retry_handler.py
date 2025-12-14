"""
Retry Handler for LLM API Calls

Provides robust retry logic for handling:
- Timeouts
- Rate limits
- Network errors
- Model overload

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import logging
import time
import random
from functools import wraps
from typing import Callable, TypeVar, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("pdf_translator.retry")

T = TypeVar("T")


class RetryStrategy(Enum):
    """Retry strategies for different error types."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CONSTANT = "constant"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    
    # Error-specific settings
    timeout_retries: int = 5
    rate_limit_retries: int = 10
    rate_limit_delay: float = 30.0


# Default configuration
DEFAULT_CONFIG = RetryConfig()


class RetryableError(Exception):
    """Base class for retryable errors."""
    pass


class TimeoutError(RetryableError):
    """LLM request timed out."""
    pass


class RateLimitError(RetryableError):
    """Rate limit exceeded."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class ModelOverloadError(RetryableError):
    """Model is overloaded."""
    pass


class NetworkError(RetryableError):
    """Network connectivity error."""
    pass


def calculate_delay(
    attempt: int,
    config: RetryConfig,
    error: Optional[Exception] = None
) -> float:
    """Calculate delay before next retry attempt."""
    
    # Check for rate limit with retry-after header
    if isinstance(error, RateLimitError) and error.retry_after:
        base_delay = error.retry_after
    elif isinstance(error, RateLimitError):
        base_delay = config.rate_limit_delay
    else:
        if config.strategy == RetryStrategy.EXPONENTIAL:
            base_delay = config.initial_delay * (config.exponential_base ** attempt)
        elif config.strategy == RetryStrategy.LINEAR:
            base_delay = config.initial_delay * (attempt + 1)
        else:  # CONSTANT
            base_delay = config.initial_delay
    
    # Cap at max delay
    base_delay = min(base_delay, config.max_delay)
    
    # Add jitter to prevent thundering herd
    if config.jitter:
        jitter = random.uniform(0, base_delay * 0.1)
        base_delay += jitter
    
    return base_delay


def classify_error(error: Exception) -> Tuple[bool, str]:
    """
    Classify an error as retryable or not.
    
    Returns (is_retryable, error_type)
    """
    error_str = str(error).lower()
    error_type = type(error).__name__
    
    # Timeout errors
    if any(x in error_str for x in ["timeout", "timed out", "deadline exceeded"]):
        return True, "timeout"
    
    # Rate limit errors
    if any(x in error_str for x in ["rate limit", "too many requests", "429"]):
        return True, "rate_limit"
    
    # Model overload
    if any(x in error_str for x in ["overload", "503", "service unavailable", "busy"]):
        return True, "overload"
    
    # Network errors
    if any(x in error_str for x in ["connection", "network", "unreachable", "refused"]):
        return True, "network"
    
    # Temporary server errors
    if any(x in error_str for x in ["500", "502", "504", "internal server"]):
        return True, "server_error"
    
    # Context length exceeded - not retryable with same input
    if any(x in error_str for x in ["context length", "too long", "max tokens"]):
        return False, "context_length"
    
    # Invalid input - not retryable
    if any(x in error_str for x in ["invalid", "malformed", "bad request", "400"]):
        return False, "invalid_input"
    
    # Authentication errors - not retryable
    if any(x in error_str for x in ["auth", "401", "403", "unauthorized", "forbidden"]):
        return False, "auth_error"
    
    # Default: not retryable
    return False, "unknown"


def with_retry(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
):
    """
    Decorator for adding retry logic to functions.
    
    Args:
        config: Retry configuration
        on_retry: Callback called before each retry (attempt, error, delay)
    
    Example:
        @with_retry(RetryConfig(max_retries=5))
        def translate_text(text: str) -> str:
            # ... translation logic ...
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_error: Optional[Exception] = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                
                except Exception as e:
                    last_error = e
                    is_retryable, error_type = classify_error(e)
                    
                    # Check if we should retry
                    if not is_retryable:
                        logger.warning(
                            f"Non-retryable error ({error_type}): {e}"
                        )
                        raise
                    
                    # Check if we have retries left
                    if attempt >= config.max_retries:
                        logger.error(
                            f"Max retries ({config.max_retries}) exceeded: {e}"
                        )
                        raise
                    
                    # Calculate delay
                    delay = calculate_delay(attempt, config, e)
                    
                    logger.warning(
                        f"Retryable error ({error_type}), "
                        f"attempt {attempt + 1}/{config.max_retries + 1}, "
                        f"waiting {delay:.1f}s: {e}"
                    )
                    
                    # Call retry callback if provided
                    if on_retry:
                        on_retry(attempt, e, delay)
                    
                    # Wait before retry
                    time.sleep(delay)
            
            # Should not reach here, but just in case
            if last_error:
                raise last_error
            raise RuntimeError("Unexpected retry loop exit")
        
        return wrapper
    return decorator


class RetryHandler:
    """
    Context manager for retry logic.
    
    Example:
        handler = RetryHandler(config)
        
        while handler.should_retry():
            try:
                result = translate_text(text)
                break
            except Exception as e:
                handler.handle_error(e)
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.attempt = 0
        self.last_error: Optional[Exception] = None
        self.total_delay = 0.0
    
    def should_retry(self) -> bool:
        """Check if we should attempt (another) try."""
        return self.attempt <= self.config.max_retries
    
    def handle_error(self, error: Exception) -> bool:
        """
        Handle an error and decide whether to retry.
        
        Returns True if retrying, raises if not retryable.
        """
        self.last_error = error
        is_retryable, error_type = classify_error(error)
        
        if not is_retryable:
            logger.warning(f"Non-retryable error ({error_type}): {error}")
            raise error
        
        if self.attempt >= self.config.max_retries:
            logger.error(f"Max retries exceeded: {error}")
            raise error
        
        # Calculate and apply delay
        delay = calculate_delay(self.attempt, self.config, error)
        self.total_delay += delay
        
        logger.warning(
            f"Retry {self.attempt + 1}/{self.config.max_retries}: "
            f"{error_type}, waiting {delay:.1f}s"
        )
        
        time.sleep(delay)
        self.attempt += 1
        
        return True
    
    def get_stats(self) -> dict:
        """Get retry statistics."""
        return {
            "attempts": self.attempt,
            "total_delay": self.total_delay,
            "last_error": str(self.last_error) if self.last_error else None,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def translate_with_retry(
    translate_func: Callable[[str], str],
    text: str,
    config: Optional[RetryConfig] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> str:
    """
    Translate text with automatic retry handling.
    
    Args:
        translate_func: Function that performs translation
        text: Text to translate
        config: Retry configuration
        progress_callback: Called with status updates
    
    Returns:
        Translated text
    """
    config = config or DEFAULT_CONFIG
    handler = RetryHandler(config)
    
    while handler.should_retry():
        try:
            return translate_func(text)
        except Exception as e:
            if progress_callback:
                progress_callback(
                    f"⚠️ Retry {handler.attempt + 1}/{config.max_retries}: {type(e).__name__}"
                )
            handler.handle_error(e)
    
    raise RuntimeError("Translation failed after all retries")


def batch_translate_with_retry(
    translate_func: Callable[[str], str],
    texts: list[str],
    config: Optional[RetryConfig] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> list[str]:
    """
    Translate multiple texts with retry handling per item.
    
    Failed translations are marked with error placeholder.
    """
    config = config or DEFAULT_CONFIG
    results = []
    
    for i, text in enumerate(texts):
        if progress_callback:
            progress_callback(i + 1, len(texts), f"Translating block {i + 1}")
        
        try:
            result = translate_with_retry(
                translate_func, 
                text, 
                config,
                lambda msg: progress_callback(i + 1, len(texts), msg) if progress_callback else None
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Block {i + 1} failed after retries: {e}")
            results.append(f"[TRANSLATION FAILED: {type(e).__name__}]")
    
    return results
