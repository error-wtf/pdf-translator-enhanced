"""
Translation Cache - Cache repeated translations for speed

Scientific papers often have repeated phrases:
- "as shown in Figure X"
- "according to Equation Y"
- Standard terminology

Caching these saves significant time and ensures consistency.

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger("pdf_translator.cache")


# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

DEFAULT_CACHE_DIR = Path.home() / ".pdf_translator_cache"
DEFAULT_TTL = 30 * 24 * 3600  # 30 days in seconds
MAX_CACHE_SIZE_MB = 500


@dataclass
class CacheStats:
    """Cache statistics."""
    total_entries: int = 0
    hits: int = 0
    misses: int = 0
    size_bytes: int = 0
    oldest_entry: Optional[float] = None
    newest_entry: Optional[float] = None
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total * 100 if total > 0 else 0.0
    
    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


# =============================================================================
# CACHE KEY GENERATION
# =============================================================================

def generate_cache_key(
    text: str,
    target_language: str,
    model: str,
    include_model: bool = True
) -> str:
    """
    Generate a unique cache key for a translation.
    
    Key includes:
    - Text content (hashed)
    - Target language
    - Model (optional, for model-specific caching)
    """
    # Normalize text
    normalized = text.strip().lower()
    
    # Create hash
    text_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    if include_model:
        return f"{target_language}:{model}:{text_hash}"
    else:
        return f"{target_language}:any:{text_hash}"


# =============================================================================
# SQLITE CACHE
# =============================================================================

class TranslationCache:
    """
    SQLite-based translation cache.
    
    Provides persistent caching with:
    - Fast hash-based lookup
    - TTL expiration
    - Size management
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        ttl: int = DEFAULT_TTL,
        max_size_mb: int = MAX_CACHE_SIZE_MB,
    ):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.cache_dir / "translations.db"
        self.ttl = ttl
        self.max_size_mb = max_size_mb
        
        self._stats = CacheStats()
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS translations (
                    cache_key TEXT PRIMARY KEY,
                    original_text TEXT NOT NULL,
                    translated_text TEXT NOT NULL,
                    target_language TEXT NOT NULL,
                    model TEXT,
                    created_at REAL NOT NULL,
                    accessed_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_language 
                ON translations(target_language)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created 
                ON translations(created_at)
            """)
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def get(
        self,
        text: str,
        target_language: str,
        model: str = "any"
    ) -> Optional[str]:
        """
        Get a cached translation.
        
        Returns translated text if found, None otherwise.
        """
        key = generate_cache_key(text, target_language, model)
        
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT translated_text, created_at 
                FROM translations 
                WHERE cache_key = ?
                """,
                (key,)
            )
            row = cursor.fetchone()
            
            if row:
                # Check TTL
                if time.time() - row["created_at"] > self.ttl:
                    # Expired - delete and return None
                    conn.execute("DELETE FROM translations WHERE cache_key = ?", (key,))
                    conn.commit()
                    self._stats.misses += 1
                    return None
                
                # Update access time and count
                conn.execute(
                    """
                    UPDATE translations 
                    SET accessed_at = ?, access_count = access_count + 1 
                    WHERE cache_key = ?
                    """,
                    (time.time(), key)
                )
                conn.commit()
                
                self._stats.hits += 1
                logger.debug(f"Cache hit: {key[:30]}...")
                return row["translated_text"]
        
        self._stats.misses += 1
        return None
    
    def put(
        self,
        text: str,
        translated: str,
        target_language: str,
        model: str = "any"
    ):
        """Store a translation in the cache."""
        key = generate_cache_key(text, target_language, model)
        now = time.time()
        
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO translations 
                (cache_key, original_text, translated_text, target_language, model, created_at, accessed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (key, text, translated, target_language, model, now, now)
            )
            conn.commit()
        
        logger.debug(f"Cached: {key[:30]}...")
        
        # Periodically check size
        if self._stats.hits + self._stats.misses % 100 == 0:
            self._check_size()
    
    def _check_size(self):
        """Check cache size and evict if necessary."""
        size = self.db_path.stat().st_size if self.db_path.exists() else 0
        size_mb = size / (1024 * 1024)
        
        if size_mb > self.max_size_mb:
            self._evict_old_entries(int(size_mb - self.max_size_mb * 0.8))
    
    def _evict_old_entries(self, target_reduction_mb: int):
        """Evict oldest entries to reduce cache size."""
        logger.info(f"Evicting old cache entries (target: -{target_reduction_mb}MB)")
        
        with self._get_connection() as conn:
            # Delete oldest 20% of entries
            conn.execute(
                """
                DELETE FROM translations 
                WHERE cache_key IN (
                    SELECT cache_key FROM translations 
                    ORDER BY accessed_at ASC 
                    LIMIT (SELECT COUNT(*) / 5 FROM translations)
                )
                """
            )
            conn.commit()
            conn.execute("VACUUM")
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM translations")
            self._stats.total_entries = cursor.fetchone()["cnt"]
            
            cursor = conn.execute(
                "SELECT MIN(created_at) as oldest, MAX(created_at) as newest FROM translations"
            )
            row = cursor.fetchone()
            self._stats.oldest_entry = row["oldest"]
            self._stats.newest_entry = row["newest"]
        
        self._stats.size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        
        return self._stats
    
    def clear(self):
        """Clear all cached translations."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM translations")
            conn.commit()
            conn.execute("VACUUM")
        
        self._stats = CacheStats()
        logger.info("Cache cleared")
    
    def clear_language(self, target_language: str):
        """Clear cache for a specific language."""
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM translations WHERE target_language = ?",
                (target_language,)
            )
            conn.commit()
        
        logger.info(f"Cache cleared for language: {target_language}")
    
    def export_to_json(self, path: str):
        """Export cache to JSON file."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM translations")
            rows = [dict(row) for row in cursor.fetchall()]
        
        Path(path).write_text(json.dumps(rows, indent=2))
        logger.info(f"Exported {len(rows)} entries to {path}")
    
    def import_from_json(self, path: str) -> int:
        """Import cache from JSON file. Returns count of imported entries."""
        rows = json.loads(Path(path).read_text())
        
        with self._get_connection() as conn:
            for row in rows:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO translations 
                    (cache_key, original_text, translated_text, target_language, model, created_at, accessed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["cache_key"],
                        row["original_text"],
                        row["translated_text"],
                        row["target_language"],
                        row.get("model", "any"),
                        row["created_at"],
                        row["accessed_at"],
                    )
                )
            conn.commit()
        
        logger.info(f"Imported {len(rows)} entries from {path}")
        return len(rows)


# =============================================================================
# CACHED TRANSLATION WRAPPER
# =============================================================================

# Global cache instance
_cache: Optional[TranslationCache] = None


def get_cache() -> TranslationCache:
    """Get the global cache instance."""
    global _cache
    if _cache is None:
        _cache = TranslationCache()
    return _cache


def cached_translate(
    text: str,
    translate_func,
    target_language: str,
    model: str = "any",
    min_length: int = 10,
) -> str:
    """
    Translate with caching.
    
    Args:
        text: Text to translate
        translate_func: Function to call if not cached
        target_language: Target language
        model: Model name for cache key
        min_length: Minimum text length to cache (short texts not worth caching)
    
    Returns:
        Translated text
    """
    # Don't cache very short texts
    if len(text.strip()) < min_length:
        return translate_func(text)
    
    cache = get_cache()
    
    # Check cache
    cached = cache.get(text, target_language, model)
    if cached is not None:
        return cached
    
    # Translate and cache
    translated = translate_func(text)
    cache.put(text, translated, target_language, model)
    
    return translated


def batch_cached_translate(
    texts: List[str],
    translate_func,
    target_language: str,
    model: str = "any",
) -> Tuple[List[str], int, int]:
    """
    Translate multiple texts with caching.
    
    Returns (translated_texts, cache_hits, cache_misses)
    """
    cache = get_cache()
    results = []
    hits = 0
    misses = 0
    
    # First pass: check cache
    to_translate = []  # (index, text) pairs
    
    for i, text in enumerate(texts):
        cached = cache.get(text, target_language, model)
        if cached is not None:
            results.append((i, cached))
            hits += 1
        else:
            to_translate.append((i, text))
            misses += 1
    
    # Translate uncached texts
    for i, text in to_translate:
        translated = translate_func(text)
        cache.put(text, translated, target_language, model)
        results.append((i, translated))
    
    # Sort by original index
    results.sort(key=lambda x: x[0])
    
    return [r[1] for r in results], hits, misses


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import tempfile
    
    print("=== Translation Cache Test ===\n")
    
    # Create temp cache
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = TranslationCache(cache_dir=Path(tmpdir))
        
        # Test basic operations
        text = "The black hole has an event horizon."
        translated = "Das schwarze Loch hat einen Ereignishorizont."
        
        # Put
        cache.put(text, translated, "German", "qwen2.5:7b")
        print(f"Stored: {text[:30]}...")
        
        # Get (hit)
        result = cache.get(text, "German", "qwen2.5:7b")
        print(f"Retrieved: {result[:30]}...")
        assert result == translated
        
        # Get (miss - different language)
        result = cache.get(text, "French", "qwen2.5:7b")
        assert result is None
        print("Miss for different language: OK")
        
        # Stats
        stats = cache.get_stats()
        print(f"\nStats:")
        print(f"  Entries: {stats.total_entries}")
        print(f"  Hits: {stats.hits}")
        print(f"  Misses: {stats.misses}")
        print(f"  Hit rate: {stats.hit_rate:.1f}%")
        print(f"  Size: {stats.size_mb:.2f}MB")
        
        # Test cached_translate wrapper
        print("\n### Wrapper Test")
        
        def mock_translate(t):
            print(f"  [API call] Translating: {t[:20]}...")
            return f"TRANSLATED: {t}"
        
        # First call - cache miss
        r1 = cached_translate("Test sentence", mock_translate, "German")
        # Second call - cache hit (no API call)
        r2 = cached_translate("Test sentence", mock_translate, "German")
        
        assert r1 == r2
        print("  Cached result matches: OK")
        
        # Export/import
        export_path = Path(tmpdir) / "export.json"
        cache.export_to_json(str(export_path))
        print(f"\nExported to {export_path}")
        
        cache.clear()
        print("Cache cleared")
        
        imported = cache.import_from_json(str(export_path))
        print(f"Imported {imported} entries")
    
    print("\n✅ Translation Cache ready")
