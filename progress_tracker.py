"""
Progress Tracker - Resumable translation with checkpoints

Large PDFs can take hours to translate. This module enables:
- Automatic checkpoint saving
- Resume from interruption
- Progress persistence across sessions

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import json
import logging
import time
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger("pdf_translator.progress")


# =============================================================================
# CHECKPOINT CONFIGURATION
# =============================================================================

CHECKPOINT_DIR = Path.home() / ".pdf_translator_checkpoints"
CHECKPOINT_INTERVAL = 5  # Save every N blocks


class TranslationPhase(Enum):
    """Phases of translation process."""
    INIT = "init"
    EXTRACTION = "extraction"
    TRANSLATION = "translation"
    LAYOUT = "layout"
    OUTPUT = "output"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class BlockProgress:
    """Progress for a single block."""
    index: int
    original: str
    translated: Optional[str] = None
    is_complete: bool = False
    error: Optional[str] = None


@dataclass
class PageProgress:
    """Progress for a single page."""
    page_num: int
    total_blocks: int
    completed_blocks: int = 0
    blocks: List[BlockProgress] = field(default_factory=list)
    
    @property
    def is_complete(self) -> bool:
        return self.completed_blocks >= self.total_blocks
    
    @property
    def progress_percent(self) -> float:
        if self.total_blocks == 0:
            return 100.0
        return self.completed_blocks / self.total_blocks * 100


@dataclass
class TranslationProgress:
    """Complete progress state for a translation job."""
    # Job identification
    job_id: str
    input_file: str
    input_hash: str  # Hash of input file for verification
    output_dir: str
    target_language: str
    model: str
    
    # Progress
    phase: TranslationPhase = TranslationPhase.INIT
    total_pages: int = 0
    completed_pages: int = 0
    current_page: int = 0
    pages: Dict[int, PageProgress] = field(default_factory=dict)
    
    # Timing
    started_at: Optional[float] = None
    updated_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Extracted content (cached for resume)
    extracted_text: Optional[str] = None
    
    @property
    def is_complete(self) -> bool:
        return self.phase == TranslationPhase.COMPLETE
    
    @property
    def is_failed(self) -> bool:
        return self.phase == TranslationPhase.FAILED
    
    @property
    def can_resume(self) -> bool:
        return self.phase not in [TranslationPhase.COMPLETE, TranslationPhase.FAILED]
    
    @property
    def overall_progress(self) -> float:
        if self.total_pages == 0:
            return 0.0
        
        # Weight by phase
        phase_weights = {
            TranslationPhase.INIT: 0,
            TranslationPhase.EXTRACTION: 10,
            TranslationPhase.TRANSLATION: 80,
            TranslationPhase.LAYOUT: 95,
            TranslationPhase.OUTPUT: 98,
            TranslationPhase.COMPLETE: 100,
        }
        
        base = phase_weights.get(self.phase, 0)
        
        if self.phase == TranslationPhase.TRANSLATION:
            # Calculate translation progress
            page_progress = self.completed_pages / self.total_pages if self.total_pages > 0 else 0
            return 10 + page_progress * 70  # 10-80%
        
        return base
    
    @property
    def elapsed_time(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.completed_at or time.time()
        return end - self.started_at
    
    @property
    def eta_seconds(self) -> Optional[float]:
        if self.overall_progress <= 0 or self.elapsed_time <= 0:
            return None
        rate = self.overall_progress / self.elapsed_time
        remaining = 100 - self.overall_progress
        return remaining / rate if rate > 0 else None
    
    def to_dict(self) -> Dict:
        d = {
            "job_id": self.job_id,
            "input_file": self.input_file,
            "input_hash": self.input_hash,
            "output_dir": self.output_dir,
            "target_language": self.target_language,
            "model": self.model,
            "phase": self.phase.value,
            "total_pages": self.total_pages,
            "completed_pages": self.completed_pages,
            "current_page": self.current_page,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "extracted_text": self.extracted_text,
            "pages": {},
        }
        
        for page_num, page in self.pages.items():
            d["pages"][str(page_num)] = {
                "page_num": page.page_num,
                "total_blocks": page.total_blocks,
                "completed_blocks": page.completed_blocks,
                "blocks": [
                    {
                        "index": b.index,
                        "original": b.original[:500],  # Truncate for storage
                        "translated": b.translated[:500] if b.translated else None,
                        "is_complete": b.is_complete,
                        "error": b.error,
                    }
                    for b in page.blocks
                ],
            }
        
        return d
    
    @classmethod
    def from_dict(cls, d: Dict) -> "TranslationProgress":
        progress = cls(
            job_id=d["job_id"],
            input_file=d["input_file"],
            input_hash=d["input_hash"],
            output_dir=d["output_dir"],
            target_language=d["target_language"],
            model=d["model"],
        )
        
        progress.phase = TranslationPhase(d["phase"])
        progress.total_pages = d["total_pages"]
        progress.completed_pages = d["completed_pages"]
        progress.current_page = d["current_page"]
        progress.started_at = d["started_at"]
        progress.updated_at = d["updated_at"]
        progress.completed_at = d["completed_at"]
        progress.extracted_text = d.get("extracted_text")
        
        for page_num_str, page_data in d.get("pages", {}).items():
            page = PageProgress(
                page_num=page_data["page_num"],
                total_blocks=page_data["total_blocks"],
                completed_blocks=page_data["completed_blocks"],
            )
            for block_data in page_data.get("blocks", []):
                page.blocks.append(BlockProgress(
                    index=block_data["index"],
                    original=block_data["original"],
                    translated=block_data.get("translated"),
                    is_complete=block_data["is_complete"],
                    error=block_data.get("error"),
                ))
            progress.pages[int(page_num_str)] = page
        
        return progress


# =============================================================================
# PROGRESS TRACKER
# =============================================================================

class ProgressTracker:
    """
    Track and persist translation progress.
    
    Usage:
        tracker = ProgressTracker("input.pdf", "output/", "German")
        
        # Check for existing progress
        if tracker.can_resume():
            tracker.resume()
        else:
            tracker.start()
        
        # Update progress
        tracker.set_phase(TranslationPhase.TRANSLATION)
        tracker.complete_block(page=1, block=0, translated="...")
        
        # Save checkpoint
        tracker.save_checkpoint()
    """
    
    def __init__(
        self,
        input_file: str,
        output_dir: str,
        target_language: str,
        model: str = "qwen2.5:7b",
        checkpoint_dir: Optional[Path] = None,
    ):
        self.checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate job ID from input file
        input_path = Path(input_file)
        self.job_id = f"{input_path.stem}_{target_language}_{hashlib.md5(str(input_path).encode()).hexdigest()[:8]}"
        
        self.checkpoint_path = self.checkpoint_dir / f"{self.job_id}.json"
        
        # Calculate file hash for verification
        file_hash = ""
        if input_path.exists():
            with open(input_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()[:16]
        
        self.progress = TranslationProgress(
            job_id=self.job_id,
            input_file=str(input_file),
            input_hash=file_hash,
            output_dir=output_dir,
            target_language=target_language,
            model=model,
        )
        
        self._block_count = 0
        self._callbacks: List[Callable] = []
    
    def add_callback(self, callback: Callable[[TranslationProgress], None]):
        """Add progress callback."""
        self._callbacks.append(callback)
    
    def _notify_callbacks(self):
        """Notify all callbacks of progress update."""
        for callback in self._callbacks:
            try:
                callback(self.progress)
            except Exception as e:
                logger.warning(f"Callback error: {e}")
    
    def has_checkpoint(self) -> bool:
        """Check if a checkpoint exists for this job."""
        return self.checkpoint_path.exists()
    
    def can_resume(self) -> bool:
        """Check if translation can be resumed."""
        if not self.has_checkpoint():
            return False
        
        try:
            self.load_checkpoint()
            
            # Verify file hash matches
            input_path = Path(self.progress.input_file)
            if input_path.exists():
                with open(input_path, "rb") as f:
                    current_hash = hashlib.md5(f.read()).hexdigest()[:16]
                if current_hash != self.progress.input_hash:
                    logger.warning("Input file has changed - cannot resume")
                    return False
            
            return self.progress.can_resume
            
        except Exception as e:
            logger.warning(f"Cannot resume: {e}")
            return False
    
    def start(self):
        """Start a new translation job."""
        self.progress.started_at = time.time()
        self.progress.phase = TranslationPhase.INIT
        self.save_checkpoint()
        logger.info(f"Started job: {self.job_id}")
    
    def resume(self) -> TranslationProgress:
        """Resume from checkpoint."""
        self.load_checkpoint()
        logger.info(f"Resumed job: {self.job_id} at {self.progress.phase.value}")
        return self.progress
    
    def load_checkpoint(self):
        """Load progress from checkpoint file."""
        if self.checkpoint_path.exists():
            data = json.loads(self.checkpoint_path.read_text())
            self.progress = TranslationProgress.from_dict(data)
    
    def save_checkpoint(self):
        """Save current progress to checkpoint file."""
        self.progress.updated_at = time.time()
        self.checkpoint_path.write_text(json.dumps(self.progress.to_dict(), indent=2))
        logger.debug(f"Checkpoint saved: {self.progress.overall_progress:.1f}%")
    
    def set_phase(self, phase: TranslationPhase):
        """Update translation phase."""
        self.progress.phase = phase
        self._notify_callbacks()
        self.save_checkpoint()
    
    def set_total_pages(self, total: int):
        """Set total page count."""
        self.progress.total_pages = total
        self.save_checkpoint()
    
    def init_page(self, page_num: int, total_blocks: int):
        """Initialize progress for a page."""
        self.progress.pages[page_num] = PageProgress(
            page_num=page_num,
            total_blocks=total_blocks,
            blocks=[
                BlockProgress(index=i, original="")
                for i in range(total_blocks)
            ]
        )
        self.progress.current_page = page_num
    
    def set_block_original(self, page_num: int, block_idx: int, original: str):
        """Set original text for a block."""
        if page_num in self.progress.pages:
            page = self.progress.pages[page_num]
            if block_idx < len(page.blocks):
                page.blocks[block_idx].original = original
    
    def complete_block(
        self,
        page_num: int,
        block_idx: int,
        translated: str,
        error: Optional[str] = None
    ):
        """Mark a block as complete."""
        if page_num not in self.progress.pages:
            return
        
        page = self.progress.pages[page_num]
        if block_idx >= len(page.blocks):
            return
        
        block = page.blocks[block_idx]
        block.translated = translated
        block.is_complete = True
        block.error = error
        
        page.completed_blocks = sum(1 for b in page.blocks if b.is_complete)
        
        self._block_count += 1
        self._notify_callbacks()
        
        # Save checkpoint periodically
        if self._block_count % CHECKPOINT_INTERVAL == 0:
            self.save_checkpoint()
    
    def complete_page(self, page_num: int):
        """Mark a page as complete."""
        if page_num in self.progress.pages:
            page = self.progress.pages[page_num]
            page.completed_blocks = page.total_blocks
            self.progress.completed_pages += 1
            self._notify_callbacks()
            self.save_checkpoint()
    
    def complete(self, output_path: Optional[str] = None):
        """Mark translation as complete."""
        self.progress.phase = TranslationPhase.COMPLETE
        self.progress.completed_at = time.time()
        self.save_checkpoint()
        logger.info(f"Job complete: {self.job_id} ({self.progress.elapsed_time:.1f}s)")
    
    def fail(self, error: str):
        """Mark translation as failed."""
        self.progress.phase = TranslationPhase.FAILED
        self.save_checkpoint()
        logger.error(f"Job failed: {self.job_id} - {error}")
    
    def get_resume_point(self) -> Dict:
        """Get the point to resume from."""
        return {
            "phase": self.progress.phase,
            "page": self.progress.current_page,
            "completed_pages": list(
                p for p, pg in self.progress.pages.items() if pg.is_complete
            ),
            "partial_pages": {
                p: [b.index for b in pg.blocks if not b.is_complete]
                for p, pg in self.progress.pages.items()
                if not pg.is_complete and pg.completed_blocks > 0
            },
        }
    
    def delete_checkpoint(self):
        """Delete checkpoint file."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.info(f"Checkpoint deleted: {self.job_id}")
    
    def get_status_string(self) -> str:
        """Get human-readable status string."""
        p = self.progress
        eta = p.eta_seconds
        eta_str = f"{eta/60:.0f}m" if eta else "?"
        
        return (
            f"[{p.phase.value}] "
            f"{p.overall_progress:.1f}% "
            f"({p.completed_pages}/{p.total_pages} pages) "
            f"ETA: {eta_str}"
        )


# =============================================================================
# LIST AND MANAGE CHECKPOINTS
# =============================================================================

def list_checkpoints(checkpoint_dir: Optional[Path] = None) -> List[Dict]:
    """List all available checkpoints."""
    dir_path = checkpoint_dir or CHECKPOINT_DIR
    if not dir_path.exists():
        return []
    
    checkpoints = []
    for path in dir_path.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            checkpoints.append({
                "job_id": data["job_id"],
                "input_file": data["input_file"],
                "target_language": data["target_language"],
                "phase": data["phase"],
                "progress": data.get("completed_pages", 0) / max(data.get("total_pages", 1), 1) * 100,
                "updated_at": data.get("updated_at"),
                "path": str(path),
            })
        except Exception as e:
            logger.warning(f"Could not read checkpoint {path}: {e}")
    
    return checkpoints


def cleanup_old_checkpoints(max_age_days: int = 30, checkpoint_dir: Optional[Path] = None):
    """Delete checkpoints older than max_age_days."""
    dir_path = checkpoint_dir or CHECKPOINT_DIR
    if not dir_path.exists():
        return
    
    cutoff = time.time() - (max_age_days * 24 * 3600)
    deleted = 0
    
    for path in dir_path.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            updated = data.get("updated_at", 0)
            if updated < cutoff:
                path.unlink()
                deleted += 1
        except Exception:
            pass
    
    if deleted:
        logger.info(f"Deleted {deleted} old checkpoints")


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import tempfile
    
    print("=== Progress Tracker Test ===\n")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock input file
        input_file = Path(tmpdir) / "test.pdf"
        input_file.write_bytes(b"mock pdf content")
        
        # Create tracker
        tracker = ProgressTracker(
            str(input_file),
            tmpdir,
            "German",
            checkpoint_dir=Path(tmpdir) / "checkpoints"
        )
        
        # Start job
        tracker.start()
        print(f"Job ID: {tracker.job_id}")
        
        # Simulate translation
        tracker.set_phase(TranslationPhase.EXTRACTION)
        tracker.set_total_pages(3)
        
        tracker.set_phase(TranslationPhase.TRANSLATION)
        
        for page in range(3):
            tracker.init_page(page, total_blocks=5)
            for block in range(5):
                tracker.set_block_original(page, block, f"Original text {page}-{block}")
                tracker.complete_block(page, block, f"Translated text {page}-{block}")
            tracker.complete_page(page)
            print(f"  {tracker.get_status_string()}")
        
        tracker.complete()
        
        print(f"\nFinal: {tracker.get_status_string()}")
        print(f"Elapsed: {tracker.progress.elapsed_time:.1f}s")
        
        # Test resume
        print("\n### Resume Test")
        
        # Create new tracker with same params
        tracker2 = ProgressTracker(
            str(input_file),
            tmpdir,
            "German",
            checkpoint_dir=Path(tmpdir) / "checkpoints"
        )
        
        if tracker2.has_checkpoint():
            print("Checkpoint found")
            progress = tracker2.resume()
            print(f"Resumed: {progress.phase.value}, {progress.completed_pages} pages done")
        
        # List checkpoints
        print("\n### Checkpoints")
        for cp in list_checkpoints(Path(tmpdir) / "checkpoints"):
            print(f"  {cp['job_id']}: {cp['progress']:.0f}% ({cp['phase']})")
    
    print("\n✅ Progress Tracker ready")
