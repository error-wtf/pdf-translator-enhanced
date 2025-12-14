"""
Enhanced Progress Display

Provides detailed progress information including:
- ETA (Estimated Time of Arrival)
- Pages per minute
- Current speed
- Remaining time

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, List
from datetime import datetime, timedelta

logger = logging.getLogger("pdf_translator.progress")


@dataclass
class ProgressStats:
    """Statistics for progress tracking."""
    total_pages: int = 0
    completed_pages: int = 0
    current_page: int = 0
    start_time: float = field(default_factory=time.time)
    page_times: List[float] = field(default_factory=list)
    status: str = "Initializing"
    last_update: float = field(default_factory=time.time)
    
    @property
    def elapsed_seconds(self) -> float:
        """Total elapsed time in seconds."""
        return time.time() - self.start_time
    
    @property
    def elapsed_formatted(self) -> str:
        """Elapsed time as formatted string."""
        return format_duration(self.elapsed_seconds)
    
    @property
    def pages_per_minute(self) -> float:
        """Average pages processed per minute."""
        if self.completed_pages == 0 or self.elapsed_seconds < 1:
            return 0.0
        return (self.completed_pages / self.elapsed_seconds) * 60
    
    @property
    def avg_page_time(self) -> float:
        """Average time per page in seconds."""
        if not self.page_times:
            return 0.0
        return sum(self.page_times) / len(self.page_times)
    
    @property
    def remaining_pages(self) -> int:
        """Number of pages remaining."""
        return max(0, self.total_pages - self.completed_pages)
    
    @property
    def eta_seconds(self) -> float:
        """Estimated seconds until completion."""
        if self.avg_page_time == 0:
            return 0.0
        return self.remaining_pages * self.avg_page_time
    
    @property
    def eta_formatted(self) -> str:
        """ETA as formatted string."""
        if self.eta_seconds == 0:
            return "Calculating..."
        return format_duration(self.eta_seconds)
    
    @property
    def eta_datetime(self) -> Optional[datetime]:
        """Expected completion datetime."""
        if self.eta_seconds == 0:
            return None
        return datetime.now() + timedelta(seconds=self.eta_seconds)
    
    @property
    def progress_percent(self) -> float:
        """Progress as percentage (0-100)."""
        if self.total_pages == 0:
            return 0.0
        return (self.completed_pages / self.total_pages) * 100
    
    @property
    def speed_trend(self) -> str:
        """Trend indicator based on recent page times."""
        if len(self.page_times) < 3:
            return "→"  # Not enough data
        
        recent = self.page_times[-3:]
        older = self.page_times[-6:-3] if len(self.page_times) >= 6 else self.page_times[:3]
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        if recent_avg < older_avg * 0.9:
            return "↑"  # Getting faster
        elif recent_avg > older_avg * 1.1:
            return "↓"  # Getting slower
        return "→"  # Stable


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 0:
        return "Unknown"
    
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


class ProgressDisplay:
    """
    Enhanced progress display with ETA calculation.
    
    Example:
        progress = ProgressDisplay(total_pages=10)
        
        for page in range(10):
            progress.start_page(page + 1)
            # ... process page ...
            progress.complete_page()
            
            print(progress.get_status_line())
    """
    
    def __init__(
        self,
        total_pages: int,
        callback: Optional[Callable[[int, int, str], None]] = None
    ):
        """
        Initialize progress display.
        
        Args:
            total_pages: Total number of pages to process
            callback: Optional callback(current, total, message)
        """
        self.stats = ProgressStats(total_pages=total_pages)
        self.callback = callback
        self._page_start_time: Optional[float] = None
    
    def start_page(self, page_num: int, status: str = "Processing"):
        """Mark the start of processing a page."""
        self._page_start_time = time.time()
        self.stats.current_page = page_num
        self.stats.status = status
        self.stats.last_update = time.time()
        
        self._notify()
    
    def complete_page(self):
        """Mark the completion of the current page."""
        if self._page_start_time:
            page_time = time.time() - self._page_start_time
            self.stats.page_times.append(page_time)
            self._page_start_time = None
        
        self.stats.completed_pages += 1
        self.stats.status = "Completed"
        self.stats.last_update = time.time()
        
        self._notify()
    
    def update_status(self, status: str):
        """Update the current status message."""
        self.stats.status = status
        self.stats.last_update = time.time()
        self._notify()
    
    def _notify(self):
        """Notify callback if set."""
        if self.callback:
            self.callback(
                self.stats.completed_pages,
                self.stats.total_pages,
                self.get_status_line()
            )
    
    def get_status_line(self) -> str:
        """Get a single-line status summary."""
        s = self.stats
        
        if s.completed_pages == 0:
            return f"📄 Page {s.current_page}/{s.total_pages}: {s.status}"
        
        return (
            f"📄 {s.completed_pages}/{s.total_pages} "
            f"({s.progress_percent:.0f}%) "
            f"| ⏱️ {s.elapsed_formatted} "
            f"| 🏁 ETA: {s.eta_formatted} "
            f"| ⚡ {s.pages_per_minute:.1f} p/min {s.speed_trend}"
        )
    
    def get_detailed_status(self) -> str:
        """Get detailed multi-line status."""
        s = self.stats
        
        lines = [
            f"{'=' * 50}",
            f"📊 Translation Progress",
            f"{'=' * 50}",
            f"",
            f"📄 Pages: {s.completed_pages}/{s.total_pages} ({s.progress_percent:.1f}%)",
            f"📍 Current: Page {s.current_page} - {s.status}",
            f"",
            f"⏱️  Elapsed: {s.elapsed_formatted}",
            f"🏁 ETA: {s.eta_formatted}",
        ]
        
        if s.eta_datetime:
            lines.append(f"🕐 Completion: {s.eta_datetime.strftime('%H:%M:%S')}")
        
        lines.extend([
            f"",
            f"⚡ Speed: {s.pages_per_minute:.2f} pages/minute {s.speed_trend}",
            f"📈 Avg page time: {s.avg_page_time:.1f}s",
            f"{'=' * 50}",
        ])
        
        return "\n".join(lines)
    
    def get_progress_bar(self, width: int = 40) -> str:
        """Get ASCII progress bar."""
        s = self.stats
        filled = int(width * s.progress_percent / 100)
        empty = width - filled
        
        bar = "█" * filled + "░" * empty
        return f"[{bar}] {s.progress_percent:.1f}%"
    
    def get_gradio_status(self) -> tuple:
        """Get status formatted for Gradio progress bar."""
        s = self.stats
        
        message = (
            f"Page {s.current_page}/{s.total_pages} - {s.status}\n"
            f"ETA: {s.eta_formatted} | Speed: {s.pages_per_minute:.1f} p/min"
        )
        
        return (s.progress_percent / 100, message)


class BatchProgressDisplay:
    """
    Progress display for batch processing multiple PDFs.
    """
    
    def __init__(
        self,
        total_files: int,
        callback: Optional[Callable[[str], None]] = None
    ):
        self.total_files = total_files
        self.completed_files = 0
        self.current_file: Optional[str] = None
        self.current_progress: Optional[ProgressDisplay] = None
        self.file_times: List[float] = []
        self.start_time = time.time()
        self.callback = callback
    
    def start_file(self, filename: str, total_pages: int):
        """Start processing a new file."""
        self.current_file = filename
        self.current_progress = ProgressDisplay(total_pages)
        
        if self.callback:
            self.callback(self.get_batch_status())
    
    def complete_file(self, processing_time: float):
        """Mark current file as complete."""
        self.completed_files += 1
        self.file_times.append(processing_time)
        
        if self.callback:
            self.callback(self.get_batch_status())
    
    @property
    def avg_file_time(self) -> float:
        """Average time per file."""
        if not self.file_times:
            return 0.0
        return sum(self.file_times) / len(self.file_times)
    
    @property
    def batch_eta(self) -> str:
        """ETA for entire batch."""
        remaining = self.total_files - self.completed_files
        if self.avg_file_time == 0 or remaining == 0:
            return "Calculating..."
        return format_duration(remaining * self.avg_file_time)
    
    def get_batch_status(self) -> str:
        """Get batch processing status."""
        elapsed = format_duration(time.time() - self.start_time)
        
        lines = [
            f"📚 Batch: {self.completed_files}/{self.total_files} files",
            f"⏱️  Elapsed: {elapsed} | ETA: {self.batch_eta}",
        ]
        
        if self.current_file and self.current_progress:
            lines.append(f"📄 Current: {self.current_file}")
            lines.append(self.current_progress.get_status_line())
        
        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_gradio_callback(progress_bar) -> Callable[[int, int, str], None]:
    """
    Create a callback function for Gradio progress bar.
    
    Example:
        with gr.Progress() as progress:
            callback = create_gradio_callback(progress)
            display = ProgressDisplay(total_pages, callback)
    """
    def callback(current: int, total: int, message: str):
        if total > 0:
            progress_bar(current / total, desc=message)
    
    return callback


def create_cli_callback() -> Callable[[int, int, str], None]:
    """Create a callback for CLI progress display."""
    def callback(current: int, total: int, message: str):
        bar_width = 30
        filled = int(bar_width * current / max(total, 1))
        bar = "█" * filled + "░" * (bar_width - filled)
        percent = (current / max(total, 1)) * 100
        
        print(f"\r[{bar}] {percent:5.1f}% {message}", end="", flush=True)
        
        if current >= total:
            print()  # New line when complete
    
    return callback
