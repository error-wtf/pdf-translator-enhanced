"""
Logging Configuration for PDF-Translator

Creates detailed log files for debugging.

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
import logging
import os
from datetime import datetime
from pathlib import Path

# Log directory
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Current log file with timestamp
LOG_FILE = LOG_DIR / f"pdf_translator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def setup_logging(level=logging.DEBUG):
    """
    Setup comprehensive logging to both console and file.
    
    Log levels:
    - DEBUG: Detailed information for debugging
    - INFO: General operational messages
    - WARNING: Something unexpected but not critical
    - ERROR: Something failed
    """
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-30s | %(funcName)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(levelname)-8s | %(name)-20s | %(message)s'
    )
    
    # File handler - captures everything
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler - INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Add our handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Set levels for specific loggers
    logging.getLogger("pdf_translator").setLevel(logging.DEBUG)
    logging.getLogger("pdf_translator.marker").setLevel(logging.DEBUG)
    logging.getLogger("pdf_translator.gradio").setLevel(logging.DEBUG)
    logging.getLogger("pdf_translator.ollama").setLevel(logging.DEBUG)
    logging.getLogger("pdf_translator.latex_build").setLevel(logging.DEBUG)
    
    # Reduce noise from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("gradio").setLevel(logging.WARNING)
    logging.getLogger("pdfminer").setLevel(logging.WARNING)
    logging.getLogger("pdfplumber").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("python_multipart").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    # Log startup info
    logger = logging.getLogger("pdf_translator.logging")
    logger.info("=" * 60)
    logger.info("PDF-Translator Logging Started")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("=" * 60)
    
    return LOG_FILE


def get_latest_log() -> Path:
    """Returns the path to the most recent log file."""
    logs = sorted(LOG_DIR.glob("pdf_translator_*.log"), reverse=True)
    return logs[0] if logs else None


def analyze_log(log_path: Path = None) -> dict:
    """
    Analyze a log file and return summary statistics.
    
    Returns dict with:
    - errors: list of error messages
    - warnings: list of warning messages
    - timeline: list of (timestamp, event) tuples
    - duration: total processing time
    """
    if log_path is None:
        log_path = get_latest_log()
    
    if not log_path or not log_path.exists():
        return {"error": "No log file found"}
    
    errors = []
    warnings = []
    timeline = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split(' | ')
            if len(parts) >= 4:
                timestamp = parts[0].strip()
                level = parts[1].strip()
                module = parts[2].strip()
                message = ' | '.join(parts[3:]).strip()
                
                if level == "ERROR":
                    errors.append({"time": timestamp, "module": module, "message": message})
                elif level == "WARNING":
                    warnings.append({"time": timestamp, "module": module, "message": message})
                
                # Track key events
                if any(kw in message.lower() for kw in ['start', 'complete', 'failed', 'success', 'error']):
                    timeline.append({"time": timestamp, "event": message[:100]})
    
    return {
        "log_file": str(log_path),
        "errors": errors,
        "warnings": warnings,
        "timeline": timeline,
        "error_count": len(errors),
        "warning_count": len(warnings),
    }


def print_log_analysis(log_path: Path = None):
    """Print a formatted analysis of the log file."""
    analysis = analyze_log(log_path)
    
    print("\n" + "=" * 60)
    print("LOG ANALYSIS")
    print("=" * 60)
    print(f"Log file: {analysis.get('log_file', 'N/A')}")
    print(f"Errors: {analysis.get('error_count', 0)}")
    print(f"Warnings: {analysis.get('warning_count', 0)}")
    
    if analysis.get('errors'):
        print("\n--- ERRORS ---")
        for err in analysis['errors']:
            print(f"  [{err['time']}] {err['module']}")
            print(f"    {err['message'][:200]}")
    
    if analysis.get('warnings'):
        print("\n--- WARNINGS ---")
        for warn in analysis['warnings'][:10]:  # Limit to 10
            print(f"  [{warn['time']}] {warn['message'][:100]}")
    
    if analysis.get('timeline'):
        print("\n--- TIMELINE ---")
        for event in analysis['timeline'][:20]:  # Limit to 20
            print(f"  [{event['time']}] {event['event']}")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test logging setup
    setup_logging()
    logger = logging.getLogger("pdf_translator.test")
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    print(f"\nLog written to: {LOG_FILE}")
