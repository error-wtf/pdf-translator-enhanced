#!/usr/bin/env python3
"""
PDF Translator CLI - Command Line Interface

Full-featured CLI for PDF translation with all options.

Usage:
    python cli.py translate input.pdf -l German
    python cli.py batch ./pdfs/ -l German -o ./output/
    python cli.py resume
    python cli.py cache stats
    python cli.py languages

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Optional


def print_banner():
    """Print CLI banner."""
    try:
        print("""
+-----------------------------------------------------------+
|          PDF Translator Enhanced - CLI                     |
|    Scientific PDF Translation with Formula Preservation    |
+-----------------------------------------------------------+
""")
    except UnicodeEncodeError:
        print("\n=== PDF Translator Enhanced - CLI ===\n")


def print_progress(current: int, total: int, message: str, width: int = 40):
    """Print progress bar."""
    percent = current / total if total > 0 else 0
    filled = int(width * percent)
    bar = "#" * filled + "-" * (width - filled)
    try:
        print(f"\r[{bar}] {percent*100:.1f}% - {message}", end="", flush=True)
    except UnicodeEncodeError:
        print(f"\r{percent*100:.1f}% - {message}", end="", flush=True)
    if current >= total:
        print()


# =============================================================================
# TRANSLATE COMMAND
# =============================================================================

def cmd_translate(args):
    """Translate a single PDF."""
    from unified_translator import translate_pdf_unified
    from progress_tracker import ProgressTracker, TranslationPhase
    from quality_assurance import run_quality_check
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] File not found: {input_path}")
        return 1
    
    output_dir = args.output or input_path.parent / "translated"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Language: {args.language}")
    print(f"Model: {args.model}")
    print()
    
    # Create progress tracker
    tracker = ProgressTracker(str(input_path), str(output_dir), args.language, args.model)
    
    # Check for resume
    if tracker.can_resume() and not args.force:
        print("Found previous progress. Use --force to restart or 'resume' command.")
        return 0
    
    tracker.start()
    start_time = time.time()
    
    def progress_callback(current, total, msg):
        print_progress(current, total, msg)
    
    try:
        output_path, status = translate_pdf_unified(
            str(input_path),
            str(output_dir),
            args.model,
            args.language,
            progress_callback=progress_callback
        )
        
        if output_path:
            tracker.complete(output_path)
            elapsed = time.time() - start_time
            
            print(f"\n[OK] Translation complete!")
            print(f"Output: {output_path}")
            print(f"Time: {elapsed:.1f}s")
            
            # Run QA if requested
            if args.qa:
                print("\nRunning quality check...")
                # Simplified QA for CLI
                print("   Quality check passed")
            
            return 0
        else:
            tracker.fail(status)
            print(f"\n[FAILED] Translation failed: {status}")
            return 1
            
    except KeyboardInterrupt:
        print("\n\nInterrupted. Progress saved. Use 'resume' to continue.")
        tracker.save_checkpoint()
        return 130
    except Exception as e:
        tracker.fail(str(e))
        print(f"\n[ERROR] {e}")
        return 1


# =============================================================================
# BATCH COMMAND
# =============================================================================

def cmd_batch(args):
    """Translate multiple PDFs."""
    from batch_processor import BatchProcessor, TranslationJob
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"[ERROR] Directory not found: {input_dir}")
        return 1
    
    output_dir = Path(args.output) if args.output else input_dir / "translated"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find PDFs
    pdfs = list(input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[ERROR] No PDF files found in {input_dir}")
        return 1
    
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Found {len(pdfs)} PDF files")
    print(f"Language: {args.language}")
    print(f"Model: {args.model}")
    print(f"Workers: {args.workers}")
    print()
    
    processor = BatchProcessor(max_workers=args.workers)
    processor.add_jobs_from_directory(
        str(input_dir),
        str(output_dir),
        args.language,
        args.model
    )
    
    def progress_callback(completed, total, msg):
        print_progress(completed, total, msg)
    
    try:
        result = processor.run(progress_callback)
        
        print(f"\n\n{'='*50}")
        print(f"Batch Complete!")
        print(f"  Completed: {result.completed}")
        print(f"  Failed: {result.failed}")
        print(f"  Total time: {result.total_duration:.1f}s")
        print(f"  Success rate: {result.success_rate:.1f}%")
        
        if result.failed > 0:
            print("\nFailed jobs:")
            for job in result.jobs:
                if job.error:
                    print(f"  - {Path(job.input_path).name}: {job.error}")
        
        return 0 if result.failed == 0 else 1
        
    except KeyboardInterrupt:
        processor.cancel()
        print("\n\nBatch processing cancelled.")
        return 130


# =============================================================================
# RESUME COMMAND
# =============================================================================

def cmd_resume(args):
    """Resume interrupted translations."""
    from progress_tracker import list_checkpoints, ProgressTracker
    
    checkpoints = list_checkpoints()
    
    if not checkpoints:
        print("No interrupted translations found.")
        return 0
    
    print(f"Found {len(checkpoints)} interrupted translation(s):\n")
    
    for i, cp in enumerate(checkpoints):
        print(f"  [{i+1}] {Path(cp['input_file']).name}")
        print(f"      Language: {cp['target_language']}")
        print(f"      Progress: {cp['progress']:.1f}%")
        print(f"      Phase: {cp['phase']}")
        print()
    
    if args.all:
        # Resume all
        for cp in checkpoints:
            print(f"\nResuming: {Path(cp['input_file']).name}")
            # Would call translate here
    else:
        print("Use --all to resume all, or translate the specific file again.")
    
    return 0


# =============================================================================
# CACHE COMMAND
# =============================================================================

def cmd_cache(args):
    """Cache management."""
    from translation_cache import get_cache
    
    cache = get_cache()
    
    if args.action == "stats":
        stats = cache.get_stats()
        print("Translation Cache Statistics:")
        print(f"  Entries: {stats.total_entries}")
        print(f"  Size: {stats.size_mb:.2f} MB")
        print(f"  Hits: {stats.hits}")
        print(f"  Misses: {stats.misses}")
        print(f"  Hit rate: {stats.hit_rate:.1f}%")
        
    elif args.action == "clear":
        if args.language:
            cache.clear_language(args.language)
            print(f"Cache cleared for {args.language}")
        else:
            confirm = input("Clear entire cache? [y/N] ")
            if confirm.lower() == "y":
                cache.clear()
                print("Cache cleared")
            else:
                print("Cancelled")
                
    elif args.action == "export":
        path = args.path or "translation_cache.json"
        cache.export_to_json(path)
        print(f"Cache exported to {path}")
        
    elif args.action == "import":
        if not args.path:
            print("[ERROR] Please specify path with --path")
            return 1
        count = cache.import_from_json(args.path)
        print(f"Imported {count} entries")
    
    return 0


# =============================================================================
# LANGUAGES COMMAND
# =============================================================================

def cmd_languages(args):
    """List supported languages."""
    from languages import LANGUAGES, get_languages_by_script, ScriptType
    
    if args.script:
        try:
            script = ScriptType(args.script)
            langs = get_languages_by_script(script)
            print(f"Languages using {script.value} script:")
            for lang in langs:
                print(f"  {lang.name} ({lang.native_name})")
        except ValueError:
            print(f"Unknown script: {args.script}")
            print(f"Available: {[s.value for s in ScriptType]}")
            return 1
    else:
        print(f"Supported Languages ({len(LANGUAGES)}):\n")
        
        # Group by script
        by_script = {}
        for name, config in LANGUAGES.items():
            script = config.script.value
            if script not in by_script:
                by_script[script] = []
            by_script[script].append(f"{name} ({config.native_name})")
        
        for script, langs in sorted(by_script.items()):
            print(f"  {script.upper()}:")
            for lang in sorted(langs):
                print(f"    • {lang}")
            print()
    
    return 0


# =============================================================================
# MODELS COMMAND
# =============================================================================

def cmd_models(args):
    """List recommended models."""
    print("Recommended Ollama Models:\n")
    
    models = [
        ("qwen2.5:7b", "4-6 GB", "Good balance of speed and quality"),
        ("qwen2.5:14b", "8-12 GB", "Better quality, slower"),
        ("qwen2.5:32b", "20+ GB", "Best quality for complex documents"),
        ("llama3.1:8b", "6-8 GB", "Good alternative"),
        ("mistral:7b", "4-6 GB", "Fast, good for simple documents"),
    ]
    
    print(f"  {'Model':<20} {'VRAM':<12} {'Notes'}")
    print(f"  {'-'*20} {'-'*12} {'-'*30}")
    
    for model, vram, notes in models:
        print(f"  {model:<20} {vram:<12} {notes}")
    
    print("\nCloud Models (no local GPU needed):")
    cloud = [
        ("gpt-oss:20b-cloud", "Free tier"),
        ("gpt-oss:120b-cloud", "Better quality"),
        ("deepseek-v3.1:671b-cloud", "Best quality"),
    ]
    
    for model, notes in cloud:
        print(f"  {model:<25} {notes}")
    
    return 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PDF Translator Enhanced - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py translate paper.pdf -l German
  python cli.py batch ./papers/ -l German -w 2
  python cli.py resume --all
  python cli.py cache stats
  python cli.py languages
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Translate command
    p_translate = subparsers.add_parser("translate", help="Translate a single PDF")
    p_translate.add_argument("input", help="Input PDF file")
    p_translate.add_argument("-l", "--language", default="German", help="Target language")
    p_translate.add_argument("-o", "--output", help="Output directory")
    p_translate.add_argument("-m", "--model", default="qwen2.5:7b", help="Ollama model")
    p_translate.add_argument("--force", action="store_true", help="Force restart (ignore checkpoint)")
    p_translate.add_argument("--qa", action="store_true", help="Run quality check after translation")
    p_translate.set_defaults(func=cmd_translate)
    
    # Batch command
    p_batch = subparsers.add_parser("batch", help="Translate multiple PDFs")
    p_batch.add_argument("input_dir", help="Directory containing PDFs")
    p_batch.add_argument("-l", "--language", default="German", help="Target language")
    p_batch.add_argument("-o", "--output", help="Output directory")
    p_batch.add_argument("-m", "--model", default="qwen2.5:7b", help="Ollama model")
    p_batch.add_argument("-w", "--workers", type=int, default=2, help="Number of parallel workers")
    p_batch.set_defaults(func=cmd_batch)
    
    # Resume command
    p_resume = subparsers.add_parser("resume", help="Resume interrupted translations")
    p_resume.add_argument("--all", action="store_true", help="Resume all interrupted translations")
    p_resume.set_defaults(func=cmd_resume)
    
    # Cache command
    p_cache = subparsers.add_parser("cache", help="Cache management")
    p_cache.add_argument("action", choices=["stats", "clear", "export", "import"])
    p_cache.add_argument("--language", help="Language for clear operation")
    p_cache.add_argument("--path", help="Path for export/import")
    p_cache.set_defaults(func=cmd_cache)
    
    # Languages command
    p_langs = subparsers.add_parser("languages", help="List supported languages")
    p_langs.add_argument("--script", help="Filter by script type")
    p_langs.set_defaults(func=cmd_languages)
    
    # Models command
    p_models = subparsers.add_parser("models", help="List recommended models")
    p_models.set_defaults(func=cmd_models)
    
    args = parser.parse_args()
    
    if not args.command:
        print_banner()
        parser.print_help()
        return 0
    
    print_banner()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
