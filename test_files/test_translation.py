#!/usr/bin/env python3
"""
Test the unified translator with the test PDF.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from unified_translator import translate_pdf_unified

def test_translation():
    input_pdf = Path(__file__).parent.parent / "test_3pages.pdf"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Input: {input_pdf}")
    print(f"Output dir: {output_dir}")
    print(f"Input exists: {input_pdf.exists()}")
    
    if not input_pdf.exists():
        print("ERROR: Input PDF not found!")
        return
    
    # Test with a simple progress callback
    def progress(current, total, status):
        print(f"  [{current}/{total}] {status}")
    
    print("\nStarting translation to German...")
    print("-" * 50)
    
    output_path, status = translate_pdf_unified(
        str(input_pdf),
        str(output_dir),
        model="",  # No model - just test extraction
        target_language="German",
        progress_callback=progress,
        use_openai=False,
        openai_api_key=None
    )
    
    print("-" * 50)
    print(f"Status: {status}")
    print(f"Output: {output_path}")
    
    if output_path and Path(output_path).exists():
        print(f"SUCCESS: Output file created ({Path(output_path).stat().st_size} bytes)")
    else:
        print("FAILED: No output file created")

if __name__ == "__main__":
    test_translation()
