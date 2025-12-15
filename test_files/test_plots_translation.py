#!/usr/bin/env python3
"""Test translation of PDF with real plots."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from unified_translator import translate_pdf_unified

def test():
    input_pdf = Path(__file__).parent.parent / "test_with_plots.pdf"
    output_dir = Path(__file__).parent / "output_plots"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Input: {input_pdf}")
    print(f"Model: gpt-oss:20b-cloud")
    
    def progress(current, total, status):
        safe_status = status.encode('ascii', 'replace').decode('ascii')
        print(f"  [{current}/{total}] {safe_status}")
    
    print("\nTranslating PDF with plots to German...")
    print("-" * 50)
    
    output_path, status = translate_pdf_unified(
        str(input_pdf),
        str(output_dir),
        model="gpt-oss:20b-cloud",
        target_language="German",
        progress_callback=progress,
        use_openai=False,
        openai_api_key=None
    )
    
    print("-" * 50)
    safe_status = status.encode('ascii', 'replace').decode('ascii') if status else "None"
    print(f"Status: {safe_status}")
    
    if output_path and Path(output_path).exists():
        size = Path(output_path).stat().st_size
        print(f"SUCCESS: {output_path} ({size} bytes)")
        os.startfile(output_path)
    else:
        print("FAILED")

if __name__ == "__main__":
    test()
