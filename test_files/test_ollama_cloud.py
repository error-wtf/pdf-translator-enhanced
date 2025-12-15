#!/usr/bin/env python3
"""
Test the unified translator with Ollama Cloud model.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from unified_translator import translate_pdf_unified

def test_ollama_cloud():
    input_pdf = Path(__file__).parent.parent / "test_3pages.pdf"
    output_dir = Path(__file__).parent / "output_cloud"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Input: {input_pdf}")
    print(f"Output dir: {output_dir}")
    print(f"Model: gpt-oss:20b-cloud")
    
    def progress(current, total, status):
        # Sanitize status for Windows console
        safe_status = status.encode('ascii', 'replace').decode('ascii')
        print(f"  [{current}/{total}] {safe_status}")
    
    print("\nStarting Ollama Cloud translation to German...")
    print("-" * 50)
    
    output_path, status = translate_pdf_unified(
        str(input_pdf),
        str(output_dir),
        model="gpt-oss:20b-cloud",  # Ollama Cloud model
        target_language="German",
        progress_callback=progress,
        use_openai=False,
        openai_api_key=None
    )
    
    print("-" * 50)
    safe_status = status.encode('ascii', 'replace').decode('ascii') if status else "None"
    print(f"Status: {safe_status}")
    print(f"Output: {output_path}")
    
    if output_path and Path(output_path).exists():
        size = Path(output_path).stat().st_size
        print(f"SUCCESS: Output file created ({size} bytes)")
        os.startfile(output_path)
    else:
        print("FAILED: No output file created")

if __name__ == "__main__":
    test_ollama_cloud()
