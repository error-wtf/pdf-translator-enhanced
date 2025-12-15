#!/usr/bin/env python3
"""
Test the unified translator with OpenAI API.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from unified_translator import translate_pdf_unified

def test_openai_translation():
    input_pdf = Path(__file__).parent.parent / "test_3pages.pdf"
    output_dir = Path(__file__).parent / "output_openai"
    output_dir.mkdir(exist_ok=True)
    
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or len(api_key) < 10:
        print("ERROR: OPENAI_API_KEY not set!")
        return
    
    print(f"Input: {input_pdf}")
    print(f"Output dir: {output_dir}")
    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
    
    def progress(current, total, status):
        # Sanitize status for Windows console
        safe_status = status.encode('ascii', 'replace').decode('ascii')
        print(f"  [{current}/{total}] {safe_status}")
    
    print("\nStarting OpenAI translation to German...")
    print("-" * 50)
    
    output_path, status = translate_pdf_unified(
        str(input_pdf),
        str(output_dir),
        model="gpt-4o-mini",  # OpenAI model
        target_language="German",
        progress_callback=progress,
        use_openai=True,
        openai_api_key=api_key
    )
    
    print("-" * 50)
    # Sanitize status for Windows console
    safe_status = status.encode('ascii', 'replace').decode('ascii') if status else "None"
    print(f"Status: {safe_status}")
    print(f"Output: {output_path}")
    
    if output_path and Path(output_path).exists():
        size = Path(output_path).stat().st_size
        print(f"SUCCESS: Output file created ({size} bytes)")
        # Open the PDF
        os.startfile(output_path)
    else:
        print("FAILED: No output file created")

if __name__ == "__main__":
    test_openai_translation()
