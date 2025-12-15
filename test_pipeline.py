"""
Pipeline Test Script - Tests extraction, table hardening, post-processing

Tests the full PDF translation pipeline in steps:
1. PDF Extraction (PyMuPDF)
2. Table Detection + Hardening
3. Scientific Post-Processor (5-Step Resolve)
4. Full Pipeline Integration

Usage:
    python test_pipeline.py test_with_plots.pdf
"""
import sys
import os
import json
from pathlib import Path

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

import fitz  # PyMuPDF


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_step(step: str):
    print(f"\n>>> {step}")
    print("-" * 50)


# =============================================================================
# STEP 1: PDF EXTRACTION
# =============================================================================

def test_extraction(pdf_path: str) -> dict:
    """Test PDF extraction using PyMuPDF."""
    print_header("STEP 1: PDF Extraction")
    
    doc = fitz.open(pdf_path)
    results = {
        "pages": len(doc),
        "extraction_data": []
    }
    
    for page_num in range(min(3, len(doc))):  # First 3 pages
        page = doc[page_num]
        print_step(f"Page {page_num + 1}")
        
        # Extract text blocks
        blocks = page.get_text("dict")["blocks"]
        text_blocks = [b for b in blocks if b.get("type") == 0]  # Text blocks
        image_blocks = [b for b in blocks if b.get("type") == 1]  # Image blocks
        
        # Extract drawings (for table lines)
        drawings = page.get_drawings()
        
        page_data = {
            "page_num": page_num + 1,
            "text_blocks": len(text_blocks),
            "image_blocks": len(image_blocks),
            "drawings": len(drawings),
            "sample_text": [],
        }
        
        # Get sample text from first few blocks
        for block in text_blocks[:5]:
            for line in block.get("lines", [])[:2]:
                text = " ".join(span.get("text", "") for span in line.get("spans", []))
                if text.strip():
                    page_data["sample_text"].append(text.strip()[:80])
        
        results["extraction_data"].append(page_data)
        
        print(f"  Text blocks: {len(text_blocks)}")
        print(f"  Image blocks: {len(image_blocks)}")
        print(f"  Drawings (lines): {len(drawings)}")
        print(f"  Sample text:")
        for t in page_data["sample_text"][:3]:
            print(f"    - {t}")
    
    doc.close()
    print(f"\n[OK] Extraction completed: {results['pages']} pages")
    return results


# =============================================================================
# STEP 2: TABLE DETECTION + HARDENING
# =============================================================================

def test_table_detection(pdf_path: str) -> dict:
    """Test table detection and hardening."""
    print_header("STEP 2: Table Detection + Hardening")
    
    try:
        from table_detector import detect_tables_in_page, TextBlock, TABLE_HARDENING_AVAILABLE
        from table_hardening import TableArtifactResolver, StitchedCell
    except ImportError as e:
        print(f"[ERROR] Could not import table modules: {e}")
        return {"error": str(e)}
    
    print(f"  Table Hardening Available: {TABLE_HARDENING_AVAILABLE}")
    
    doc = fitz.open(pdf_path)
    results = {
        "tables_found": 0,
        "hardening_available": TABLE_HARDENING_AVAILABLE,
        "page_results": []
    }
    
    for page_num in range(min(3, len(doc))):
        page = doc[page_num]
        print_step(f"Page {page_num + 1} - Table Detection")
        
        # Get text blocks for table detection
        blocks_dict = page.get_text("dict")["blocks"]
        text_blocks = []
        
        for block in blocks_dict:
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        bbox = span.get("bbox", [0, 0, 100, 20])
                        text_blocks.append({
                            "text": span.get("text", ""),
                            "x": bbox[0],
                            "y": bbox[1],
                            "width": bbox[2] - bbox[0],
                            "height": bbox[3] - bbox[1],
                            "font_size": span.get("size", 10),
                            "is_bold": "Bold" in span.get("font", ""),
                        })
        
        # Detect tables
        tables, remaining = detect_tables_in_page(
            text_blocks, 
            page.rect.width, 
            page.rect.height,
            use_ml=False
        )
        
        page_result = {
            "page_num": page_num + 1,
            "tables_detected": len(tables),
            "remaining_blocks": len(remaining),
            "table_details": []
        }
        
        for i, table in enumerate(tables):
            detail = {
                "rows": table.rows,
                "cols": table.cols,
                "confidence": table.confidence,
                "method": table.detection_method,
                "has_header": table.has_header,
                "sample_cells": []
            }
            
            # Get sample cells
            for cell in table.cells[:6]:
                detail["sample_cells"].append({
                    "row": cell.row,
                    "col": cell.col,
                    "text": cell.text[:50] if cell.text else "(empty)"
                })
            
            page_result["table_details"].append(detail)
            print(f"  Table {i+1}: {table.rows}x{table.cols}, conf={table.confidence:.2f}")
        
        results["tables_found"] += len(tables)
        results["page_results"].append(page_result)
        
        if not tables:
            print("  No tables detected on this page")
    
    doc.close()
    
    # Test artifact resolver
    print_step("Table Artifact Resolver Test")
    resolver = TableArtifactResolver()
    
    test_cells = [
        ("10?2? units", "Exponent fix"),
        ("100?500 MHz", "Range fix"),
        ("m?s", "Unit fix"),
    ]
    
    for text, desc in test_cells:
        cell = StitchedCell(row=0, col=0, text=text, tokens=[], is_numeric=True)
        resolved = resolver.resolve_cell(cell)
        status = "FIXED" if resolved.artifact_fixed else "unchanged"
        print(f"  {desc}: '{text}' -> '{resolved.text}' [{status}]")
    
    print(f"\n[OK] Table detection completed: {results['tables_found']} tables found")
    return results


# =============================================================================
# STEP 3: SCIENTIFIC POST-PROCESSOR
# =============================================================================

def test_postprocessor(pdf_path: str) -> dict:
    """Test scientific post-processor with sample text."""
    print_header("STEP 3: Scientific Post-Processor")
    
    try:
        from scientific_postprocessor import ScientificPostProcessor, RepairMode
    except ImportError as e:
        print(f"[ERROR] Could not import post-processor: {e}")
        return {"error": str(e)}
    
    # Extract some text from PDF for testing
    doc = fitz.open(pdf_path)
    sample_texts = []
    
    for page_num in range(min(2, len(doc))):
        page = doc[page_num]
        text = page.get_text()
        # Get first 500 chars
        if text.strip():
            sample_texts.append(text[:500])
    
    doc.close()
    
    results = {
        "mode": "safe_repair",
        "tests": []
    }
    
    processor = ScientificPostProcessor(mode=RepairMode.SAFE_REPAIR)
    
    print_step("Processing sample text from PDF")
    
    for i, sample in enumerate(sample_texts[:2]):
        print(f"\n  Sample {i+1} (first 200 chars):")
        print(f"    Input: {sample[:200].replace(chr(10), ' ')}...")
        
        result, report = processor.process(sample)
        
        test_result = {
            "sample_num": i + 1,
            "input_len": len(sample),
            "output_len": len(result),
            "fixes_applied": report.total_fixes,
        }
        results["tests"].append(test_result)
        
        print(f"    Fixes applied: {report.total_fixes}")
        if report.actions:
            for action in report.actions[:3]:
                print(f"      - {action.rule_name}: '{action.original}' -> '{action.fixed}'")
    
    # Test with known artifact patterns
    print_step("Testing known artifact patterns")
    
    test_patterns = [
        ("ESO?Spektroskopie achieves 99,1?% accuracy", "Unicode compound"),
        ("For r r*, SSZ predicts lower values", "Missing operator"),
        ("Value is 10?2? meters", "Broken exponent"),
        ("The the results show improvement", "Duplicated word"),
    ]
    
    for text, desc in test_patterns:
        result, report = processor.process(text)
        status = "FIXED" if report.total_fixes > 0 else "unchanged"
        print(f"  {desc}:")
        print(f"    Input:  '{text}'")
        print(f"    Output: '{result}' [{status}, {report.total_fixes} fixes]")
    
    print(f"\n[OK] Post-processor test completed")
    return results


# =============================================================================
# STEP 4: FULL PIPELINE TEST
# =============================================================================

def test_full_pipeline(pdf_path: str) -> dict:
    """Test the complete translation pipeline."""
    print_header("STEP 4: Full Pipeline Test")
    
    try:
        from unified_translator import translate_pdf_unified
    except ImportError as e:
        print(f"[ERROR] Could not import unified_translator: {e}")
        return {"error": str(e)}
    
    # Create output directory
    output_dir = Path(pdf_path).parent / "test_output"
    output_dir.mkdir(exist_ok=True)
    
    print_step("Running translate_pdf_unified")
    print(f"  Input: {pdf_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Target language: German")
    print(f"  Repair mode: safe_repair")
    
    # Progress callback (takes 3 args: current, total, message)
    def progress_callback(current, total, msg):
        print(f"  >> [{current}/{total}] {msg}")
    
    try:
        # Run translation (this will use Ollama if available)
        result_path, status = translate_pdf_unified(
            input_pdf=pdf_path,
            output_dir=str(output_dir),
            model="llama3.2",  # Use local model
            target_language="German",
            progress_callback=progress_callback,
            use_openai=False,
            repair_mode="safe_repair"
        )
        
        results = {
            "success": result_path is not None,
            "output_path": result_path,
            "status": status
        }
        
        if result_path:
            print(f"\n[OK] Pipeline completed successfully!")
            print(f"  Output: {result_path}")
        else:
            print(f"\n[WARNING] Pipeline returned no output: {status}")
        
        return results
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("  PDF TRANSLATOR PIPELINE TEST")
    print("=" * 70)
    
    # Get PDF path
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "test_with_plots.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"[ERROR] PDF not found: {pdf_path}")
        sys.exit(1)
    
    print(f"\nTest PDF: {pdf_path}")
    print(f"Size: {os.path.getsize(pdf_path) / 1024:.1f} KB")
    
    # Run all tests
    results = {}
    
    # Step 1: Extraction
    results["extraction"] = test_extraction(pdf_path)
    
    # Step 2: Table Detection
    results["table_detection"] = test_table_detection(pdf_path)
    
    # Step 3: Post-Processor
    results["postprocessor"] = test_postprocessor(pdf_path)
    
    # Step 4: Full Pipeline
    # Check for --full flag or run automatically
    run_full = "--full" in sys.argv or len(sys.argv) > 2
    if run_full:
        results["full_pipeline"] = test_full_pipeline(pdf_path)
    else:
        print("\n[INFO] Full pipeline test skipped (add --full to run)")
        results["full_pipeline"] = {"skipped": True}
    
    # Summary
    print_header("TEST SUMMARY")
    
    print(f"  Extraction: {results['extraction'].get('pages', 0)} pages")
    print(f"  Tables found: {results['table_detection'].get('tables_found', 0)}")
    print(f"  Hardening available: {results['table_detection'].get('hardening_available', False)}")
    
    if "error" not in results.get("postprocessor", {}):
        print(f"  Post-processor: OK")
    else:
        print(f"  Post-processor: ERROR")
    
    if results.get("full_pipeline", {}).get("success"):
        print(f"  Full pipeline: SUCCESS")
    elif results.get("full_pipeline", {}).get("skipped"):
        print(f"  Full pipeline: SKIPPED")
    else:
        print(f"  Full pipeline: {results.get('full_pipeline', {}).get('error', 'FAILED')}")
    
    print("\n" + "=" * 70)
    print("  TEST COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
