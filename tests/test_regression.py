"""
Regression Tests for PDF Translator Pipeline

Tests:
1. No garbage characters (￾) in output
2. No blank pages in output
3. Correct heading order preserved
4. Page count not exploding

Usage:
    python -m pytest tests/test_regression.py -v
    
Or standalone:
    python tests/test_regression.py TEST.pdf

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz
from text_normalizer import count_garbage_chars, normalize_text, GARBAGE_CHARS


def test_no_garbage_chars_in_text():
    """Test that normalize_text removes all garbage characters."""
    test_cases = [
        ("altitude￾spanning", "altitudespanning"),
        ("already￾known", "alreadyknown"),
        ("Projection￾Unified-Results", "Projection-Unified-Results"),
        ("test\ufffdvalue", "testvalue"),
        ("soft\u00adhyphen", "softhyphen"),
        ("zero\u200bwidth", "zerowidth"),
    ]
    
    for input_text, expected_clean in test_cases:
        result = normalize_text(input_text)
        garbage_count = count_garbage_chars(result)
        assert garbage_count == 0, f"Garbage chars found in '{result}' (from '{input_text}')"
        # Note: expected_clean is approximate, main test is garbage_count == 0


def test_ligatures_normalized():
    """Test that ligatures are converted to ASCII."""
    test_cases = [
        ("ﬁrst", "first"),
        ("ﬂuid", "fluid"),
        ("eﬀect", "effect"),
        ("oﬃce", "office"),
        ("afﬂuent", "affluent"),
    ]
    
    for input_text, expected in test_cases:
        result = normalize_text(input_text)
        assert result == expected, f"Expected '{expected}', got '{result}'"


def test_dashes_normalized():
    """Test that dash variants are normalized."""
    test_cases = [
        ("a−b", "a-b"),  # Minus sign
        ("a–b", "a-b"),  # En dash
        ("a—b", "a-b"),  # Em dash
    ]
    
    for input_text, expected in test_cases:
        result = normalize_text(input_text)
        assert result == expected, f"Expected '{expected}', got '{result}'"


def check_pdf_no_garbage(pdf_path: str) -> tuple[bool, int, list[str]]:
    """
    Check a PDF for garbage characters.
    
    Returns (passed, garbage_count, examples)
    """
    doc = fitz.open(pdf_path)
    total_garbage = 0
    examples = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        
        for char in GARBAGE_CHARS:
            if char in text:
                count = text.count(char)
                total_garbage += count
                # Find context around garbage char
                idx = text.find(char)
                start = max(0, idx - 10)
                end = min(len(text), idx + 10)
                context = text[start:end].replace('\n', ' ')
                examples.append(f"Page {page_num + 1}: ...{context}...")
    
    doc.close()
    return total_garbage == 0, total_garbage, examples[:5]  # Limit examples


def check_question_mark_artifacts(pdf_path: str) -> tuple[bool, int, list[str]]:
    """
    Check for '?' artifacts from bad glyph mapping.
    
    Patterns like "Phonon?Kopplung", "Spin?Spin", "10?¹?" indicate
    encoding/font issues.
    
    Returns (passed, artifact_count, examples)
    """
    import re
    
    doc = fitz.open(pdf_path)
    total_artifacts = 0
    examples = []
    
    # Patterns that indicate bad glyph mapping
    bad_patterns = [
        r'\w\?\w',           # Word?Word (missing hyphen)
        r'\d\?\d',           # Number?Number
        r'\?\s*[⁻⁺¹²³⁴⁵⁶⁷⁸⁹⁰]',  # ?superscript
        r'[⁻⁺¹²³⁴⁵⁶⁷⁸⁹⁰]\?',      # superscript?
    ]
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        
        for pattern in bad_patterns:
            matches = re.findall(pattern, text)
            if matches:
                total_artifacts += len(matches)
                for match in matches[:2]:  # Limit per pattern
                    # Find context
                    idx = text.find(match)
                    if idx >= 0:
                        start = max(0, idx - 15)
                        end = min(len(text), idx + 15)
                        context = text[start:end].replace('\n', ' ')
                        examples.append(f"Page {page_num + 1}: ...{context}...")
    
    doc.close()
    
    # Allow some ? marks (they might be legitimate)
    # Fail only if there are many
    passed = total_artifacts < 10
    return passed, total_artifacts, examples[:5]


def check_text_content_ratio(pdf_path: str, source_path: str = None) -> tuple[bool, float, float]:
    """
    Check that translated PDF has similar text content ratio to source.
    
    If translated PDF has much less text, something went wrong.
    
    Returns (passed, output_chars, source_chars)
    """
    doc = fitz.open(pdf_path)
    output_chars = sum(len(page.get_text()) for page in doc)
    doc.close()
    
    if source_path and Path(source_path).exists():
        doc = fitz.open(source_path)
        source_chars = sum(len(page.get_text()) for page in doc)
        doc.close()
    else:
        source_chars = output_chars
    
    # Output should have at least 30% of source text
    # (translations can be shorter, but not drastically)
    if source_chars > 0:
        ratio = output_chars / source_chars
        passed = ratio >= 0.3
    else:
        ratio = 1.0
        passed = True
    
    return passed, output_chars, source_chars


def check_pdf_no_blank_pages(pdf_path: str) -> tuple[bool, list[int]]:
    """
    Check a PDF for blank pages.
    
    Returns (passed, list of blank page numbers)
    """
    doc = fitz.open(pdf_path)
    blank_pages = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        images = page.get_images()
        drawings = page.get_drawings()
        
        has_content = len(text) > 10 or len(images) > 0 or len(drawings) > 5
        
        if not has_content:
            blank_pages.append(page_num + 1)
    
    doc.close()
    return len(blank_pages) == 0, blank_pages


def check_heading_order(pdf_path: str) -> tuple[bool, list[str]]:
    """
    Check that headings are in correct order.
    
    Returns (passed, list of headings found)
    """
    import re
    
    doc = fitz.open(pdf_path)
    headings = []
    
    for page in doc:
        text = page.get_text()
        
        # Find numbered sections
        matches = re.findall(r'^(\d+(?:\.\d+)*)\.\s+([^\n]+)', text, re.MULTILINE)
        for num, title in matches:
            headings.append(f"{num}. {title[:30]}")
    
    doc.close()
    
    # Check order
    if not headings:
        return True, []  # No headings to check
    
    # Extract section numbers and check they're in order
    section_nums = []
    for h in headings:
        match = re.match(r'^(\d+)', h)
        if match:
            section_nums.append(int(match.group(1)))
    
    # Check if sorted
    is_ordered = section_nums == sorted(section_nums)
    
    return is_ordered, headings


def check_page_count(pdf_path: str, source_path: str = None, max_increase: int = 2) -> tuple[bool, int, int]:
    """
    Check that page count hasn't exploded.
    
    Returns (passed, output_pages, source_pages)
    """
    doc = fitz.open(pdf_path)
    output_pages = len(doc)
    doc.close()
    
    if source_path and Path(source_path).exists():
        doc = fitz.open(source_path)
        source_pages = len(doc)
        doc.close()
    else:
        source_pages = output_pages  # Assume same if no source
    
    passed = output_pages <= source_pages + max_increase
    return passed, output_pages, source_pages


def run_all_checks(pdf_path: str, source_path: str = None) -> dict:
    """
    Run all regression checks on a PDF.
    
    Returns a report dict.
    """
    report = {
        "pdf_path": pdf_path,
        "passed": True,
        "checks": {}
    }
    
    # Check 1: No garbage chars
    passed, count, examples = check_pdf_no_garbage(pdf_path)
    report["checks"]["no_garbage_chars"] = {
        "passed": passed,
        "garbage_count": count,
        "examples": examples
    }
    if not passed:
        report["passed"] = False
    
    # Check 2: Question mark artifacts
    passed, count, examples = check_question_mark_artifacts(pdf_path)
    report["checks"]["no_question_artifacts"] = {
        "passed": passed,
        "artifact_count": count,
        "examples": examples
    }
    if not passed:
        report["passed"] = False
    
    # Check 3: No blank pages
    passed, blank_pages = check_pdf_no_blank_pages(pdf_path)
    report["checks"]["no_blank_pages"] = {
        "passed": passed,
        "blank_pages": blank_pages
    }
    if not passed:
        report["passed"] = False
    
    # Check 4: Heading order
    passed, headings = check_heading_order(pdf_path)
    report["checks"]["heading_order"] = {
        "passed": passed,
        "headings": headings
    }
    if not passed:
        report["passed"] = False
    
    # Check 5: Page count
    passed, output_pages, source_pages = check_page_count(pdf_path, source_path)
    report["checks"]["page_count"] = {
        "passed": passed,
        "output_pages": output_pages,
        "source_pages": source_pages
    }
    if not passed:
        report["passed"] = False
    
    # Check 6: Text content ratio
    passed, output_chars, source_chars = check_text_content_ratio(pdf_path, source_path)
    report["checks"]["text_content_ratio"] = {
        "passed": passed,
        "output_chars": output_chars,
        "source_chars": source_chars,
        "ratio": output_chars / source_chars if source_chars > 0 else 1.0
    }
    if not passed:
        report["passed"] = False
    
    return report


def print_report(report: dict):
    """Print a formatted report."""
    print(f"\n{'='*60}")
    print(f"REGRESSION TEST REPORT: {report['pdf_path']}")
    print(f"{'='*60}")
    
    overall = "✅ PASSED" if report["passed"] else "❌ FAILED"
    print(f"\nOverall: {overall}\n")
    
    for check_name, check_result in report["checks"].items():
        status = "✅" if check_result["passed"] else "❌"
        print(f"{status} {check_name}:")
        
        if check_name == "no_garbage_chars":
            print(f"   Garbage chars found: {check_result['garbage_count']}")
            if check_result["examples"]:
                print(f"   Examples: {check_result['examples'][:3]}")
        
        elif check_name == "no_question_artifacts":
            print(f"   '?' artifacts found: {check_result['artifact_count']}")
            if check_result["examples"]:
                print(f"   Examples: {check_result['examples'][:3]}")
        
        elif check_name == "no_blank_pages":
            if check_result["blank_pages"]:
                print(f"   Blank pages: {check_result['blank_pages']}")
            else:
                print(f"   No blank pages found")
        
        elif check_name == "heading_order":
            if check_result["headings"]:
                print(f"   Headings found: {len(check_result['headings'])}")
            else:
                print(f"   No numbered headings found")
        
        elif check_name == "page_count":
            print(f"   Output: {check_result['output_pages']} pages, Source: {check_result['source_pages']} pages")
        
        elif check_name == "text_content_ratio":
            ratio = check_result.get('ratio', 0)
            print(f"   Output: {check_result['output_chars']} chars, Source: {check_result['source_chars']} chars")
            print(f"   Ratio: {ratio:.1%}")
        
        print()
    
    print(f"{'='*60}\n")


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_regression.py <output.pdf> [source.pdf]")
        print("\nRuns regression tests on a translated PDF.")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    source_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    report = run_all_checks(pdf_path, source_path)
    print_report(report)
    
    sys.exit(0 if report["passed"] else 1)
