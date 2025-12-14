"""Tests for table_handler module."""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from table_handler import (
    is_numeric_content, is_header_row, should_translate_cell,
    protect_table_numbers, restore_table_numbers
)

class TestNumericDetection:
    def test_pure_number(self):
        assert is_numeric_content("123.45")
    
    def test_percentage(self):
        assert is_numeric_content("95.5%")
    
    def test_scientific_notation(self):
        assert is_numeric_content("1.5 × 10⁻³")
    
    def test_currency(self):
        assert is_numeric_content("$100.00")
    
    def test_text_not_numeric(self):
        assert not is_numeric_content("Hello World")
    
    def test_mixed_low_numeric(self):
        assert not is_numeric_content("Value is about 5")

class TestHeaderDetection:
    def test_text_header(self):
        cells = ["Name", "Value", "Unit", "Description"]
        assert is_header_row(cells)
    
    def test_numeric_row(self):
        cells = ["1.5", "2.3", "4.7", "8.9"]
        assert not is_header_row(cells)
    
    def test_empty_row(self):
        assert not is_header_row([])

class TestShouldTranslate:
    def test_text_should_translate(self):
        assert should_translate_cell("Description of the experiment")
    
    def test_number_should_not_translate(self):
        assert not should_translate_cell("123.45")
    
    def test_empty_should_not_translate(self):
        assert not should_translate_cell("")
    
    def test_single_char_should_not(self):
        assert not should_translate_cell("x")
    
    def test_units_should_not(self):
        assert not should_translate_cell("± 0.5%")

class TestNumberProtection:
    def test_protect_percentage(self):
        text = "Efficiency is 95.5% at peak."
        protected, mapping = protect_table_numbers(text)
        assert "95.5%" not in protected
        assert len(mapping) >= 1
    
    def test_protect_currency(self):
        text = "Cost: $100.00"
        protected, mapping = protect_table_numbers(text)
        assert "$100" not in protected
    
    def test_restore_numbers(self):
        text = "Value is 12.5%"
        protected, mapping = protect_table_numbers(text)
        restored = restore_table_numbers(protected, mapping)
        assert "12.5%" in restored

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
