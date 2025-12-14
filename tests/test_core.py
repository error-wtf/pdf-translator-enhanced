"""
Unit Tests - Core Module Tests

Comprehensive tests for all core translation modules.

Run with: pytest tests/test_core.py -v

© 2025 Sven Kalinowski with small help of Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
import pytest
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# FORMULA ISOLATOR TESTS
# =============================================================================

class TestFormulaIsolator:
    """Tests for formula_isolator.py"""
    
    def test_extract_inline_math(self):
        """Test extraction of inline math $...$"""
        from formula_isolator import extract_formulas
        
        text = "The equation $E=mc^2$ is famous."
        result = extract_formulas(text)
        
        assert len(result.formulas) == 1
        assert result.formulas[0].original == "$E=mc^2$"
        assert "⟦" in result.text_with_placeholders
    
    def test_extract_display_math(self):
        """Test extraction of display math $$...$$"""
        from formula_isolator import extract_formulas
        
        text = "Consider: $$\\frac{a}{b} = c$$"
        result = extract_formulas(text)
        
        assert len(result.formulas) >= 1
        assert any("\\frac" in f.original for f in result.formulas)
    
    def test_extract_equation_environment(self):
        """Test extraction of equation environment"""
        from formula_isolator import extract_formulas
        
        text = "\\begin{equation}x = y\\end{equation}"
        result = extract_formulas(text)
        
        assert len(result.formulas) >= 1
    
    def test_extract_citations(self):
        """Test extraction of citation commands"""
        from formula_isolator import extract_formulas
        
        text = "As shown by \\cite{einstein1905} and \\citet{bohr1913}."
        result = extract_formulas(text)
        
        assert len(result.formulas) == 2
    
    def test_restore_formulas(self):
        """Test formula restoration"""
        from formula_isolator import extract_formulas
        
        text = "The $E=mc^2$ equation."
        result = extract_formulas(text)
        
        # Simulate translation (just the text part)
        translated = result.text_with_placeholders.replace("The", "Die").replace("equation", "Gleichung")
        
        # Restore
        restored = result.restore(translated)
        
        assert "$E=mc^2$" in restored
        assert "Die" in restored
    
    def test_placeholder_uniqueness(self):
        """Test that placeholders are unique"""
        from formula_isolator import generate_placeholder
        
        p1 = generate_placeholder("$x$", 0)
        p2 = generate_placeholder("$y$", 1)
        p3 = generate_placeholder("$x$", 2)  # Same content, different index
        
        assert p1 != p2
        assert p1 != p3
        assert "⟦" in p1
    
    def test_formula_integrity_check(self):
        """Test formula integrity checking"""
        from formula_isolator import check_formula_integrity
        
        original = "The formula $E=mc^2$ shows energy."
        translated_good = "Die Formel $E=mc^2$ zeigt Energie."
        translated_bad = "Die Formel E=mc² zeigt Energie."  # Formula corrupted
        
        result_good = check_formula_integrity(original, translated_good)
        result_bad = check_formula_integrity(original, translated_bad)
        
        assert result_good["is_valid"]
        assert not result_bad["is_valid"]


# =============================================================================
# GLOSSARY TESTS
# =============================================================================

class TestGlossary:
    """Tests for glossary.py"""
    
    def test_never_translate_terms(self):
        """Test that certain terms are never translated"""
        from glossary import NEVER_TRANSLATE
        
        assert "SSZ" in NEVER_TRANSLATE
        assert "GR" in NEVER_TRANSLATE
        assert "LIGO" in NEVER_TRANSLATE
    
    def test_apply_glossary(self):
        """Test glossary application"""
        from glossary import apply_glossary
        
        text = "The SSZ metric differs from GR."
        protected, mapping = apply_glossary(text)
        
        assert "SSZ" not in protected or "__" in protected
        assert len(mapping) > 0
    
    def test_restore_glossary(self):
        """Test glossary restoration"""
        from glossary import apply_glossary, restore_glossary
        
        text = "The SSZ metric."
        protected, mapping = apply_glossary(text)
        
        # Simulate translation
        translated = protected.replace("The", "Die").replace("metric", "Metrik")
        
        restored = restore_glossary(translated, mapping)
        assert "SSZ" in restored


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestValidation:
    """Tests for validation.py"""
    
    def test_balanced_braces(self):
        """Test brace balance checking"""
        from validation import check_balanced_braces
        
        valid = "This is {balanced}"
        invalid = "This is {unbalanced"
        
        assert len(check_balanced_braces(valid)) == 0
        assert len(check_balanced_braces(invalid)) > 0
    
    def test_math_delimiters(self):
        """Test math delimiter checking"""
        from validation import check_math_delimiters
        
        valid = "Formula $x=1$ and $y=2$."
        invalid = "Formula $x=1 and $y=2$."  # Odd $
        
        assert len(check_math_delimiters(valid)) == 0
        assert len(check_math_delimiters(invalid)) > 0
    
    def test_environment_matching(self):
        """Test environment matching"""
        from validation import check_environments
        
        valid = "\\begin{equation}x\\end{equation}"
        invalid = "\\begin{equation}x\\end{align}"  # Mismatch
        
        assert len(check_environments(valid)) == 0
        assert len(check_environments(invalid)) > 0
    
    def test_validate_latex(self):
        """Test complete LaTeX validation"""
        from validation import validate_latex
        
        valid = "Text with $E=mc^2$ formula."
        result = validate_latex(valid)
        
        assert result.is_valid
        assert result.score > 0.8
    
    def test_translation_completeness(self):
        """Test translation completeness check"""
        from validation import check_completeness
        
        original = "This is a long sentence with many words."
        good_trans = "Dies ist ein langer Satz mit vielen Wörtern."
        short_trans = "Kurz."
        
        score_good, issues_good = check_completeness(original, good_trans)
        score_short, issues_short = check_completeness(original, short_trans)
        
        assert score_good > score_short
        assert len(issues_short) > 0


# =============================================================================
# LANGUAGES TESTS
# =============================================================================

class TestLanguages:
    """Tests for languages.py"""
    
    def test_language_count(self):
        """Test that we have 25+ languages"""
        from languages import LANGUAGES
        
        assert len(LANGUAGES) >= 25
    
    def test_get_language(self):
        """Test language retrieval"""
        from languages import get_language
        
        de = get_language("German")
        assert de is not None
        assert de.code == "de"
        assert de.native_name == "Deutsch"
        
        # Test case insensitivity
        de2 = get_language("german")
        assert de2 is not None
        
        # Test by code
        de3 = get_language("de")
        assert de3 is not None
    
    def test_rtl_languages(self):
        """Test RTL language detection"""
        from languages import is_rtl, get_rtl_languages
        
        assert is_rtl("Arabic")
        assert is_rtl("Hebrew")
        assert not is_rtl("German")
        
        rtl = get_rtl_languages()
        assert len(rtl) >= 2
    
    def test_font_recommendations(self):
        """Test font recommendations"""
        from languages import get_font_for_language
        
        de_font = get_font_for_language("German")
        zh_font = get_font_for_language("Chinese")
        
        assert de_font is not None
        assert zh_font is not None
        assert de_font != zh_font
    
    def test_language_choices(self):
        """Test UI language choices"""
        from languages import get_language_choices
        
        choices = get_language_choices()
        assert len(choices) >= 25
        
        # Check format
        name, display = choices[0]
        assert isinstance(name, str)
        assert "(" in display  # Has native name


# =============================================================================
# TRANSLATION CACHE TESTS
# =============================================================================

class TestTranslationCache:
    """Tests for translation_cache.py"""
    
    def test_cache_put_get(self):
        """Test basic cache put and get"""
        from translation_cache import TranslationCache
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TranslationCache(cache_dir=Path(tmpdir))
            
            cache.put("Hello", "Hallo", "German")
            result = cache.get("Hello", "German")
            
            assert result == "Hallo"
    
    def test_cache_miss(self):
        """Test cache miss"""
        from translation_cache import TranslationCache
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TranslationCache(cache_dir=Path(tmpdir))
            
            result = cache.get("Nonexistent", "German")
            assert result is None
    
    def test_cache_language_specific(self):
        """Test that cache is language-specific"""
        from translation_cache import TranslationCache
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TranslationCache(cache_dir=Path(tmpdir))
            
            cache.put("Hello", "Hallo", "German")
            cache.put("Hello", "Bonjour", "French")
            
            assert cache.get("Hello", "German") == "Hallo"
            assert cache.get("Hello", "French") == "Bonjour"
    
    def test_cache_stats(self):
        """Test cache statistics"""
        from translation_cache import TranslationCache
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TranslationCache(cache_dir=Path(tmpdir))
            
            cache.put("Test1", "Test1_DE", "German")
            cache.get("Test1", "German")  # Hit
            cache.get("Test2", "German")  # Miss
            
            stats = cache.get_stats()
            assert stats.hits >= 1
            assert stats.misses >= 1


# =============================================================================
# QUALITY ASSURANCE TESTS
# =============================================================================

class TestQualityAssurance:
    """Tests for quality_assurance.py"""
    
    def test_quality_metrics(self):
        """Test quality metrics calculation"""
        from quality_assurance import run_quality_check
        
        original = "The black hole $M=10M_\\odot$ is massive."
        translated = "Das schwarze Loch $M=10M_\\odot$ ist massiv."
        
        metrics = run_quality_check(original, translated)
        
        assert metrics.overall_score > 0
        assert metrics.formula_score > 0
    
    def test_quality_levels(self):
        """Test quality level assignment"""
        from quality_assurance import QualityMetrics, QualityLevel
        
        excellent = QualityMetrics(overall_score=95)
        good = QualityMetrics(overall_score=80)
        poor = QualityMetrics(overall_score=30)
        
        assert excellent.level == QualityLevel.EXCELLENT
        assert good.level == QualityLevel.GOOD
        assert poor.level == QualityLevel.FAILED
    
    def test_back_translation_similarity(self):
        """Test back-translation similarity calculation"""
        from quality_assurance import calculate_back_translation_similarity
        
        original = "The black hole has an event horizon."
        similar = "The black hole has an event horizon region."
        different = "Completely unrelated text."
        
        sim_high = calculate_back_translation_similarity(original, similar)
        sim_low = calculate_back_translation_similarity(original, different)
        
        assert sim_high > sim_low


# =============================================================================
# PROGRESS TRACKER TESTS
# =============================================================================

class TestProgressTracker:
    """Tests for progress_tracker.py"""
    
    def test_tracker_creation(self):
        """Test tracker creation"""
        from progress_tracker import ProgressTracker, TranslationPhase
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock input file
            input_file = Path(tmpdir) / "test.pdf"
            input_file.write_bytes(b"mock")
            
            tracker = ProgressTracker(
                str(input_file),
                tmpdir,
                "German",
                checkpoint_dir=Path(tmpdir) / "cp"
            )
            
            assert tracker.job_id is not None
            assert tracker.progress.phase == TranslationPhase.INIT
    
    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading"""
        from progress_tracker import ProgressTracker, TranslationPhase
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "test.pdf"
            input_file.write_bytes(b"mock")
            
            tracker = ProgressTracker(
                str(input_file), tmpdir, "German",
                checkpoint_dir=Path(tmpdir) / "cp"
            )
            
            tracker.start()
            tracker.set_phase(TranslationPhase.TRANSLATION)
            tracker.save_checkpoint()
            
            # Create new tracker
            tracker2 = ProgressTracker(
                str(input_file), tmpdir, "German",
                checkpoint_dir=Path(tmpdir) / "cp"
            )
            
            assert tracker2.has_checkpoint()
            tracker2.resume()
            assert tracker2.progress.phase == TranslationPhase.TRANSLATION


# =============================================================================
# BATCH PROCESSOR TESTS
# =============================================================================

class TestBatchProcessor:
    """Tests for batch_processor.py"""
    
    def test_job_creation(self):
        """Test job creation"""
        from batch_processor import TranslationJob, JobStatus
        
        job = TranslationJob(
            id="test_job",
            input_path="/tmp/test.pdf",
            output_dir="/tmp/output",
            target_language="German"
        )
        
        assert job.status == JobStatus.PENDING
        assert not job.is_finished
    
    def test_processor_add_jobs(self):
        """Test adding jobs to processor"""
        from batch_processor import BatchProcessor, TranslationJob
        
        processor = BatchProcessor(max_workers=2)
        
        for i in range(3):
            job = TranslationJob(
                id=f"job_{i}",
                input_path=f"/tmp/test_{i}.pdf",
                output_dir="/tmp/output",
                target_language="German",
                priority=i
            )
            processor.add_job(job)
        
        # Jobs should be sorted by priority (descending)
        assert processor.jobs[0].priority == 2
        assert len(processor.jobs) == 3


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
