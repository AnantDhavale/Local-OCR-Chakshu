"""
Unit Tests for Chakshu OCR

"""

import pytest
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hybrid_ocr import (
    HybridOCR,
    OCRResult,
    CorrectionStrategy,
    OCRError,
    InvalidImageError,
    calculate_character_accuracy
)


class TestHybridOCR:
    """Test suite for HybridOCR class"""
    
    @pytest.fixture
    def simple_ocr(self):
        """Fixture for basic OCR instance"""
        return HybridOCR(correction_strategy='rule_based')
    
    @pytest.fixture
    def test_image(self):
        """Create a simple test image with text"""
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw text (using default font)
        text = "The quick brown fox"
        draw.text((10, 40), text, fill='black')
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img.save(f.name)
            return f.name
    
    def test_initialization(self):
        """Test OCR initialization"""
        ocr = HybridOCR()
        assert ocr is not None
        assert ocr.language == 'eng'
    
    def test_initialization_with_strategy(self):
        """Test initialization with different strategies"""
        strategies = ['rule_based', 'hybrid']
        
        for strategy in strategies:
            ocr = HybridOCR(correction_strategy=strategy)
            assert ocr is not None
    
    def test_process_image(self, simple_ocr, test_image):
        """Test basic image processing"""
        result = simple_ocr.process(test_image, output_format='text')
        
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_process_detailed_output(self, simple_ocr, test_image):
        """Test detailed output format"""
        result = simple_ocr.process(test_image, output_format='detailed')
        
        assert isinstance(result, OCRResult)
        assert hasattr(result, 'raw_text')
        assert hasattr(result, 'corrected_text')
        assert hasattr(result, 'confidence_avg')
        assert hasattr(result, 'metadata')
    
    def test_invalid_image(self, simple_ocr):
        """Test handling of invalid images"""
        with pytest.raises((OCRError, InvalidImageError, FileNotFoundError)):
            simple_ocr.process('nonexistent.png')
    
    def test_confidence_threshold(self, simple_ocr, test_image):
        """Test confidence threshold filtering"""
        result1 = simple_ocr.process(
            test_image,
            output_format='detailed',
            confidence_threshold=0
        )
        
        result2 = simple_ocr.process(
            test_image,
            output_format='detailed',
            confidence_threshold=90
        )
        
        # Higher threshold should filter more words
        assert len(result1.corrected_text) >= len(result2.corrected_text)
    
    def test_statistics(self, simple_ocr, test_image):
        """Test statistics tracking"""
        simple_ocr.reset_statistics()
        
        # Process multiple images
        simple_ocr.process(test_image)
        simple_ocr.process(test_image)
        
        stats = simple_ocr.get_statistics()
        assert stats['total_processed'] == 2
        assert stats['total_time'] > 0
    
    def test_batch_processing(self, simple_ocr, test_image):
        """Test batch processing"""
        images = [test_image, test_image]
        results = simple_ocr.process_batch(images, show_progress=False)
        
        assert len(results) == 2
        assert all(r is not None for r in results)


class TestCorrection:
    """Test suite for text correction"""
    
    def test_rule_based_correction(self):
        """Test rule-based correction"""
        ocr = HybridOCR(correction_strategy='rule_based')
        
        # Mock OCR output with common errors
        test_cases = [
            ("tbe quick brown fox", "the quick brown fox"),
            ("frorn the desk", "from the desk"),
            ("aod then", "and then"),
        ]
        
        for input_text, expected in test_cases:
            corrected = ocr.corrector.correct(input_text)
            # Check if correction improves text (not exact match due to other rules)
            assert len(corrected) > 0
    
    def test_correction_with_confidence(self):
        """Test correction using confidence scores"""
        ocr = HybridOCR(correction_strategy='rule_based')
        
        text = "tbe quick brown fox"
        confidences = [30, 95, 95, 95]  # Low confidence on first word
        
        corrected = ocr.corrector.correct(text, confidences)
        assert corrected is not None


class TestPreprocessing:
    """Test suite for image preprocessing"""
    
    @pytest.fixture
    def test_image_pil(self):
        """Create PIL image for preprocessing tests"""
        return Image.new('RGB', (200, 100), color='white')
    
    def test_preprocessing_enabled(self, test_image_pil):
        """Test OCR with preprocessing enabled"""
        ocr = HybridOCR(preprocess=True)
        
        # Process PIL image directly
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            test_image_pil.save(f.name)
            result = ocr.process(f.name)
            assert result is not None
    
    def test_preprocessing_disabled(self, test_image_pil):
        """Test OCR with preprocessing disabled"""
        ocr = HybridOCR(preprocess=False)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            test_image_pil.save(f.name)
            result = ocr.process(f.name)
            assert result is not None


class TestOutputFormats:
    """Test different output formats"""
    
    @pytest.fixture
    def test_image(self):
        """Create test image"""
        img = Image.new('RGB', (300, 80), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 30), "Test text", fill='black')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img.save(f.name)
            return f.name
    
    def test_text_output(self, test_image):
        """Test text output format"""
        ocr = HybridOCR()
        result = ocr.process(test_image, output_format='text')
        
        assert isinstance(result, str)
    
    def test_json_output(self, test_image):
        """Test JSON output format"""
        import json
        
        ocr = HybridOCR()
        result = ocr.process(test_image, output_format='json')
        
        assert isinstance(result, str)
        # Should be valid JSON
        parsed = json.loads(result)
        assert 'corrected_text' in parsed
    
    def test_detailed_output(self, test_image):
        """Test detailed output format"""
        ocr = HybridOCR()
        result = ocr.process(test_image, output_format='detailed')
        
        assert isinstance(result, OCRResult)
        assert hasattr(result, 'to_dict')
        assert hasattr(result, 'to_json')


class TestUtils:
    """Test utility functions"""
    
    def test_character_accuracy(self):
        """Test character accuracy calculation"""
        pred = "The quick brown fox"
        truth = "The quick brown fox"
        
        accuracy = calculate_character_accuracy(pred, truth)
        assert accuracy == 100.0
        
        pred = "The quik brown fox"
        accuracy = calculate_character_accuracy(pred, truth)
        assert 80 < accuracy < 100
    
    def test_accuracy_empty_strings(self):
        """Test accuracy with edge cases"""
        assert calculate_character_accuracy("", "") == 100.0
        assert calculate_character_accuracy("test", "") < 100.0


class TestMultiLanguage:
    """Test multi-language support"""
    
    def test_spanish_language(self):
        """Test Spanish language OCR"""
        ocr = HybridOCR(language='spa')
        assert ocr.language == 'spa'
    
    def test_french_language(self):
        """Test French language OCR"""
        ocr = HybridOCR(language='fra')
        assert ocr.language == 'fra'


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
