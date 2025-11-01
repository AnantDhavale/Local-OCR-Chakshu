"""
Unit tests for OCR functionality
"""

import pytest
from hybrid_ocr import HybridOCR
from pathlib import Path

TEST_IMAGE_DIR = Path(__file__).parent / "test_images"


def test_ocr_initialization():
    """Test OCR can be initialized"""
    ocr = HybridOCR(use_llm=False)
    assert ocr is not None


def test_basic_ocr():
    """Test basic OCR extraction"""
    ocr = HybridOCR(use_llm=False)
    
    # You'll need to create a test image
    test_image = TEST_IMAGE_DIR / "sample1.png"
    
    if test_image.exists():
        result = ocr.process(str(test_image))
        
        assert 'raw_text' in result
        assert 'corrected_text' in result
        assert 'metadata' in result
        assert result['metadata']['words_processed'] > 0


def test_correction_improves_accuracy():
    """Test that correction improves over raw OCR"""
    ocr = HybridOCR(use_llm=False)
    
    # Mock test - replace with real test image + ground truth
    # result = ocr.process("noisy_image.png")
    # assert len(result['corrected_text']) > 0
    pass  # Implement with your test images


if __name__ == "__main__":
    pytest.main([__file__])
