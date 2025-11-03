"""
Chakshu OCR - Hybrid OCR with intelligent correction
"""

from .ocr import HybridOCR, OCRResult, OCRWord, OutputFormat, ImagePreprocessor
from .corrector import CorrectionStrategy, RuleBasedCorrector, LLMCorrector, HybridCorrector
from .exceptions import (
    OCRError,
    ImageProcessingError,
    TesseractNotFoundError,
    InvalidImageError,
    CorrectionError,
    ModelLoadError
)
from .utils import (
    setup_logging,
    validate_image,
    calculate_character_accuracy,
    calculate_word_accuracy,
    format_confidence,
    is_supported_image,
    get_supported_image_formats
)

__version__ = "0.1.0"
__author__ = "Anant Dhavale"
__license__ = "MIT"

__all__ = [
    'HybridOCR',
    'OCRResult',
    'OCRWord',
    'OutputFormat',
    'ImagePreprocessor',
    'CorrectionStrategy',
    'RuleBasedCorrector',
    'LLMCorrector',
    'HybridCorrector',
    'OCRError',
    'ImageProcessingError',
    'TesseractNotFoundError',
    'InvalidImageError',
    'CorrectionError',
    'ModelLoadError',
    'setup_logging',
    'validate_image',
    'calculate_character_accuracy',
    'calculate_word_accuracy',
    'format_confidence',
    'is_supported_image',
    'get_supported_image_formats',
]
