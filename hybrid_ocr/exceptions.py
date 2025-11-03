"""
Custom exceptions for Chakshu OCR
"""


class OCRError(Exception):
    """Base exception for OCR errors"""
    pass


class ImageProcessingError(OCRError):
    """Raised when image preprocessing fails"""
    pass


class TesseractNotFoundError(OCRError):
    """Raised when Tesseract is not found or not properly installed"""
    pass


class InvalidImageError(OCRError):
    """Raised when an invalid image is provided"""
    pass


class CorrectionError(OCRError):
    """Raised when text correction fails"""
    pass


class ModelLoadError(OCRError):
    """Raised when a model fails to load"""
    pass
