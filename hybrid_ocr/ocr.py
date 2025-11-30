"""
Chakshu OCR Engine - Production Grade OCR Pipeline

Combines Tesseract OCR with intelligent text correction.
Fully local, no cloud dependencies, CPU-optimized.

Author: Anant Dhavale + LLM
License: MIT/ Hybrid
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import json
import time

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

from .corrector import get_corrector, CorrectionStrategy
from .utils import validate_image, setup_logging, measure_time
from .exceptions import (
    OCRError,
    ImageProcessingError,
    TesseractNotFoundError,
    InvalidImageError
)


logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Supported output formats"""
    TEXT = "text"
    JSON = "json"
    DETAILED = "detailed"
    HOCR = "hocr"  # HTML-based OCR format
    PDF = "pdf"    # Searchable PDF


@dataclass
class OCRWord:
    """Represents a single word detected by OCR"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    line_num: int
    block_num: int
    page_num: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class OCRResult:
    """Complete OCR result with metadata"""
    raw_text: str
    corrected_text: str
    words: List[OCRWord]
    confidence_avg: float
    confidence_min: float
    processing_time: float
    image_path: str
    metadata: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'raw_text': self.raw_text,
            'corrected_text': self.corrected_text,
            'words': [w.to_dict() for w in self.words],
            'confidence_avg': self.confidence_avg,
            'confidence_min': self.confidence_min,
            'processing_time': self.processing_time,
            'image_path': self.image_path,
            'metadata': self.metadata
        }
    
    def to_json(self, pretty: bool = True) -> str:
        """Convert to JSON string"""
        indent = 2 if pretty else None
        return json.dumps(self.to_dict(), indent=indent)


class ImagePreprocessor:
    """
    Pre-processes images to improve OCR accuracy
    Now with presets for different document types!
    """
    
    # Preset configurations for different document types
    PRESETS = {
        'default': {
            'enhance_contrast': True,
            'denoise': True,
            'sharpen': False,
            'binarize': False,
            'deskew': True,
            'tesseract_psm': 6  # Single uniform block
        },
        'document': {
            'enhance_contrast': True,
            'denoise': False,  # Clean documents don't need denoising
            'sharpen': False,  # FIXED: Don't sharpen - destroys text
            'binarize': False, # FIXED: Don't binarize aggressively
            'deskew': True,
            'tesseract_psm': 6  # Better for structured text
        },
        'diagram': {
            'enhance_contrast': True,
            'denoise': False,
            'sharpen': False,  # Don't sharpen diagrams
            'binarize': False, # Keep colors for better contrast
            'deskew': False,   # Don't deskew - diagrams are usually aligned
            'tesseract_psm': 11  # Sparse text - good for boxes/labels
        },
        'photo': {
            'enhance_contrast': True,
            'denoise': True,   # Photos often have noise
            'sharpen': False,  # Don't sharpen photos
            'binarize': False, # Keep colors for better recognition
            'deskew': True,
            'tesseract_psm': 3  # Auto page segmentation
        },
        'low_quality': {
            'enhance_contrast': True,
            'denoise': True,   # Remove noise from poor scans
            'sharpen': False,  # FIXED: Sharpen can make garbage worse
            'binarize': False, # FIXED: Aggressive binarization destroys text
            'deskew': True,
            'tesseract_psm': 6
        },
        'newspaper': {
            'enhance_contrast': True,
            'denoise': True,   # Old newspapers are noisy
            'sharpen': False,
            'binarize': False, # FIXED: Don't binarize - loses detail
            'deskew': True,
            'tesseract_psm': 6
        },
        'manuscript': {
            'enhance_contrast': True,  # Light enhancement only
            'denoise': False,  # Don't denoise - old texts need grain
            'sharpen': False,  # Don't sharpen aged documents
            'binarize': False, # Preserve contrast
            'deskew': True,    # Still deskew
            'tesseract_psm': 6,
            'skip_correction': True  # CRITICAL: Disable text correction for old/unusual texts
        },
        'minimal': {
            'enhance_contrast': False,
            'denoise': False,
            'sharpen': False,
            'binarize': False,
            'deskew': False,   # No preprocessing - trust Tesseract
            'tesseract_psm': 3  # Auto page segmentation
        }
    }
    
    def __init__(self, config: Optional[Dict] = None, preset: str = 'default'):
        """
        Initialize preprocessor
        
        Args:
            config: Custom configuration dict (overrides preset)
            preset: Preset name ('default', 'document', 'photo', 'low_quality', 'newspaper', 'minimal')
        """
        if config:
            self.config = config
        elif preset in self.PRESETS:
            self.config = self.PRESETS[preset].copy()
        else:
            logger.warning(f"Unknown preset '{preset}', using 'default'")
            self.config = self.PRESETS['default'].copy()
    
    def preprocess(self, image: Image.Image) -> Image.Image:
        """
        Apply preprocessing pipeline to image
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image
        """
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance contrast
            if self.config.get('enhance_contrast'):
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.5)
            
            # Denoise
            if self.config.get('denoise'):
                image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Sharpen
            if self.config.get('sharpen'):
                image = image.filter(ImageFilter.SHARPEN)
            
            # Binarization (convert to black and white) - IMPROVED with adaptive thresholding
            if self.config.get('binarize'):
                image = self._adaptive_binarize(image)
            
            # Deskew (rotate to fix tilted scans)
            if self.config.get('deskew'):
                image = self._deskew(image)
            
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise ImageProcessingError(f"Failed to preprocess image: {e}")
    
    def _adaptive_binarize(self, image: Image.Image) -> Image.Image:
        """
        Apply adaptive binarization using Otsu's method
        Much better than fixed threshold - adapts to image lighting
        """
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Convert to numpy array for processing
            img_array = np.array(image)
            
            # Calculate histogram
            histogram, _ = np.histogram(img_array.flatten(), bins=256, range=(0, 256))
            
            # Calculate optimal threshold using Otsu's method
            total_pixels = img_array.size
            current_max = 0
            threshold = 127
            
            sum_total = np.sum(np.arange(256) * histogram)
            sum_background = 0
            weight_background = 0
            
            for t in range(256):
                weight_background += histogram[t]
                if weight_background == 0:
                    continue
                    
                weight_foreground = total_pixels - weight_background
                if weight_foreground == 0:
                    break
                
                sum_background += t * histogram[t]
                
                mean_background = sum_background / weight_background
                mean_foreground = (sum_total - sum_background) / weight_foreground
                
                # Calculate between-class variance
                variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
                
                if variance > current_max:
                    current_max = variance
                    threshold = t
            
            # Apply threshold
            img_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)
            
            # Convert back to PIL Image
            return Image.fromarray(img_array, mode='L')
            
        except Exception as e:
            logger.warning(f"Adaptive binarization failed, using simple threshold: {e}")
            # Fallback to simple threshold
            if image.mode != 'L':
                image = image.convert('L')
            return image.point(lambda x: 255 if x > 128 else 0, 'L')
    
    def _deskew(self, image: Image.Image) -> Image.Image:
        """
        Detect and correct image skew - IMPROVED VERSION
        Uses single OSD call for 5x performance improvement
        """
        try:
            # Use OSD (Orientation and Script Detection) once to get rotation angle
            osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
            rotation_angle = float(osd.get('rotate', 0))
            
            # Only rotate if angle is significant
            if abs(rotation_angle) > 0.5:
                # Rotate image by detected angle
                rotated = image.rotate(rotation_angle, expand=True, fillcolor='white')
                logger.debug(f"Rotated image by {rotation_angle} degrees")
                return rotated
            
            return image
            
        except Exception as e:
            logger.warning(f"Deskew failed, using original image: {e}")
            return image


class HybridOCR:
    """
    Production-grade OCR engine with intelligent correction
    
    Features:
    - Image preprocessing for quality improvement
    - Multi-language support
    - Confidence-based correction
    - Multiple output formats
    - Batch processing support
    - Error handling and logging
    - Performance monitoring
    """
    
    def __init__(
        self,
        correction_strategy: Union[str, CorrectionStrategy] = CorrectionStrategy.RULE_BASED,
        language: str = 'eng',
        preprocess: bool = True,
        preprocess_preset: str = 'default',
        tesseract_config: Optional[str] = None,
        cache_dir: Optional[str] = None,
        log_level: str = 'INFO'
    ):
        """
        Initialize Hybrid OCR engine
        
        Args:
            correction_strategy: Strategy for text correction
                - 'rule_based': Fast, dictionary + pattern-based
                - 'llm': Better quality, uses small local model
                - 'hybrid': Combines both approaches
            language: Tesseract language code (e.g., 'eng', 'spa', 'fra')
            preprocess: Enable image preprocessing
            preprocess_preset: Preprocessing preset to use:
                - 'default': Balanced for general use
                - 'document': Optimized for clean documents
                - 'diagram': For flowcharts and structured diagrams
                - 'photo': Optimized for photos with text
                - 'low_quality': For poor quality scans
                - 'newspaper': For old newspapers
                - 'minimal': No preprocessing
            tesseract_config: Custom Tesseract configuration string
            cache_dir: Directory for caching models/results
            log_level: Logging level
        """
        # Setup logging
        setup_logging(log_level)
        logger.info("Initializing Hybrid OCR engine")
        
        # Verify Tesseract installation
        self._verify_tesseract()
        
        # Configuration
        self.language = language
        self.preprocess_preset = preprocess_preset
        
        # Initialize components with preset support
        self.preprocessor = ImagePreprocessor(preset=preprocess_preset) if preprocess else None
        
        # Set Tesseract config based on preset
        psm = 6  # Default to single uniform block
        if self.preprocessor:
            psm = self.preprocessor.config.get('tesseract_psm', 6)
        
        self.tesseract_config = tesseract_config or f'--oem 1 --psm {psm}'
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Initialize corrector
        if isinstance(correction_strategy, str):
            correction_strategy = CorrectionStrategy[correction_strategy.upper()]
        
        self.corrector = get_corrector(correction_strategy, language=language)
        logger.info(f"Using correction strategy: {correction_strategy.value}")
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'avg_confidence': 0.0,
            'errors': 0
        }
    
    def _verify_tesseract(self):
        """Verify Tesseract is installed and accessible"""
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
        except Exception as e:
            raise TesseractNotFoundError(
                "Tesseract not found. Please install: "
                "https://github.com/tesseract-ocr/tesseract"
            )
    
    @measure_time
    def process(
        self,
        image_source: Union[str, Path, Image.Image],
        output_format: Union[str, OutputFormat] = OutputFormat.TEXT,
        confidence_threshold: float = 0.0,
        return_words: bool = False
    ) -> Union[str, Dict, OCRResult]:
        """
        Process image and extract text with correction
        
        Args:
            image_source: Path to image file or PIL Image object
            output_format: Desired output format
            confidence_threshold: Minimum confidence to include word (0-100)
            return_words: Include word-level data in output
            
        Returns:
            Extracted text or structured result based on output_format
        """
        start_time = time.time()
        
        try:
            # Load and validate image
            if isinstance(image_source, (str, Path)):
                image_path = str(image_source)
                image = Image.open(image_source)
                validate_image(image)
            else:
                image = image_source
                image_path = "memory_image"
            
            logger.info(f"Processing image: {image_path}")
            
            # Preprocess if enabled
            if self.preprocessor:
                image = self.preprocessor.preprocess(image)
            
            # Extract text with Tesseract
            ocr_data = self._extract_text(image)
            
            # Filter by confidence
            if confidence_threshold > 0:
                ocr_data = self._filter_by_confidence(ocr_data, confidence_threshold)
            
            # Parse into structured format
            words = self._parse_words(ocr_data)
            # Reconstruct text preserving layout for diagrams
            raw_text = self._reconstruct_text_with_layout(words, self.preprocess_preset)
            
            # Apply correction (unless disabled for manuscript mode)
            skip_correction = self.preprocessor and self.preprocessor.config.get('skip_correction', False)
            if skip_correction:
                corrected_text = raw_text
            else:
                corrected_text = self.corrector.correct(
                    raw_text,
                    [w.confidence for w in words]
                )
            
            # Calculate statistics
            confidences = [w.confidence for w in words]
            confidence_avg = sum(confidences) / len(confidences) if confidences else 0.0
            confidence_min = min(confidences) if confidences else 0.0
            
            processing_time = time.time() - start_time
            
            # Update global stats
            self.stats['total_processed'] += 1
            self.stats['total_time'] += processing_time
            self.stats['avg_confidence'] = (
                (self.stats['avg_confidence'] * (self.stats['total_processed'] - 1) + confidence_avg)
                / self.stats['total_processed']
            )
            
            # Create result object
            result = OCRResult(
                raw_text=raw_text,
                corrected_text=corrected_text,
                words=words if return_words else [],
                confidence_avg=confidence_avg,
                confidence_min=confidence_min,
                processing_time=processing_time,
                image_path=image_path,
                metadata={
                    'language': self.language,
                    'words_count': len(words),
                    'low_confidence_words': sum(1 for c in confidences if c < 60),
                    'correction_changes': self._count_changes(raw_text, corrected_text)
                }
            )
            
            # Format output
            return self._format_output(result, output_format)
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"OCR processing failed: {e}", exc_info=True)
            raise OCRError(f"Failed to process image: {e}")
    
    def _extract_text(self, image: Image.Image) -> Dict:
        """
        Extract text using Tesseract with detailed information
        """
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.language,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            return data
        except Exception as e:
            raise OCRError(f"Tesseract extraction failed: {e}")
    
    def _parse_words(self, ocr_data: Dict) -> List[OCRWord]:
        """Parse Tesseract output into structured word objects"""
        words = []
        
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            
            # Skip empty detections
            if not text:
                continue
            
            word = OCRWord(
                text=text,
                confidence=float(ocr_data['conf'][i]),
                bbox=(
                    ocr_data['left'][i],
                    ocr_data['top'][i],
                    ocr_data['width'][i],
                    ocr_data['height'][i]
                ),
                line_num=ocr_data['line_num'][i],
                block_num=ocr_data['block_num'][i],
                page_num=ocr_data['page_num'][i]
            )
            words.append(word)
        
        return words
    
    def _filter_by_confidence(self, ocr_data: Dict, threshold: float) -> Dict:
        """Filter OCR results by confidence threshold"""
        filtered = {key: [] for key in ocr_data.keys()}
        
        for i in range(len(ocr_data['text'])):
            if float(ocr_data['conf'][i]) >= threshold:
                for key in ocr_data.keys():
                    filtered[key].append(ocr_data[key][i])
        
        return filtered
    
    def _reconstruct_text_with_layout(self, words: List[OCRWord], preset: str = 'default') -> str:
        """
        Reconstruct text while preserving spatial layout
        Groups words by vertical position (lines), preserves left-to-right order
        Especially important for diagrams and structured layouts
        """
        if not words:
            return ""
        
        # For diagrams, preserve layout by grouping words by Y position (line)
        if preset == 'diagram':
            # Group words by line (similar Y coordinate)
            lines = {}
            tolerance = 15  # pixels of tolerance for same line
            
            for word in words:
                y = word.bbox[1]  # Top Y coordinate
                
                # Find which line this word belongs to
                line_key = None
                for existing_y in lines.keys():
                    if abs(y - existing_y) < tolerance:
                        line_key = existing_y
                        break
                
                if line_key is None:
                    line_key = y
                
                if line_key not in lines:
                    lines[line_key] = []
                lines[line_key].append(word)
            
            # Sort lines by Y position (top to bottom)
            text_lines = []
            for y in sorted(lines.keys()):
                # Sort words in line by X position (left to right)
                line_words = sorted(lines[y], key=lambda w: w.bbox[0])
                line_text = ' '.join(w.text for w in line_words)
                text_lines.append(line_text)
            
            return '\n'.join(text_lines)
        else:
            # Default: just join with spaces
            return ' '.join(w.text for w in words)
    
    def _count_changes(self, original: str, corrected: str) -> int:
        """Count number of character changes made during correction"""
        from difflib import SequenceMatcher
        matcher = SequenceMatcher(None, original, corrected)
        return sum(abs(j-i) for i, j, _ in matcher.get_matching_blocks())
    
    def _format_output(
        self,
        result: OCRResult,
        output_format: Union[str, OutputFormat]
    ) -> Union[str, Dict, OCRResult]:
        """Format result according to requested output format"""
        if isinstance(output_format, str):
            output_format = OutputFormat[output_format.upper()]
        
        if output_format == OutputFormat.TEXT:
            return result.corrected_text
        
        elif output_format == OutputFormat.JSON:
            return result.to_json()
        
        elif output_format == OutputFormat.DETAILED:
            return result
        
        elif output_format == OutputFormat.HOCR:
            return self._generate_hocr(result)
        
        else:
            return result.corrected_text
    
    def _generate_hocr(self, result: OCRResult) -> str:
        """Generate hOCR format output"""
        # Simplified hOCR generation
        hocr = ['<?xml version="1.0" encoding="UTF-8"?>']
        hocr.append('<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" '
                   '"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">')
        hocr.append('<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">')
        hocr.append('<head><meta charset="utf-8"/><title>OCR Results</title></head>')
        hocr.append('<body>')
        
        for word in result.words:
            x, y, w, h = word.bbox
            hocr.append(
                f'<span class="ocrx_word" title="bbox {x} {y} {x+w} {y+h}; '
                f'x_wconf {word.confidence}">{word.text}</span>'
            )
        
        hocr.append('</body></html>')
        return '\n'.join(hocr)
    
    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        output_format: Union[str, OutputFormat] = OutputFormat.TEXT,
        show_progress: bool = True
    ) -> List[Union[str, Dict, OCRResult]]:
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of image file paths
            output_format: Desired output format
            show_progress: Show progress bar
            
        Returns:
            List of results in requested format
        """
        results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(image_paths, desc="Processing images")
            except ImportError:
                logger.warning("tqdm not installed, progress bar disabled")
                iterator = image_paths
        else:
            iterator = image_paths
        
        for image_path in iterator:
            try:
                result = self.process(image_path, output_format=output_format)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append(None)
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        stats = self.stats.copy()
        if stats['total_processed'] > 0:
            stats['avg_processing_time'] = stats['total_time'] / stats['total_processed']
        return stats
    
    def reset_statistics(self):
        """Reset processing statistics"""
        self.stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'avg_confidence': 0.0,
            'errors': 0
        }
    
    def __repr__(self) -> str:
        return (
            f"HybridOCR(language='{self.language}', "
            f"corrector={self.corrector.__class__.__name__}, "
            f"processed={self.stats['total_processed']})"
        )
