"""
Helper functions for the Chakshu OCR system.
"""

import logging
import time
from functools import wraps
from typing import Optional, Callable
from pathlib import Path

from PIL import Image


def setup_logging(level: str = "INFO"):
    """
    Configure logging for the application
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def measure_time(func: Callable) -> Callable:
    """
    Decorator to measure function execution time
    
    Usage:
        @measure_time
        def my_function():
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        logger = logging.getLogger(func.__module__)
        logger.debug(f"{func.__name__} completed in {elapsed_time:.3f}s")
        
        return result
    
    return wrapper


def validate_image(image: Image.Image) -> bool:
    """
    Validate that image is suitable for OCR
    
    Args:
        image: PIL Image object
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If image is invalid
    """
    if image is None:
        raise ValueError("Image is None")
    
    if image.size[0] < 10 or image.size[1] < 10:
        raise ValueError(f"Image too small: {image.size}")
    
    if image.size[0] > 10000 or image.size[1] > 10000:
        raise ValueError(f"Image too large: {image.size}")
    
    return True


def ensure_directory(path: Path) -> Path:
    """
    Ensure directory exists, create if needed
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_supported_image_formats() -> list:
    """Return list of supported image formats"""
    return ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif']


def is_supported_image(file_path: str) -> bool:
    """
    Check if file is a supported image format
    
    Args:
        file_path: Path to file
        
    Returns:
        True if supported
    """
    path = Path(file_path)
    return path.suffix.lower() in get_supported_image_formats()


def format_confidence(confidence: float) -> str:
    """
    Format confidence score for display
    
    Args:
        confidence: Confidence score (0-100)
        
    Returns:
        Formatted string with emoji
    """
    if confidence >= 90:
        return f"🟢 {confidence:.1f}% (Excellent)"
    elif confidence >= 75:
        return f"🟡 {confidence:.1f}% (Good)"
    elif confidence >= 60:
        return f"🟠 {confidence:.1f}% (Fair)"
    else:
        return f"🔴 {confidence:.1f}% (Poor)"


def chunk_text(text: str, max_length: int = 512) -> list:
    """
    Split text into chunks for processing
    
    Args:
        text: Text to split
        max_length: Maximum chunk length
        
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        
        if current_length + word_length > max_length:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def calculate_word_accuracy(predicted: str, ground_truth: str) -> float:
    """
    Calculate word-level accuracy
    
    Args:
        predicted: Predicted text
        ground_truth: Ground truth text
        
    Returns:
        Accuracy percentage (0-100)
    """
    pred_words = set(predicted.lower().split())
    truth_words = set(ground_truth.lower().split())
    
    if not truth_words:
        return 0.0
    
    correct = len(pred_words & truth_words)
    total = len(truth_words)
    
    return (correct / total) * 100


def calculate_character_accuracy(predicted: str, ground_truth: str) -> float:
    """
    Calculate character-level accuracy using edit distance
    
    Args:
        predicted: Predicted text
        ground_truth: Ground truth text
        
    Returns:
        Accuracy percentage (0-100)
    """
    from difflib import SequenceMatcher
    
    similarity = SequenceMatcher(None, predicted, ground_truth).ratio()
    return similarity * 100


def safe_filename(filename: str) -> str:
    """
    Convert string to safe filename
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename string
    """
    import re
    
    # Remove or replace unsafe characters
    safe = re.sub(r'[^\w\s-]', '', filename)
    safe = re.sub(r'[-\s]+', '-', safe)
    
    return safe.strip('-_')


def load_test_image(name: str = "sample") -> Optional[Image.Image]:
    """
    Load a test image from test_images directory
    
    Args:
        name: Image name (without extension)
        
    Returns:
        PIL Image or None if not found
    """
    test_dir = Path(__file__).parent.parent / "tests" / "test_images"
    
    for ext in get_supported_image_formats():
        image_path = test_dir / f"{name}{ext}"
        if image_path.exists():
            return Image.open(image_path)
    
    return None


class Timer:
    """
    Context manager for timing code blocks
    
    Usage:
        with Timer("Processing"):
            # code here
    """
    
    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        self.logger.info(f"{self.name} completed in {self.elapsed:.3f}s")


def retry(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator to retry function on failure
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
                        raise
        
        return wrapper
    return decorator
