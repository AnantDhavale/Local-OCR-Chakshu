# Chakshu OCR - Hybrid OCR System

## Project Overview

Chakshu is a hybrid OCR (Optical Character Recognition) system that runs 100% locally with zero cost and no GPU requirements. It combines Tesseract OCR with intelligent text correction for high-quality text extraction from images.

**Key Features:**
- 100% local processing - no cloud services or API calls
- No GPU required - runs on any CPU
- Privacy-first - documents never leave your machine
- Fast processing - 3-5 seconds per page
- Multiple correction strategies (rule-based, LLM, hybrid)
- Multi-language support via Tesseract
- **NEW: Dictionary-based spell checking**
- **NEW: Character confusion pattern recognition**
- **NEW: Adaptive binarization**
- **NEW: Preprocessing presets**
- **NEW: 5x faster deskew performance**

## Project Structure

```
├── hybrid_ocr/          # Main OCR package
│   ├── __init__.py     # Package exports
│   ├── ocr.py          # Core OCR engine
│   ├── corrector.py    # Text correction strategies
│   ├── utils.py        # Helper utilities
│   └── exceptions.py   # Custom exceptions
├── examples/           # Usage examples
├── tests/              # Unit tests
├── dictionaries/       # Language dictionaries
├── app.py             # Streamlit web demo
└── requirements.txt   # Python dependencies
```

## Technology Stack

- **Python 3.11**: Core language
- **Tesseract OCR**: Text extraction engine
- **Pillow**: Image processing
- **NumPy**: Numerical operations for adaptive binarization
- **Streamlit**: Web demo interface

## Current State (v0.2.0 - Major Improvements)

The project has been significantly enhanced with production-grade features:

### ✨ What's New in v0.2.0

1. **Dictionary-Based Spell Checking**
   - Loads language-specific dictionaries from `dictionaries/` folder
   - Confidence-based filtering (only corrects low-confidence words)
   - Uses fuzzy matching to find closest dictionary words
   - Falls back to common words if dictionary unavailable

2. **Character Confusion Patterns**
   - Fixes common OCR mistakes: rn→m, vv→w, cl→d, ii→u
   - Number/letter confusion detection: 0/O, 1/l/I, 5/S, 8/B
   - Context-aware fixes for numbers vs letters
   - Handles special character confusions

3. **5x Faster Deskew** ⚡
   - **Before**: 5 OCR calls to find best rotation (slow!)
   - **After**: Single OSD call to detect angle (5x faster!)
   - Only rotates if angle is significant (>0.5 degrees)
   - Dramatically improved performance

4. **Adaptive Binarization** 🎨
   - **Before**: Fixed threshold of 128 (poor for varying lighting)
   - **After**: Otsu's method calculates optimal threshold per image
   - Adapts to image lighting conditions automatically
   - Much better results for varied document types

5. **Preprocessing Presets** 🎯
   - `default`: Balanced for general use
   - `document`: Optimized for clean documents (sharpen + binarize)
   - `photo`: Optimized for text in photos (denoise, no binarize)
   - `low_quality`: For poor scans (denoise + sharpen + binarize)
   - `newspaper`: For old newspapers (denoise + binarize)
   - `minimal`: No preprocessing (trust Tesseract)

6. **Confidence-Based Filtering**
   - Only corrects words below 70% confidence
   - Preserves high-confidence text untouched
   - Prevents over-correction of correct words
   - More accurate overall results

7. **Improved LLM Corrector**
   - Processes sentence-by-sentence instead of full text
   - Better context preservation
   - More accurate corrections
   - Still 100% local on CPU

## How to Use

### Web Interface (Recommended)

The Streamlit demo app is running on port 5000. Features:
1. Upload an image containing text
2. Select correction strategy (rule-based or hybrid)
3. Choose preprocessing preset based on document type
4. Select language (English, Spanish, French, German)
5. Adjust confidence threshold
6. Click "Extract Text" to process

### Python API

```python
from hybrid_ocr import HybridOCR

# Basic usage with new features
ocr = HybridOCR(
    correction_strategy='rule_based',
    preprocess_preset='document',  # NEW: Use document preset
    language='eng'
)

# Process image
result = ocr.process("document.png")
print(result)  # Returns corrected text

# Get detailed results
result = ocr.process("document.png", output_format='detailed')
print(f"Confidence: {result.confidence_avg:.1f}%")
print(f"Text: {result.corrected_text}")
```

### Advanced Usage

```python
# Use low_quality preset for poor scans
ocr_poor = HybridOCR(preprocess_preset='low_quality')

# Use photo preset for text in images
ocr_photo = HybridOCR(preprocess_preset='photo')

# Spanish language with hybrid correction
ocr_spanish = HybridOCR(
    language='spa',
    correction_strategy='hybrid'
)
```

## Correction Strategies

1. **Rule-Based** (Recommended for most users)
   - Dictionary-based spell checking
   - Character confusion pattern fixes
   - Confidence-based filtering
   - Instant processing
   - No additional dependencies

2. **Hybrid**
   - Combines rule-based + LLM
   - Fast rule-based for high-confidence text
   - LLM correction for low-confidence sections
   - Requires: `pip install transformers torch`

3. **LLM**
   - Uses small local language model (80MB)
   - Better quality correction
   - Runs on CPU
   - Requires: `pip install transformers torch`

## Recent Changes

- **2025-11-03 v0.2.0**: Major feature release
  - Implemented dictionary-based spell checking
  - Added character confusion pattern recognition
  - Optimized deskew (5x performance improvement)
  - Implemented adaptive binarization (Otsu's method)
  - Added preprocessing presets for different document types
  - Implemented confidence-based correction filtering
  - Improved LLM corrector to process sentence-by-sentence
  - Fixed language propagation throughout pipeline
  - Updated Streamlit UI to expose all new features

- **2025-11-03 v0.1.0**: Initial Replit setup
  - Created missing Python module files
  - Installed all dependencies
  - Created Streamlit web demo on port 5000
  - Verified workflow runs successfully

## Performance Improvements

| Feature | Before | After | Improvement |
|---------|--------|-------|-------------|
| Deskew | 5 OCR calls | 1 OSD call | 5x faster |
| Binarization | Fixed threshold | Adaptive (Otsu) | Better quality |
| Correction | 6 hardcoded rules | Dictionary + patterns | Much more accurate |
| Language Support | English only (broken) | Full multilingual | Fixed |

## Known Issues & Limitations

1. **Tesseract Required**: System dependency must be available
2. **LLM Features**: Optional LLM corrector requires additional packages
3. **Language Packs**: Additional Tesseract language packs needed for non-English languages
4. **Image Quality**: Works best with clear, high-resolution images
5. **Dictionaries**: Only English dictionary is populated (Spanish, French, German need word lists)

## Development Notes

### Adding Language Support

To use other languages:
```python
ocr = HybridOCR(language='spa')  # Spanish
ocr = HybridOCR(language='fra')  # French
ocr = HybridOCR(language='deu')  # German
```

Note: For best results with non-English languages, add word lists to `dictionaries/es_common.txt`, `dictionaries/fr_common.txt`, etc.

### Testing

```bash
# Run all tests
pytest tests/test_ocr.py -v

# Run specific test
pytest tests/test_ocr.py::TestHybridOCR::test_process_image
```

## User Preferences

None documented yet.

## Architecture Decisions

1. **Package Naming**: Using `hybrid_ocr` instead of `ocr` to match PyPI package name
2. **Modular Design**: Separated concerns into ocr.py, corrector.py, utils.py, exceptions.py
3. **Strategy Pattern**: Correction strategies use base class for extensibility
4. **Web Demo**: Streamlit chosen for quick, user-friendly interface
5. **Local-First**: All processing happens locally, no external API dependencies
6. **Performance-First**: Optimized deskew and adaptive algorithms for speed
7. **Language Normalization**: Maps Tesseract 3-letter codes to 2-letter dictionary codes

## Dependencies

### System
- Tesseract OCR

### Python (requirements.txt)
- pytesseract>=0.3.10
- Pillow>=9.0.0
- numpy (for adaptive binarization)
- streamlit (for web demo)

### Optional (for LLM correction)
- transformers>=4.30.0
- torch>=2.0.0
- sentencepiece>=0.1.99

## License

MIT License - see LICENSE file
