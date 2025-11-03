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
- **Streamlit**: Web demo interface
- **NumPy**: Numerical operations

## Current State

The project has been successfully set up in the Replit environment with:

1. **Core OCR Engine**: Fully functional with image preprocessing, text extraction, and correction
2. **Web Demo**: Streamlit app running on port 5000 for easy testing
3. **Dependencies**: All required packages installed
   - Tesseract OCR system package
   - Python packages: pytesseract, Pillow, numpy, streamlit
4. **Package Structure**: Properly organized as `hybrid_ocr` package

## How to Use

### Web Interface (Recommended)

The Streamlit demo app is running on port 5000. Simply:
1. Upload an image containing text
2. Select correction strategy and language
3. Click "Extract Text" to process

### Python API

```python
from hybrid_ocr import HybridOCR

# Initialize OCR engine
ocr = HybridOCR(correction_strategy='rule_based')

# Process image
result = ocr.process("document.png")
print(result)  # Returns corrected text

# Get detailed results
result = ocr.process("document.png", output_format='detailed')
print(f"Confidence: {result.confidence_avg:.1f}%")
print(f"Text: {result.corrected_text}")
```

### CLI Usage

```bash
# Run the web demo
streamlit run app.py --server.port 5000

# Run examples
python examples/basic_usage.py

# Run tests
pytest tests/
```

## Correction Strategies

1. **Rule-Based**: Fast, zero dependencies, instant processing
   - Uses predefined patterns to fix common OCR errors
   - Best for quick processing

2. **Hybrid**: Combines rules + LLM
   - Fast rule-based for high-confidence text
   - LLM correction for low-confidence sections
   - Requires additional packages (transformers, torch)

3. **LLM**: Uses small local language model
   - Better quality correction
   - Runs on CPU (~80MB model)
   - Requires: `pip install transformers torch`

## Recent Changes

- **2025-11-03**: Initial Replit setup
  - Created missing Python module files (exceptions.py, complete __init__.py)
  - Renamed package from 'ocr' to 'hybrid_ocr' for consistency
  - Installed all dependencies (Tesseract, Python packages)
  - Created Streamlit web demo on port 5000
  - Fixed syntax errors in utils.py
  - Verified workflow runs successfully

## Known Issues & Limitations

1. **Tesseract Required**: System dependency must be available
2. **LLM Features**: Optional LLM corrector requires additional packages
3. **Language Packs**: Additional Tesseract language packs needed for non-English languages
4. **Image Quality**: Works best with clear, high-resolution images

## Development Notes

### Adding Language Support

To use other languages:
```python
ocr = HybridOCR(language='spa')  # Spanish
ocr = HybridOCR(language='fra')  # French
```

Ensure Tesseract language packs are installed for the language.

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

## Dependencies

### System
- Tesseract OCR

### Python (requirements.txt)
- pytesseract>=0.3.10
- Pillow>=9.0.0
- numpy
- streamlit (for web demo)

### Optional (for LLM correction)
- transformers>=4.30.0
- torch>=2.0.0
- sentencepiece>=0.1.99

## License

MIT License - see LICENSE file
