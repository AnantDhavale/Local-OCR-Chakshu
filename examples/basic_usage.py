#!/usr/bin/env python3
"""
Basic usage example for Hybrid OCR
"""

from hybrid_ocr import HybridOCR

# Example 1: Simple usage (rule-based, instant)
print("Example 1: Basic OCR with rule-based correction")
ocr = HybridOCR(use_llm=False)
result = ocr.process("test_image.png")
print(f"Corrected text: {result['corrected_text']}")
print(f"Confidence: {result['metadata']['avg_confidence']:.1f}%")
print()

# Example 2: With LLM correction (better quality, still local)
print("Example 2: OCR with local LLM correction")
ocr_llm = HybridOCR(use_llm=True)  # Downloads ~80MB model first time
result = ocr_llm.process("test_image.png")
print(f"Corrected text: {result['corrected_text']}")
print()

# Example 3: Get detailed information
print("Example 3: Detailed output with word-level data")
result = ocr.process("test_image.png", output_format="detailed")
print(f"Total words: {result['metadata']['words_processed']}")
print(f"Low confidence words: {result['metadata']['low_confidence_words']}")
print(f"Text changed by: {result['metadata']['change_percentage']}%")
