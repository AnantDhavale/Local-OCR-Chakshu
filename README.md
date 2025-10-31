# Local-OCR-Chakshu
An hybrid OCR that works absolutely locally with zero cost and no GPUs.
# Hybrid OCR 🔍

100% Local OCR with Intelligent Correction**

Combines Tesseract's battle-tested OCR with smart text correction - all running locally on your machine. No cloud services, no GPU required, no external API calls.

## Why Hybrid OCR?

Fully Local: Runs on CPU, no internet needed
Smart Correction: Fixes common OCR errors automatically  
No GPU Required: Works on any laptop or desktop
Privacy First: Your documents never leave your machine
Open Source: MIT licensed, fully transparent

Quick Start

\`\`\`bash
# Install
pip install hybrid-ocr

# Use
from hybrid_ocr import HybridOCR

ocr = HybridOCR()
result = ocr.process("document.png")
print(result['corrected_text'])
\`\`\`

## Installation

See [docs/installation.md](docs/installation.md) for detailed instructions.

## Features

Rule-based correction (zero dependencies, instant)
Optional LLM correction (better quality, still local)
Multi-language support (via Tesseract)
Batch processing 
CLI tool for quick tasks
Python API for integration

## Documentation

- [Installation Guide](docs/installation.md)
- [Usage Examples](docs/usage.md)
- [API Reference](docs/api.md)
- [Architecture](docs/architecture.md)

## Contributing

See [CONTRIBUTING.md](docs/contributing.md)

## License

MIT License - see LICENSE file
