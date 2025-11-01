from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hybrid-ocr",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Local OCR with intelligent correction - no cloud, no GPU needed",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hybrid-ocr",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pytesseract>=0.3.10",
        "Pillow>=9.0.0",
    ],
    extras_require={
        "llm": [
            "transformers>=4.30.0",
            "torch>=2.0.0",
            "sentencepiece>=0.1.99",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hybrid-ocr=hybrid_ocr.cli:main",
        ],
    },
)
