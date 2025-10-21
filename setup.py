#!/usr/bin/env python3
"""
Setup script for deterministic-llm package.

Install with: pip install -e .
"""

from setuptools import setup, find_packages

# Read README for long description
try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "100% deterministic inference for language models"

setup(
    name="deterministic-llm",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="100% deterministic inference for language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/deterministic-llm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    keywords=[
        "llm",
        "deterministic",
        "inference",
        "language-model",
        "pytorch",
        "transformers",
        "batch-invariant",
        "reproducible",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/deterministic-llm/issues",
        "Source": "https://github.com/yourusername/deterministic-llm",
        "Documentation": "https://github.com/yourusername/deterministic-llm/blob/main/USAGE_GUIDE.md",
    },
)
