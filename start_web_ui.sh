#!/bin/bash
# Quick start script for the web UI

echo "=================================="
echo "Deterministic LLM Inference Web UI"
echo "=================================="
echo ""

# Function to check if in virtual environment
in_venv() {
    python -c "import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)"
}

# Check if in virtual environment
if ! in_venv; then
    echo "⚠️  Not in a virtual environment!"
    echo ""
    echo "To avoid 'externally-managed-environment' error, please:"
    echo "1. Create virtual environment:"
    echo "   python3 -m venv venv"
    echo "2. Activate it:"
    echo "   source venv/bin/activate"
    echo "3. Run this script again"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if gradio is installed
if ! python -c "import gradio" 2>/dev/null; then
    echo "⚠️  Gradio not found. Installing..."
    pip install gradio || pip install --user gradio
    echo ""
fi

# Check if torch is installed
if ! python -c "import torch" 2>/dev/null; then
    echo "⚠️  PyTorch not found. Installing..."
    pip install torch transformers || pip install --user torch transformers
    echo ""
fi

echo "Starting web UI..."
echo ""
python web_ui.py
