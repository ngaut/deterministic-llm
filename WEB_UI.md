# Web UI Guide

**Interactive web interface for deterministic LLM inference**

![Status](https://img.shields.io/badge/status-ready-brightgreen)
![Interface](https://img.shields.io/badge/interface-Gradio-orange)

---

## Quick Start

### 1. Install Dependencies

**⚠️ Note**: If you get "externally-managed-environment" error, use a virtual environment:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch transformers gradio
```

**Or** install for user only:

```bash
pip install --user torch transformers gradio
```

### 2. Start the Web UI

```bash
python web_ui.py
```

**Or use the quick start script:**

```bash
./start_web_ui.sh
```

### 3. Open in Browser

The interface will automatically open at: **http://localhost:7860**

---

## Features

The web UI provides 4 main tabs:

### 📝 Text Generation

**Generate deterministic text with any model**

- Enter a prompt
- Select model (GPT-2 variants)
- Set max length
- Run multiple times to verify determinism
- View top-5 next token predictions

**Example:**
- Prompt: "Once upon a time"
- Model: GPT-2
- Max Length: 50
- Number of Runs: 3

All 3 runs will produce **identical** text!

---

### 🔬 Determinism Test

**Verify that inference is truly deterministic**

- Enter a test prompt
- Select model
- Run 10-100 times
- Check that all outputs are identical

**What it shows:**
- ✅ DETERMINISTIC or ❌ NON-DETERMINISTIC
- Max difference between runs (should be 0.0)
- Detailed test results

---

### 📊 Batch Invariance Test

**Verify single vs batch processing gives same result**

- Enter a test prompt
- Select model
- Compares:
  - Single input (batch size 1)
  - Batch input (batch size 4)

**What it shows:**
- ✅ BATCH-INVARIANT or ❌ NOT BATCH-INVARIANT
- Max difference (should be 0.0)

---

### 📚 Documentation

**Built-in documentation and help**

- How to use each tab
- Features and limitations
- Installation instructions
- Links to full documentation

---

## Supported Models

The web UI supports any HuggingFace model, but comes pre-configured with:

- **gpt2** (124M params) - Fast, verified working
- **Qwen/Qwen3-0.6B** (600M params) - Modern architecture, verified working

**To add your own model:**

Edit the dropdown choices in `web_ui.py`:

```python
gen_model = gr.Dropdown(
    label="Model",
    choices=["gpt2", "Qwen/Qwen3-0.6B", "your-model-name"],
    value="gpt2"
)
```

---

## Screenshots

### Text Generation Tab

```
┌─────────────────────────────────────────────────────────┐
│ Prompt: "Once upon a time"                              │
│ Model: GPT-2                                            │
│ Max Length: 50                                          │
│ Number of Runs: 3                                       │
│                                                         │
│ [Generate] ←─ Click to generate                        │
└─────────────────────────────────────────────────────────┘

Output:
════════════════════════════════════════════════════════════
MODEL: gpt2
PROMPT: Once upon a time
MAX LENGTH: 50
RUNS: 3
════════════════════════════════════════════════════════════

DETERMINISM CHECK: ✅ PASS

Run 1:
────────────────────────────────────────────────────────────
Once upon a time, the world was a place of great beauty...

Run 2:
────────────────────────────────────────────────────────────
Once upon a time, the world was a place of great beauty...

Run 3:
────────────────────────────────────────────────────────────
Once upon a time, the world was a place of great beauty...
```

---

## Usage Tips

### Best Practices

1. **Start with small models** (gpt2) for testing
2. **Use "Number of Runs" > 1** to verify determinism
3. **Enable "Show logits"** to understand predictions
4. **Test with your own prompts** to see deterministic behavior

### Performance

- First generation is slower (model loading)
- Subsequent generations are faster (model cached)
- GPT-2: ~2-3 seconds per generation
- GPT-2-medium: ~5-10 seconds per generation

### Troubleshooting

**Problem**: "Model not found"
- **Solution**: Check model name is correct
- Try pre-configured models first (gpt2, gpt2-medium)

**Problem**: "Out of memory"
- **Solution**: Use smaller model (distilgpt2)
- Reduce max length

**Problem**: "Slow generation"
- **Solution**: This is expected (~40-60% slower than standard)
- Use smaller models for faster testing

---

## Advanced Usage

### Custom Configuration

Edit `web_ui.py` to customize:

```python
# Change port
demo.launch(server_port=8080)

# Enable sharing (creates public URL)
demo.launch(share=True)

# Add authentication
demo.launch(auth=("username", "password"))
```

### Adding New Features

The web UI is built with Gradio, making it easy to add new features:

```python
# Add a new tab
with gr.Tab("My Feature"):
    # Your UI components here
    pass
```

See [Gradio documentation](https://gradio.app/docs) for more.

---

## API vs Web UI

| Feature | Web UI | Python API |
|---------|--------|------------|
| **Ease of use** | ✅ Very easy | Moderate |
| **Interactive** | ✅ Yes | No |
| **Visualization** | ✅ Yes | No |
| **Batch processing** | Limited | ✅ Full support |
| **Customization** | Limited | ✅ Full control |
| **Production use** | No | ✅ Yes |

**Recommendation:**
- Use **Web UI** for: Testing, demos, exploration
- Use **Python API** for: Production, automation, batch processing

---

## Examples

### Example 1: Quick Test

1. Open web UI
2. Go to "Text Generation" tab
3. Enter: "The quick brown fox"
4. Set runs to 3
5. Click "Generate"
6. Verify all 3 outputs are identical ✅

### Example 2: Determinism Verification

1. Go to "Determinism Test" tab
2. Enter: "Hello world"
3. Set runs to 100
4. Click "Test Determinism"
5. Check: Max difference = 0.00e+00 ✅

### Example 3: Batch Invariance

1. Go to "Batch Invariance Test" tab
2. Enter: "Test prompt"
3. Click "Test Batch Invariance"
4. Verify: BATCH-INVARIANT ✅

---

## Limitations

- **Greedy decoding only** (temperature=0.0)
- **No sampling** (temperature > 0 not supported)
- **CPU only** (GPU not tested)
- **Single user** (not optimized for concurrent users)
- **Model caching** (first load is slow)

---

## FAQ

**Q: Can I use this in production?**

A: The web UI is for testing/demos. Use the Python API for production.

**Q: Can I add my own model?**

A: Yes! Edit the model dropdown in `web_ui.py`

**Q: Why is it slow?**

A: Deterministic inference is ~40-60% slower than standard inference. This is the trade-off for determinism.

**Q: Can I run on GPU?**

A: GPU support is not tested. The implementation may work but needs validation.

**Q: How do I share the interface?**

A: Launch with `share=True` in `web_ui.py` to get a public URL

---

## Next Steps

1. ✅ Try the web UI: `python web_ui.py`
2. ✅ Test with different prompts
3. ✅ Verify determinism with multiple runs
4. ✅ Read the [Usage Guide](docs/USAGE_GUIDE.md)
5. ✅ Try the [Python API](examples/example_simple.py)

---

## Support

- **Documentation**: See `docs/` folder
- **Examples**: See `examples/` folder
- **Tests**: See `tests/` folder
- **Issues**: Open an issue on GitHub

---

**Enjoy deterministic LLM inference! 🎯**
