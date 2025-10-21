#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
"""
Test deterministic inference on REAL GPT-2 model from Hugging Face.
"""

import torch
import sys

print("="*80)
print("TESTING REAL GPT-2 MODEL")
print("="*80)

# Check if transformers is installed
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print("✓ transformers library available")
except ImportError:
    print("✗ transformers not installed")
    print("Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers"])
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print("✓ transformers installed")

from deterministic_llm.inference import DeterministicInferenceEngine

print("\nLoading GPT-2 model (this may take a moment)...")
torch.manual_seed(42)

# Load smallest GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

print(f"✓ Model loaded: GPT-2 ({sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters)")

# Create deterministic engine
print("\nCreating DeterministicInferenceEngine...")
engine = DeterministicInferenceEngine(model, patch_model=True)
print("✓ Engine created with patched model")

# Test 1: Determinism across multiple runs
print("\n" + "="*80)
print("Test 1: Multiple runs with same input (determinism)")
print("="*80)

text = "The quick brown fox"
print(f"Input text: '{text}'")

input_ids = tokenizer.encode(text, return_tensors="pt")
print(f"Input shape: {input_ids.shape}")

outputs = []
for i in range(10):
    with torch.no_grad():
        output = engine.forward(input_ids)
    # Get logits
    if hasattr(output, 'logits'):
        logits = output.logits
    else:
        logits = output[0] if isinstance(output, tuple) else output
    outputs.append(logits.clone())

all_identical = all(torch.equal(outputs[0], out) for out in outputs[1:])
max_diff = max(torch.abs(outputs[0] - out).max().item() for out in outputs[1:])

print(f"\nAll runs bitwise identical: {all_identical}")
print(f"Max difference: {max_diff:.2e}")

if all_identical and max_diff == 0.0:
    print("✓ PASS: 100% deterministic across runs")
    test1_pass = True
else:
    print("✗ FAIL: Not deterministic")
    test1_pass = False

# Test 2: Batch size invariance
print("\n" + "="*80)
print("Test 2: Batch size invariance")
print("="*80)

# Reference with batch size 1
torch.manual_seed(100)
text_ref = "Hello world"
input_ids_ref = tokenizer.encode(text_ref, return_tensors="pt")

with torch.no_grad():
    output_ref = engine.forward(input_ids_ref)
logits_ref = output_ref.logits if hasattr(output_ref, 'logits') else output_ref

print(f"\nTesting different batch sizes with same first input:")
batch_sizes = [1, 2, 4, 8, 16]
all_match = True

for bs in batch_sizes:
    # Create batch with same first element
    torch.manual_seed(100)
    input_ids_batch = tokenizer.encode(text_ref, return_tensors="pt")
    if bs > 1:
        # Add more samples
        for _ in range(bs - 1):
            extra = tokenizer.encode("Different text", return_tensors="pt")
            # Pad to same length
            if extra.shape[1] < input_ids_batch.shape[1]:
                extra = torch.nn.functional.pad(extra, (0, input_ids_batch.shape[1] - extra.shape[1]))
            elif extra.shape[1] > input_ids_batch.shape[1]:
                input_ids_batch = torch.nn.functional.pad(input_ids_batch, (0, extra.shape[1] - input_ids_batch.shape[1]))
            input_ids_batch = torch.cat([input_ids_batch, extra], dim=0)

    with torch.no_grad():
        output_batch = engine.forward(input_ids_batch)
    logits_batch = output_batch.logits if hasattr(output_batch, 'logits') else output_batch

    # Compare first element
    diff = torch.abs(logits_ref[0] - logits_batch[0]).max().item()
    identical = torch.equal(logits_ref[0], logits_batch[0])

    status = "✓" if identical else "✗"
    print(f"  {status} Batch size {bs:2d}: diff = {diff:.2e}, identical = {identical}")

    if not identical:
        all_match = False

test2_pass = all_match
print(f"\nAll batch sizes match: {all_match}")

# Test 3: Text generation determinism
print("\n" + "="*80)
print("Test 3: Text generation determinism")
print("="*80)

prompt = "Once upon a time"
print(f"Prompt: '{prompt}'")

# Generate 5 times
generations = []
for i in range(5):
    torch.manual_seed(42)  # Same seed each time
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        # Greedy generation
        gen_ids = engine.generate(input_ids, max_length=20, temperature=0.0)

    generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    generations.append(generated_text)
    print(f"  Generation {i+1}: {generated_text[:50]}...")

# Check if all identical
all_same = all(g == generations[0] for g in generations[1:])
test3_pass = all_same

if all_same:
    print("\n✓ PASS: All generations identical")
else:
    print("\n✗ FAIL: Generations differ")
    for i, gen in enumerate(generations):
        print(f"  {i+1}: {gen}")

# Final summary
print("\n" + "="*80)
print("FINAL RESULTS - REAL GPT-2 MODEL")
print("="*80)

print(f"\nTest 1 (Determinism across runs):     {'✓ PASS' if test1_pass else '✗ FAIL'}")
print(f"Test 2 (Batch size invariance):       {'✓ PASS' if test2_pass else '✗ FAIL'}")
print(f"Test 3 (Generation determinism):      {'✓ PASS' if test3_pass else '✗ FAIL'}")

all_pass = test1_pass and test2_pass and test3_pass

print("\n" + "="*80)
if all_pass:
    print("✓✓✓ SUCCESS: GPT-2 IS 100% DETERMINISTIC ✓✓✓")
    print("Real production LLM verified!")
else:
    print("✗✗✗ FAILED: GPT-2 is not fully deterministic ✗✗✗")
print("="*80)

sys.exit(0 if all_pass else 1)
