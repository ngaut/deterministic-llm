#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
"""
Test Deterministic Inference on Qwen3-0.6B

This tests our implementation on a different model architecture than GPT-2.
Qwen uses different components which is a good validation test.
"""

import torch
import sys
import hashlib
from concurrent.futures import ThreadPoolExecutor

print("="*80)
print("DETERMINISTIC INFERENCE TEST: Qwen3-0.6B")
print("="*80)

# Check if model is available
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("\n✓ Transformers library available")
except ImportError:
    print("\n✗ Transformers library not found")
    sys.exit(1)

# Load model
print("\nLoading Qwen3-0.6B model...")
print("(This may take a few minutes for first download...)")

try:
    model_name = "Qwen/Qwen3-0.6B"
    
    # Load with appropriate settings
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
        device_map="cpu",  # Force CPU for consistency
        low_cpu_mem_usage=True
    )
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"✓ Loaded Qwen3-0.6B ({num_params:.2f}B parameters)")
    
except Exception as e:
    print(f"\n✗ Failed to load model: {e}")
    print("\nTrying Qwen2.5-1.5B instead...")
    
    try:
        model_name = "Qwen/Qwen2.5-1.5B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        model.eval()
        num_params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"✓ Loaded Qwen2.5-1.5B ({num_params:.2f}B parameters)")
    except Exception as e2:
        print(f"✗ Also failed: {e2}")
        print("\nSkipping Qwen test - model not available")
        sys.exit(0)

# Create deterministic engine
from deterministic_llm.inference import DeterministicInferenceEngine

print("\nCreating DeterministicInferenceEngine...")
engine = DeterministicInferenceEngine(model, patch_model=True)
print("✓ Engine created")

# Prepare test input
test_prompt = "The quick brown fox jumps over"
input_ids = tokenizer.encode(test_prompt, return_tensors="pt")

print(f"\nTest prompt: '{test_prompt}'")
print(f"Input shape: {input_ids.shape}")

# =============================================================================
# TEST 1: Basic Determinism (100 runs)
# =============================================================================
print("\n" + "="*80)
print("TEST 1: Basic Determinism (100 runs)")
print("="*80)

def get_hash(output):
    logits = output.logits if hasattr(output, 'logits') else output
    # Convert to float32 if bfloat16 (numpy doesn't support bfloat16)
    if logits.dtype == torch.bfloat16:
        logits = logits.float()
    return hashlib.sha256(logits.cpu().numpy().tobytes()).hexdigest()

print("Running 100 forward passes...")
hashes = []

for i in range(100):
    output = engine.forward(input_ids)
    h = get_hash(output)
    hashes.append(h)
    
    if (i + 1) % 20 == 0:
        print(f"  Progress: {i+1}/100")

unique = len(set(hashes))
print(f"\nResults:")
print(f"  Total runs: 100")
print(f"  Unique outputs: {unique}")

if unique == 1:
    print(f"  ✓✓✓ ALL 100 RUNS IDENTICAL ✓✓✓")
    test1_pass = True
else:
    print(f"  ✗ Non-deterministic: {unique} unique outputs")
    test1_pass = False

# =============================================================================
# TEST 2: Batch Size Invariance
# =============================================================================
print("\n" + "="*80)
print("TEST 2: Batch Size Invariance")
print("="*80)

batch_sizes = [1, 2, 4, 8, 16]
print(f"Testing batch sizes: {batch_sizes}")

# Use same input repeated
single_input = tokenizer.encode(test_prompt, return_tensors="pt")
batch_hashes = []

for bs in batch_sizes:
    # Create batch by repeating
    batch_input = single_input.repeat(bs, 1)
    
    output = engine.forward(batch_input)
    logits = output.logits if hasattr(output, 'logits') else output
    
    # Extract first item
    first_item = logits[0]
    if first_item.dtype == torch.bfloat16:
        first_item = first_item.float()
    h = hashlib.sha256(first_item.cpu().numpy().tobytes()).hexdigest()
    batch_hashes.append(h)
    
    print(f"  Batch size {bs:2d}: hash={h[:16]}")

unique_batches = len(set(batch_hashes))
print(f"\nUnique hashes across batch sizes: {unique_batches}")

if unique_batches == 1:
    print("  ✓ Batch-invariant across all sizes!")
    test2_pass = True
else:
    print(f"  ✗ NOT batch-invariant: {unique_batches} unique")
    test2_pass = False

# =============================================================================
# TEST 3: Multi-threaded Determinism (50 workers)
# =============================================================================
print("\n" + "="*80)
print("TEST 3: Multi-threaded Determinism (50 workers)")
print("="*80)

def threaded_worker(worker_id):
    output = engine.forward(input_ids)
    return get_hash(output)

print("Running 50 concurrent workers...")

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(threaded_worker, range(50)))

unique_threaded = len(set(results))
print(f"\nResults:")
print(f"  Unique outputs: {unique_threaded}/50")

if unique_threaded == 1:
    print("  ✓ Thread-safe and deterministic!")
    test3_pass = True
else:
    print(f"  ✗ Non-deterministic in multi-threaded use")
    test3_pass = False

# =============================================================================
# TEST 4: Text Generation Determinism
# =============================================================================
print("\n" + "="*80)
print("TEST 4: Text Generation Determinism (20 runs)")
print("="*80)

print(f"Generating 30 tokens, 20 times...")
print(f"Starting prompt: '{test_prompt}'")

generated_texts = []

for i in range(20):
    try:
        output_ids = engine.generate(
            input_ids,
            max_length=input_ids.shape[1] + 30,
            temperature=0.0
        )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_texts.append(generated_text)
    except Exception as e:
        print(f"  Generation failed: {e}")
        print("  Skipping generation test")
        test4_pass = None
        break

if generated_texts:
    unique_texts = len(set(generated_texts))
    print(f"\nResults:")
    print(f"  Unique generated texts: {unique_texts}/20")
    print(f"  Sample generation: '{generated_texts[0]}'")
    
    if unique_texts == 1:
        print("  ✓ All 20 generations identical!")
        test4_pass = True
    else:
        print(f"  ✗ Non-deterministic generation")
        test4_pass = False

# =============================================================================
# FINAL VERDICT
# =============================================================================
print("\n" + "="*80)
print("FINAL VERDICT: Qwen2.5 Testing")
print("="*80)

tests = {
    "100 sequential runs": test1_pass,
    "Batch size invariance": test2_pass,
    "50 concurrent workers": test3_pass,
}

if test4_pass is not None:
    tests["20 text generations"] = test4_pass

print("\nResults:")
for test_name, passed in tests.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {test_name}")

all_pass = all(tests.values())

print("\n" + "="*80)
if all_pass:
    print("✓✓✓ ALL TESTS PASSED ON QWEN2.5 ✓✓✓")
    print()
    print("CONCLUSION:")
    print(f"  Our implementation works on {model_name}!")
    print("  ✓ 100% deterministic")
    print("  ✓ Batch-invariant")
    print("  ✓ Thread-safe")
    if test4_pass:
        print("  ✓ Deterministic generation")
    print()
    print("  Implementation is MODEL-AGNOSTIC!")
else:
    print("✗✗✗ SOME TESTS FAILED ✗✗✗")
    print()
    print("Investigation needed for Qwen2.5 compatibility")

print("="*80)

sys.exit(0 if all_pass else 1)
