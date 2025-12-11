#!/usr/bin/env python3
"""
Check quantized model files and load correctly
Run this in Kaggle to diagnose and fix loading issues
"""

import os
from pathlib import Path
import json

quantized_path = "artifacts/gptq/llama3-8b-medical-4bit"

print("="*70)
print("CHECKING QUANTIZED MODEL DIRECTORY")
print("="*70)
print()

# Check if directory exists
if not Path(quantized_path).exists():
    print(f"❌ ERROR: Directory not found: {quantized_path}")
    print("   Did quantization complete successfully?")
    exit(1)

print(f"✅ Directory exists: {quantized_path}")
print()

# List all files
print("Files in directory:")
print("-" * 70)
files = sorted(os.listdir(quantized_path))
for f in files:
    size = os.path.getsize(os.path.join(quantized_path, f))
    size_mb = size / (1024**2)
    if size_mb > 1:
        print(f"  📄 {f:<40} ({size_mb:>8.2f} MB)")
    else:
        print(f"  📄 {f:<40} ({size:>8} bytes)")

print()

# Check for required files
print("="*70)
print("CHECKING REQUIRED FILES")
print("="*70)
print()

required_files = {
    "config.json": "Model configuration",
    "quantize_config.json": "Quantization configuration",
    "tokenizer.model": "Tokenizer vocabulary",
    "tokenizer_config.json": "Tokenizer configuration",
}

# GPTQ weight files (different naming patterns)
gptq_weight_patterns = [
    "gptq_model-*.safetensors",
    "model.safetensors",
    "pytorch_model.bin",
    "*.safetensors"
]

missing_files = []
for fname, description in required_files.items():
    fpath = Path(quantized_path) / fname
    if fpath.exists():
        print(f"  ✅ {fname:<30} - {description}")
    else:
        print(f"  ❌ {fname:<30} - {description} (MISSING)")
        missing_files.append(fname)

print()
print("Checking for GPTQ weight files:")

# Look for any safetensors or bin files
weight_files = list(Path(quantized_path).glob("*.safetensors")) + \
               list(Path(quantized_path).glob("*.bin"))

if weight_files:
    print("  ✅ Found weight files:")
    for wf in weight_files:
        size_mb = wf.stat().st_size / (1024**2)
        print(f"     - {wf.name} ({size_mb:.2f} MB)")
else:
    print("  ❌ No weight files found (.safetensors or .bin)")
    missing_files.append("weight files")

print()

# Check quantize_config.json content
if Path(quantized_path, "quantize_config.json").exists():
    print("Quantization Config:")
    with open(Path(quantized_path) / "quantize_config.json", 'r') as f:
        config = json.load(f)
        for k, v in config.items():
            if k != "meta":
                print(f"  {k}: {v}")
    print()

# Diagnosis
print("="*70)
print("DIAGNOSIS")
print("="*70)
print()

if missing_files:
    print("❌ PROBLEM: Missing required files")
    print()
    print("Missing files:", ", ".join(missing_files))
    print()
    print("SOLUTIONS:")
    print()
    print("1. If quantization is still running:")
    print("   - Wait for it to complete")
    print("   - Look for 'Quantization complete!' message")
    print()
    print("2. If quantization failed or was interrupted:")
    print("   - Re-run quantization from scratch")
    print("   - Check for error messages in quantization output")
    print()
    print("3. If quantization completed but files are missing:")
    print("   - Check if files are in a different location")
    print("   - Look for output directory messages in quantization logs")
    exit(1)

print("✅ All required files present!")
print()

# Try loading
print("="*70)
print("LOADING MODEL")
print("="*70)
print()

try:
    from gptqmodel import GPTQModel
    from transformers import AutoTokenizer
    import torch

    print("Method 1: Using GPTQModel.load() (CORRECT for GPTQ models)")
    print("-" * 70)

    print("Loading model...")
    model = GPTQModel.load(
        quantized_path,
        device_map="auto",
        dtype=torch.float16  # Important for T4
    )
    print("✅ Model loaded successfully!")

    print()
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(quantized_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("✅ Tokenizer loaded successfully!")

    print()
    print("="*70)
    print("SUCCESS! Model is ready for inference")
    print("="*70)
    print()

    # Quick test
    print("Quick test generation:")
    prompt = "Hepatic steatosis means"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input:  {prompt}")
    print(f"Output: {result}")
    print()
    print("✅ Inference working!")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    print()
    import traceback
    traceback.print_exc()
    print()
    print("TROUBLESHOOTING:")
    print("1. Make sure gptqmodel is installed: pip install gptqmodel")
    print("2. Check CUDA is available: python -c 'import torch; print(torch.cuda.is_available())'")
    print("3. Verify files are not corrupted")
    exit(1)

print()
print("="*70)
print("HOW TO USE THIS MODEL")
print("="*70)
print()
print("CORRECT way to load GPTQ models:")
print()
print("from gptqmodel import GPTQModel")
print("from transformers import AutoTokenizer")
print("import torch")
print()
print("# Load model")
print(f"model = GPTQModel.load(")
print(f"    '{quantized_path}',")
print(f"    device_map='auto',")
print(f"    dtype=torch.float16  # Required for T4 GPUs")
print(f")")
print()
print(f"# Load tokenizer")
print(f"tokenizer = AutoTokenizer.from_pretrained('{quantized_path}')")
print()
print("# Generate")
print("inputs = tokenizer('Your prompt', return_tensors='pt').to('cuda')")
print("outputs = model.generate(**inputs, max_new_tokens=100)")
print("print(tokenizer.decode(outputs[0]))")
print()
print("="*70)
print()
print("❌ WRONG way (will not work):")
print()
print("from transformers import AutoModelForCausalLM  # ← Don't use this!")
print(f"model = AutoModelForCausalLM.from_pretrained('{quantized_path}')  # ← Will fail!")
print()
print("GPTQ quantized models must be loaded with GPTQModel.load()!")
