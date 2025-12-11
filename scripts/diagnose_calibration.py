#!/usr/bin/env python3
"""
Diagnostic script to investigate calibration dataset issues

Run this in Kaggle to understand why only 21 samples are being used
"""

import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

def diagnose_calibration_data(jsonl_path: str, model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
    """Diagnose calibration dataset to understand sample count issues"""

    print("=" * 80)
    print("CALIBRATION DATASET DIAGNOSTIC")
    print("=" * 80)
    print()

    # Check if file exists
    file_path = Path(jsonl_path)
    if not file_path.exists():
        print(f"❌ ERROR: File does not exist: {jsonl_path}")
        print()
        print("You need to create the calibration data first:")
        print("  python scripts/prepare_medical_calibration.py \\")
        print("      --mix radiology \\")
        print("      --samples 512 \\")
        print(f"      --output {jsonl_path}")
        return

    print(f"✅ File exists: {jsonl_path}")
    print(f"   File size: {file_path.stat().st_size / 1024:.2f} KB")
    print()

    # Count lines
    with open(jsonl_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"📊 Total samples in file: {len(lines)}")
    print()

    if len(lines) == 0:
        print("❌ ERROR: File is empty!")
        return

    # Parse samples
    samples = []
    for i, line in enumerate(lines):
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"⚠️  Warning: Line {i+1} is not valid JSON: {e}")

    print(f"✅ Valid JSON samples: {len(samples)}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        print(f"❌ ERROR loading tokenizer: {e}")
        return

    print("✅ Tokenizer loaded")
    print()

    # Analyze token lengths
    print("Analyzing sample lengths...")
    print()

    lengths = []
    for i, sample in enumerate(samples):
        text = sample.get("text", "")
        if not text:
            print(f"⚠️  Warning: Sample {i+1} has no 'text' field")
            continue

        tokens = tokenizer(text, return_tensors="pt", truncation=False)
        length = tokens.input_ids.shape[1]
        lengths.append(length)

    if not lengths:
        print("❌ ERROR: No valid text samples found!")
        return

    # Statistics
    import numpy as np

    print(f"Token Length Statistics:")
    print(f"  Mean:   {np.mean(lengths):.1f} tokens")
    print(f"  Median: {np.median(lengths):.1f} tokens")
    print(f"  Min:    {np.min(lengths)} tokens")
    print(f"  Max:    {np.max(lengths)} tokens")
    print(f"  Std:    {np.std(lengths):.1f} tokens")
    print()

    # Filter analysis (min_length=256 is used by default)
    min_lengths = [256, 512, 1024, 2048]
    print("Samples passing minimum length filters:")
    print()
    for min_len in min_lengths:
        passing = sum(1 for l in lengths if l >= min_len)
        pct = 100 * passing / len(lengths)
        status = "✅" if passing >= 256 else "⚠️ "
        print(f"  {status} >= {min_len:4d} tokens: {passing:3d} samples ({pct:5.1f}%)")

    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print()

    samples_above_256 = sum(1 for l in lengths if l >= 256)

    if samples_above_256 < 256:
        print(f"⚠️  ISSUE FOUND: Only {samples_above_256} samples are >= 256 tokens")
        print()
        print("The quantization pipeline filters out samples < 256 tokens.")
        print(f"This is why you're seeing only {samples_above_256} samples instead of {len(samples)}.")
        print()
        print("SOLUTIONS:")
        print()
        print("1. Re-generate calibration data with longer samples:")
        print("   python scripts/prepare_medical_calibration.py \\")
        print("       --mix radiology \\")
        print("       --samples 512 \\")
        print("       --min-length 256 \\")
        print("       --max-length 4096 \\")
        print(f"       --output {jsonl_path}")
        print()
        print("2. Use a different calibration dataset (wikitext2, c4):")
        print("   quantized_path = quantize_llama3_gptq(")
        print("       model_id='meta-llama/Meta-Llama-3-8B-Instruct',")
        print("       calib_dataset='wikitext2',  # Use built-in dataset")
        print("       max_calib_samples=512,")
        print("       ...)")
        print()
    else:
        print(f"✅ Dataset looks good: {samples_above_256} samples >= 256 tokens")
        print()
        print("If you're still seeing only 21 samples, check:")
        print("1. Are you passing the correct file path?")
        print("2. Is the file in the right location relative to your notebook?")
        print(f"3. Try absolute path: {file_path.absolute()}")

    print()

    # Show sample texts
    print("=" * 80)
    print("SAMPLE PREVIEW (first 3 samples)")
    print("=" * 80)
    print()

    for i, sample in enumerate(samples[:3]):
        text = sample.get("text", "")
        tokens = tokenizer(text, return_tensors="pt", truncation=False)
        length = tokens.input_ids.shape[1]

        print(f"Sample {i+1}:")
        print(f"  Source: {sample.get('source', 'unknown')}")
        print(f"  Type:   {sample.get('type', 'unknown')}")
        print(f"  Length: {length} tokens")
        print(f"  Text preview: {text[:200]}...")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose calibration dataset issues")
    parser.add_argument(
        "--file",
        type=str,
        default="data/medical_calibration.jsonl",
        help="Path to calibration JSONL file"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model ID for tokenizer"
    )

    args = parser.parse_args()

    diagnose_calibration_data(args.file, args.model_id)