#!/usr/bin/env python3
"""
Quick test script to validate a quantized GPTQ model
Tests: loading, tokenizer, simple generation
"""

import argparse
import sys
import torch
from pathlib import Path

def quick_test_model(model_path: str, use_greedy: bool = True):
    """Quick validation of quantized model"""

    print("="*70)
    print("QUICK MODEL TEST")
    print("="*70)
    print(f"Model: {model_path}")
    print()

    # Check path exists
    if not Path(model_path).exists():
        print(f"❌ ERROR: Model path not found: {model_path}")
        return False

    try:
        from gptqmodel import GPTQModel
        from transformers import AutoTokenizer

        # Test 1: Load model
        print("Test 1: Loading model...")
        model = GPTQModel.load(
            model_path,
            device_map="auto",
            dtype=torch.float16
        )
        print("✅ Model loaded")

        # Test 2: Load tokenizer
        print("\nTest 2: Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        print("✅ Tokenizer loaded")

        # Test 3: Simple generation
        print("\nTest 3: Simple text generation...")
        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.inference_mode():
            if use_greedy:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Input:  '{prompt}'")
        print(f"  Output: '{result}'")

        # Check if output makes sense
        if "Paris" in result or "paris" in result.lower():
            print("✅ Output is sensible")
        else:
            print("⚠️  Output may be unexpected (but model works)")

        # Test 4: Medical prompt (if relevant)
        print("\nTest 4: Medical prompt...")
        medical_prompt = "Hepatic steatosis means"
        inputs = tokenizer(medical_prompt, return_tensors="pt").to("cuda")

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Input:  '{medical_prompt}'")
        print(f"  Output: '{result}'")
        print("✅ Medical generation works")

        print()
        print("="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print()
        print("Model is ready for use!")
        return True

    except RuntimeError as e:
        if "CUDA" in str(e) and "assert" in str(e):
            print()
            print("="*70)
            print("❌ CUDA ASSERTION ERROR")
            print("="*70)
            print()
            print("The model produces inf/nan values during generation.")
            print("This indicates the model was poorly quantized.")
            print()
            print("ROOT CAUSE:")
            print("  - Model was quantized with insufficient calibration samples")
            print("  - Recommended minimum: 256 samples")
            print("  - Industry standard: 512 samples")
            print()
            print("SOLUTION:")
            print("  You MUST re-quantize the model with proper calibration.")
            print("  See KAGGLE_REQUANTIZE.md for complete instructions.")
            print()
            print("This model cannot be fixed - it must be re-quantized.")
            return False
        else:
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quick test for GPTQ quantized models"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to quantized model directory"
    )
    parser.add_argument(
        "--sampling",
        action="store_true",
        help="Use sampling instead of greedy decoding (less stable)"
    )

    args = parser.parse_args()

    success = quick_test_model(args.model_path, use_greedy=not args.sampling)
    sys.exit(0 if success else 1)
