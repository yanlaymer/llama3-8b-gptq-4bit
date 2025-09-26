#!/usr/bin/env python3
"""Test script to verify GPTQModel installation and imports"""

import sys
import torch

def test_environment():
    """Test the environment and imports"""
    print("🔧 Testing Environment:")
    print(f"   Python: {sys.version}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")

    print()

def test_gptqmodel():
    """Test GPTQModel imports"""
    print("📦 Testing GPTQModel:")
    try:
        from gptqmodel import GPTQModel, QuantizeConfig
        print("   ✅ GPTQModel imported successfully")
        print("   ✅ QuantizeConfig imported successfully")
        return True
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False

def test_toolkit():
    """Test toolkit imports"""
    print("🔧 Testing Toolkit:")
    try:
        from innova_llama3_gptq import quantize_llama3_gptq
        print("   ✅ Toolkit imported successfully")
        return True
    except ImportError as e:
        print(f"   ❌ Toolkit import failed: {e}")
        return False

def test_dependencies():
    """Test other dependencies"""
    print("📚 Testing Dependencies:")
    deps = [
        ("transformers", "AutoTokenizer"),
        ("datasets", "load_dataset"),
        ("accelerate", "init_empty_weights"),
        ("safetensors", "safe_open"),
        ("huggingface_hub", "HfApi")
    ]

    all_good = True
    for module, obj in deps:
        try:
            __import__(module)
            exec(f"from {module} import {obj}")
            print(f"   ✅ {module} ({obj})")
        except ImportError as e:
            print(f"   ❌ {module}: {e}")
            all_good = False

    return all_good

def main():
    """Run all tests"""
    print("🚀 GPTQModel Installation Test\n")

    test_environment()
    gptq_ok = test_gptqmodel()
    toolkit_ok = test_toolkit()
    deps_ok = test_dependencies()

    print("\n📊 Summary:")
    if gptq_ok and toolkit_ok and deps_ok:
        print("   🎉 All tests passed! Ready for quantization.")
        return 0
    else:
        print("   ⚠️  Some issues found. Check installation.")
        if not gptq_ok:
            print("   💡 Install GPTQModel: pip install -v gptqmodel --no-build-isolation")
        return 1

if __name__ == "__main__":
    sys.exit(main())