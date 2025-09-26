# Tesla T4 + CUDA 12.4 Installation Guide

## 🔧 Hardware Verified
- **GPU**: Tesla T4 (15GB VRAM, SM_75)
- **CUDA**: 12.4
- **Driver**: 550.54.15
- **Status**: ✅ Fully Compatible

## 📋 Quick Installation

### Recommended Installation (GPTQModel)
```bash
# 1. Install PyTorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. Install GPTQModel (optimized for Tesla T4)
pip install -v gptqmodel --no-build-isolation

# 3. Install the toolkit
cd innova-llama3-gptq
pip install -e .
```

### Using Requirements File
```bash
pip install -r requirements-t4-cuda124.txt
```

## ⚠️ Common Issues & Solutions

### Issue: GPTQModel installation fails
**Solution**: Install PyTorch first, then GPTQModel with no-build-isolation
```bash
pip install torch first
pip install -v gptqmodel --no-build-isolation
```

### Issue: CUDA compilation errors
**Solution**: GPTQModel comes with pre-built wheels, avoiding compilation issues

### Issue: "BFloat16 not supported on T4"
**Solution**: T4 supports FP16, not BF16. Our toolkit handles this automatically.

### Issue: Out of memory during quantization
**Solutions**:
- Reduce calibration samples: `--max-calib-samples 256`
- Use batch_size=1 (default)
- Monitor with `nvidia-smi`

## 🧪 Verification Test

```python
# Test CUDA and GPTQ
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test GPTQModel
try:
    from gptqmodel import GPTQModel, QuantizeConfig
    print("✅ GPTQModel installed correctly")
except ImportError as e:
    print(f"❌ GPTQModel issue: {e}")

# Test toolkit
try:
    from innova_llama3_gptq import quantize_llama3_gptq
    print("✅ GPTQ toolkit ready")
except ImportError as e:
    print(f"❌ Toolkit issue: {e}")
```

## 📊 T4 Optimization Settings

```yaml
# T4-optimized config (already set in configs/)
quantization:
  bits: 4
  group_size: 128  # Optimal for T4
  desc_act: true
  use_cuda_fp16: true  # T4 supports FP16
  use_triton: false    # Not needed on T4
  batch_size: 1        # Optimal for 15GB VRAM

calibration:
  max_samples: 512     # Good balance for T4
  max_length: 2048     # Fits in T4 memory
```

## 🚀 Expected Performance

| Metric | Value |
|--------|-------|
| Quantization Time | 25-35 minutes |
| Peak VRAM Usage | ~14GB |
| Final Model Size | 4.2GB |
| Compression Ratio | 4× |
| Inference Speedup | 2-3× |

## ✨ Why GPTQModel?

This toolkit now exclusively uses GPTQModel (the modern successor to auto-gptq):

**Key Benefits**:
- ✅ Better Tesla T4 support with optimized kernels
- ✅ Faster installation with pre-built wheels
- ✅ More stable on CUDA 12.4
- ✅ Active development and better documentation
- ✅ Cleaner API with better error messages
- ✅ No compilation issues in cloud environments

## ✅ Final Check

After installation, run:
```bash
cd innova-llama3-gptq/examples
export HF_TOKEN="hf_your_token"
make help  # Should show all options
nvidia-smi  # Check GPU is visible
python -c "import torch; print(torch.cuda.is_available())"
```

Your Tesla T4 is perfectly suited for this task! 🚀