# Tesla T4 + CUDA 12.4 Installation Guide

## 🔧 Hardware Verified
- **GPU**: Tesla T4 (15GB VRAM, SM_75)
- **CUDA**: 12.4
- **Driver**: 550.54.15
- **Status**: ✅ Fully Compatible

## 📋 Quick Installation

### Option A: Auto-GPTQ (Original)
```bash
# 1. Install PyTorch with CUDA 12.1 (compatible with 12.4)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. Install auto-gptq with CUDA 12.1 wheels
pip install auto-gptq>=0.7.1 \
    --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu121/

# 3. Install the toolkit
cd innova-llama3-gptq
pip install -e .
```

### Option B: GPTQModel (Recommended)
```bash
# 1. Install PyTorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. Install GPTQModel (better T4 support)
pip install -v gptqmodel --no-build-isolation

# 3. Install the toolkit (modify to use gptqmodel)
cd innova-llama3-gptq
pip install -e .
```

### Using Requirements File
```bash
pip install -r requirements-t4-cuda124.txt
```

## ⚠️ Common Issues & Solutions

### Issue: "torch not installed" during auto-gptq install
**Solution**: Install PyTorch first, then auto-gptq
```bash
pip install torch first
pip install auto-gptq
```

### Issue: CUDA kernel compilation fails
**Solution**: Use pre-built wheels instead of building from source
```bash
pip install auto-gptq --no-build-isolation \
    --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu121/
```

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

# Test auto-gptq
try:
    from auto_gptq import AutoGPTQForCausalLM
    print("✅ auto-gptq installed correctly")
except ImportError as e:
    print(f"❌ auto-gptq issue: {e}")

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

## 🔄 Migration from auto-gptq to GPTQModel

If you want to use the newer GPTQModel library:

1. **Install GPTQModel**:
```bash
pip uninstall auto-gptq
pip install -v gptqmodel --no-build-isolation
```

2. **Update imports** in the toolkit:
```python
# Old
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# New
from gptqmodel import GPTQModel, QuantizeConfig
```

3. **Benefits**:
- Better T4 support
- Faster quantization
- More stable on CUDA 12.4
- Active development

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