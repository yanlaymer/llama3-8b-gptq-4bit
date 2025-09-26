# 🚀 Run GPTQ Quantization in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yanlaymer/llama3-8b-gptq-4bit/blob/main/examples/notebooks/Colab_GPTQ_Quantization.ipynb)

## Quick Start (5 minutes to setup, 30 minutes to run)

### 1. Open in Colab
Click the "Open in Colab" badge above or go to:
`https://colab.research.google.com/github/yanlaymer/llama3-8b-gptq-4bit/blob/main/examples/notebooks/Colab_GPTQ_Quantization.ipynb`

### 2. Set GPU Runtime
- Runtime → Change runtime type → GPU (T4/A100)
- Hardware accelerator: GPU

### 3. Set Your HF Token
In the first code cell, replace:
```python
%env HF_TOKEN=hf_your_token_here
```
With your actual token from [HuggingFace Settings](https://huggingface.co/settings/tokens)

### 4. Run All Cells
- Runtime → Run all (Ctrl+F9)
- Or run each cell sequentially

## What Happens

✅ **Clones** this repository automatically
✅ **Installs** all dependencies
✅ **Downloads** Llama-3-8B-Instruct
✅ **Quantizes** to 4-bit GPTQ (~20 minutes)
✅ **Tests** the quantized model
✅ **Uploads** to your HuggingFace account
✅ **Creates** downloadable zip file

## Requirements

- **Colab Account** (free works, Pro/Pro+ recommended)
- **HuggingFace Account** with token
- **Llama-3 Access** (request at https://llama.meta.com/llama3)

## GPU Compatibility

| GPU | VRAM | Status | Time |
|-----|------|--------|------|
| T4 (Free) | 16GB | ✅ Works | ~45 min |
| T4 (Pro) | 16GB | ✅ Works | ~30 min |
| A100 (Pro+) | 40GB | ✅ Fast | ~15 min |

## Output

After completion, you'll have:
- **Quantized Model**: 4.2GB (75% smaller than original)
- **HuggingFace Upload**: `nalrunyan/llama3-8b-gptq-4bit`
- **Local Download**: `quantized_llama3_8b_gptq.zip`
- **Performance**: ~4x compression, 2-3x faster inference

## Troubleshooting

### "Permission denied" for Llama-3
1. Go to https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
2. Click "Request access"
3. Accept license terms
4. Wait for approval (usually instant)

### "Out of memory" error
- Try Colab Pro for better GPU
- Reduce `CALIBRATION_SAMPLES` from 256 to 128

### "Token invalid"
- Generate new token at https://huggingface.co/settings/tokens
- Use "Write" permissions
- Check token format: `hf_...`

## Using Your Quantized Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("nalrunyan/llama3-8b-gptq-4bit")
model = AutoModelForCausalLM.from_pretrained("nalrunyan/llama3-8b-gptq-4bit", device_map="auto")

prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

## Support

- **Issues**: https://github.com/yanlaymer/llama3-8b-gptq-4bit/issues
- **Discussions**: https://github.com/yanlaymer/llama3-8b-gptq-4bit/discussions
- **HuggingFace**: https://huggingface.co/nalrunyan/llama3-8b-gptq-4bit

---

**Ready to quantize? Click the Colab badge above! 🚀**