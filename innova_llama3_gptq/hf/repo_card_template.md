---
license: {{LICENSE}}
language:
- en
library_name: transformers
tags:
- llama-3
- gptq
- quantization
- innova
datasets:
- {{CALIBRATION_DATASET}}
base_model: {{BASE_MODEL}}
---

# {{MODEL_NAME}} - GPTQ Quantized

This is a {{BITS}}-bit GPTQ quantized version of [{{BASE_MODEL}}](https://huggingface.co/{{BASE_MODEL}}) optimized by Innova Engineering.

## Model Details

### Model Description

This model has been quantized using GPTQ (Generative Pre-trained Transformer Quantization) to reduce model size while maintaining performance. The quantization was performed with the following configuration:

- **Bits**: {{BITS}}
- **Group Size**: {{GROUP_SIZE}}
- **Activation Order**: {{DESC_ACT}}
- **Calibration Dataset**: {{CALIBRATION_DATASET}}
- **Calibration Samples**: {{CALIBRATION_SAMPLES}}

### Intended Use

- **Primary Use**: Efficient text generation with reduced memory footprint
- **Intended Users**: Developers and researchers requiring efficient LLM deployment
- **Out-of-scope**: This model inherits the limitations of the base model

## Performance Metrics

### Perplexity Results

| Dataset | Score |
|---------|-------|
{{PERPLEXITY_TABLE}}

### Task Evaluation

| Task | Score |
|------|-------|
{{TASK_TABLE}}

### Inference Performance

| Metric | Value |
|--------|-------|
| Latency (BS=1) | {{LATENCY_MS}} ms |
| Throughput | {{THROUGHPUT_TPS}} tokens/s |
| Model Size | {{MODEL_SIZE}} |
| Memory Usage | {{MEMORY_USAGE}} GB |

## Usage

### Installation

```bash
pip install transformers accelerate gptqmodel
```

### Loading the Model

```python
from transformers import AutoTokenizer
from gptqmodel import GPTQModel

model_id = "{{HF_REPO_ID}}"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = GPTQModel.load(
    model_id,
    device_map="auto",
    trust_remote_code=False
)

# Generate text
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### Using with Text Generation Inference (TGI)

```bash
docker run --gpus all --shm-size 1g -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id {{HF_REPO_ID}} \
  --quantize gptq
```

### Using with vLLM

```python
from vllm import LLM, SamplingParams

llm = LLM(model="{{HF_REPO_ID}}", quantization="gptq")
sampling_params = SamplingParams(temperature=0.7, top_p=0.9)

prompts = ["The meaning of life is"]
outputs = llm.generate(prompts, sampling_params)
```

## Quantization Details

### Method

We use GPTQ (Generative Pre-trained Transformer Quantization) which is a one-shot weight quantization method based on approximate second-order information. The key advantages include:

- Minimal accuracy degradation
- Significant memory reduction ({{COMPRESSION_RATIO}}x compression)
- Fast inference with specialized kernels
- No need for retraining

### Hardware Requirements

**Minimum Requirements:**
- GPU: NVIDIA GPU with 16GB+ VRAM
- CUDA: 11.8 or higher
- System RAM: 32GB recommended

**Tested Hardware:**
- {{TESTED_GPU}}
- CUDA Version: {{CUDA_VERSION}}
- Driver Version: {{DRIVER_VERSION}}

## Limitations and Bias

This model inherits all limitations and biases from the base {{BASE_MODEL}} model. Additionally:

- Quantization may slightly affect output quality in edge cases
- Performance characteristics vary by hardware
- Not optimized for CPU inference

## Citation

If you use this model, please cite:

```bibtex
@misc{innova_gptq_2024,
  title={{{MODEL_NAME}} GPTQ Quantized},
  author={Innova Engineering},
  year={2024},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/{{HF_REPO_ID}}}}
}
```

## Support

For support and questions:
- Email: support@innova.example
- GitHub: https://github.com/innova-ai/llama3-gptq
- Issues: https://github.com/innova-ai/llama3-gptq/issues

## License

This model is subject to the license of the original {{BASE_MODEL}} model. Please refer to the original model card for license details.

---

*Quantized with ❤️ by [Innova Engineering](https://innova.example)*