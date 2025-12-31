# LLaMA3-8B-GPTQ-4bit Medical Model Validation

Deployment and validation toolkit for the quantized LLaMA3-8B medical model.

## Model Information

- **Model**: [nalrunyan/llama3-8b-gptq-4bit](https://huggingface.co/nalrunyan/llama3-8b-gptq-4bit)
- **Base**: Meta-Llama-3-8B-Instruct
- **Quantization**: 4-bit GPTQ (5.35 GiB, 3x smaller than FP16)
- **Medical Calibration**: PubMedQA (60%) + PMC-Patients (40%)

## Validation Results (December 2024)

Validated on **GCP g2-standard-4 (NVIDIA L4 23GB)** with **vLLM v0.13.0**.

### Quality Metrics

| Metric | Result |
|--------|--------|
| **Overall Pass Rate** | 86.7% |
| **Average Coverage** | 83.2% |
| **Categories at 100% Pass** | 3/5 |

### Category Breakdown

| Category | Pass Rate | Avg Coverage |
|----------|-----------|--------------|
| Medication | 100% | 100.0% |
| Clinical QA | 100% | 88.7% |
| Patient Communication | 100% | 78.6% |
| Radiology | 75% | 78.9% |
| Diagnosis | 50% | 60.7% |

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Throughput** | 321.8 tokens/sec |
| **Avg Latency** | 1.37 seconds |
| **Model Memory** | 5.35 GiB |

See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for complete details.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Deploy with vLLM

**Interactive mode:**
```bash
python deploy_vllm.py --interactive
```

**Single prompt:**
```bash
python deploy_vllm.py --prompt "What are the symptoms of pneumonia?"
```

**API server (OpenAI-compatible):**
```bash
python serve_vllm_api.py
```

### 3. Run Validation

```bash
python validate_model.py --test-file medical_test_cases.json --output results.json
```

With verbose output:
```bash
python validate_model.py --verbose
```

## Files

| File | Description |
|------|-------------|
| `deploy_vllm.py` | Main deployment script with vLLM |
| `serve_vllm_api.py` | OpenAI-compatible API server |
| `validate_model.py` | Validation and benchmarking script |
| `medical_test_cases.json` | Medical domain test cases |
| `requirements.txt` | Python dependencies |

## Test Categories

- **Radiology**: Report summarization, findings interpretation
- **Clinical QA**: Differential diagnosis, clinical reasoning
- **Medication**: Mechanism explanation, drug interactions
- **Diagnosis**: Lab interpretation, ECG analysis
- **Patient Communication**: Simplified explanations

## GCP Deployment

```bash
# SSH to GPU instance
gcloud compute ssh <instance-name> --zone <zone>

# Clone and setup
git clone <this-repo>
cd quantcheck
pip install -r requirements.txt

# Run server
python serve_vllm_api.py
```

## API Usage

Once the server is running:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nalrunyan/llama3-8b-gptq-4bit",
    "prompt": "Summarize the diagnosis:",
    "max_tokens": 256
  }'
```

## Important Note

All medical outputs should be reviewed by qualified healthcare professionals. This model is a tool to assist, not replace, medical judgment.
