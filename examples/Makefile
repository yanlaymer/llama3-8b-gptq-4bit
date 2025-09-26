# Innova Llama-3 GPTQ Makefile
PYTHON ?= python3
MODEL_ID ?= meta-llama/Meta-Llama-3-8B-Instruct
HF_ORG ?= nalrunyan
REPO_NAME ?= llama3-8b-gptq-4bit

.PHONY: help install clean quantize-4bit quantize-3bit eval eval-quick export-hf full-pipeline

help:
	@echo "Innova Llama-3 GPTQ Toolkit"
	@echo ""
	@echo "Usage:"
	@echo "  make install         Install dependencies"
	@echo "  make quantize-4bit   Quantize model to 4-bit GPTQ"
	@echo "  make quantize-3bit   Quantize model to 3-bit GPTQ"
	@echo "  make eval           Run full evaluation suite"
	@echo "  make eval-quick     Run quick evaluation (perplexity only)"
	@echo "  make export-hf      Export to Hugging Face format"
	@echo "  make full-pipeline  Run complete pipeline (quantize -> eval -> export)"
	@echo ""
	@echo "Environment Variables:"
	@echo "  MODEL_ID    Base model ID (default: $(MODEL_ID))"
	@echo "  HF_ORG      Hugging Face org (default: $(HF_ORG))"
	@echo "  REPO_NAME   Repository name (default: $(REPO_NAME))"
	@echo "  HF_TOKEN    Hugging Face auth token (set to: [REDACTED_HF_TOKEN])"

install:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e ..
	$(PYTHON) -m pip install transformers>=4.44.0 accelerate>=0.33.0
	$(PYTHON) -m pip install datasets huggingface_hub auto-gptq
	$(PYTHON) -m pip install lm_eval safetensors pyyaml

clean:
	rm -rf artifacts/
	rm -rf results/
	find .. -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find .. -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

quantize-4bit:
	$(PYTHON) ../scripts/quantize_llama3_gptq.py \
		--model-id $(MODEL_ID) \
		--bits 4 \
		--group-size 128 \
		--desc-act \
		--dataset wikitext2 \
		--max-calib-samples 512 \
		--out-dir artifacts/gptq/4bit \
		--seed 42 \
		--auth-token [REDACTED_HF_TOKEN]

quantize-3bit:
	$(PYTHON) ../scripts/quantize_llama3_gptq.py \
		--model-id $(MODEL_ID) \
		--bits 3 \
		--group-size 128 \
		--desc-act \
		--dataset wikitext2 \
		--max-calib-samples 1024 \
		--out-dir artifacts/gptq/3bit \
		--seed 42 \
		--auth-token [REDACTED_HF_TOKEN]

eval:
	@if [ ! -d "artifacts/gptq/4bit" ]; then \
		echo "Error: Model not found. Run 'make quantize-4bit' first."; \
		exit 1; \
	fi
	$(PYTHON) ../scripts/eval_gptq.py \
		--model-dir artifacts/gptq/4bit \
		--config ../configs/eval.yaml \
		--results-dir results \
		--batch-size 1

eval-quick:
	@if [ ! -d "artifacts/gptq/4bit" ]; then \
		echo "Error: Model not found. Run 'make quantize-4bit' first."; \
		exit 1; \
	fi
	$(PYTHON) ../scripts/eval_gptq.py \
		--model-dir artifacts/gptq/4bit \
		--config ../configs/eval.yaml \
		--results-dir results \
		--skip-tasks \
		--skip-latency

export-hf:
	@if [ ! -d "artifacts/gptq/4bit" ]; then \
		echo "Error: Model not found. Run 'make quantize-4bit' first."; \
		exit 1; \
	fi
	$(PYTHON) ../scripts/export_hf_gptq.py \
		--model-dir artifacts/gptq/4bit \
		--repo-id $(HF_ORG)/$(REPO_NAME) \
		--base-model $(MODEL_ID) \
		--results-dir results \
		--private

export-hf-push:
	@if [ ! -d "artifacts/gptq/4bit" ]; then \
		echo "Error: Model not found. Run 'make quantize-4bit' first."; \
		exit 1; \
	fi
	$(PYTHON) ../scripts/export_hf_gptq.py \
		--model-dir artifacts/gptq/4bit \
		--repo-id $(HF_ORG)/$(REPO_NAME) \
		--base-model $(MODEL_ID) \
		--results-dir results \
		--private \
		--push \
		--token [REDACTED_HF_TOKEN]

full-pipeline: quantize-4bit eval export-hf
	@echo "========================================="
	@echo "Pipeline complete!"
	@echo "Model: artifacts/gptq/4bit"
	@echo "Results: results/"
	@echo "========================================="

serve-vllm:
	@if [ ! -d "artifacts/gptq/4bit" ]; then \
		echo "Error: Model not found. Run 'make quantize-4bit' first."; \
		exit 1; \
	fi
	$(PYTHON) -m vllm.entrypoints.openai.api_server \
		--model artifacts/gptq/4bit \
		--quantization gptq \
		--max-model-len 2048 \
		--port 8000

benchmark:
	@if [ ! -d "artifacts/gptq/4bit" ]; then \
		echo "Error: Model not found. Run 'make quantize-4bit' first."; \
		exit 1; \
	fi
	$(PYTHON) ../scripts/eval_gptq.py \
		--model-dir artifacts/gptq/4bit \
		--config ../configs/eval.yaml \
		--results-dir results/benchmark \
		--skip-perplexity \
		--skip-tasks