"""
vLLM OpenAI-Compatible API Server for Medical Model
====================================================

Launches an OpenAI-compatible API server for the quantized model.
"""

import subprocess
import sys


def main():
    """Start the vLLM OpenAI-compatible API server."""

    model_id = "nalrunyan/llama3-8b-gptq-4bit"

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_id,
        "--quantization",
        "gptq",
        "--dtype",
        "half",
        "--gpu-memory-utilization",
        "0.85",
        "--max-model-len",
        "4096",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--trust-remote-code",
    ]

    print("Starting vLLM OpenAI-compatible API server...")
    print(f"Model: {model_id}")
    print("Endpoint will be available at: http://localhost:8000/v1")
    print("\nUsage examples:")
    print('  curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d \'{"model": "nalrunyan/llama3-8b-gptq-4bit", "prompt": "Explain diabetes:", "max_tokens": 100}\'')
    print("\nPress Ctrl+C to stop the server.\n")

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
