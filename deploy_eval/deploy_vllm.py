"""
vLLM Deployment Script for LLaMA3-8B-GPTQ-4bit Medical Model
============================================================

This script deploys the quantized model using vLLM for high-throughput inference.
"""

import argparse
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


MODEL_ID = "nalrunyan/llama3-8b-gptq-4bit"

# Medical-optimized system prompt
MEDICAL_SYSTEM_PROMPT = """You are a medical AI assistant trained to help healthcare professionals.
You provide accurate, evidence-based medical information while acknowledging limitations.
Always recommend verification by qualified healthcare professionals for clinical decisions."""


def load_model(
    model_id: str = MODEL_ID,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096,
    quantization: str = "gptq",
):
    """Load the quantized model with vLLM."""

    print(f"Loading model: {model_id}")
    print(f"Quantization: {quantization}")
    print(f"GPU memory utilization: {gpu_memory_utilization}")

    llm = LLM(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        quantization=quantization,
        dtype="half",
        trust_remote_code=True,
    )

    print("Model loaded successfully!")
    return llm


def create_sampling_params(
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 512,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
) -> SamplingParams:
    """Create sampling parameters for generation."""

    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )


def format_medical_prompt(user_query: str, include_system: bool = True) -> str:
    """Format prompt using LLaMA3 chat template for medical queries."""

    if include_system:
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{MEDICAL_SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    else:
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt


def generate_response(llm: LLM, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
    """Generate responses for a batch of prompts."""

    outputs = llm.generate(prompts, sampling_params)

    responses = []
    for output in outputs:
        generated_text = output.outputs[0].text
        responses.append(generated_text)

    return responses


def run_interactive_session(llm: LLM):
    """Run an interactive session with the model."""

    print("\n" + "=" * 60)
    print("Medical AI Assistant - Interactive Mode")
    print("Type 'quit' or 'exit' to end the session")
    print("=" * 60 + "\n")

    sampling_params = create_sampling_params(temperature=0.7, max_tokens=512)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nExiting interactive session. Goodbye!")
                break

            if not user_input:
                continue

            prompt = format_medical_prompt(user_input)
            responses = generate_response(llm, [prompt], sampling_params)

            print(f"\nAssistant: {responses[0]}")

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Deploy LLaMA3-8B-GPTQ medical model with vLLM")
    parser.add_argument("--model-id", type=str, default=MODEL_ID, help="HuggingFace model ID")
    parser.add_argument("--tensor-parallel", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu-memory", type=float, default=0.85, help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model context length")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--prompt", type=str, help="Single prompt to process")

    args = parser.parse_args()

    # Load model
    llm = load_model(
        model_id=args.model_id,
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
    )

    if args.interactive:
        run_interactive_session(llm)
    elif args.prompt:
        sampling_params = create_sampling_params()
        prompt = format_medical_prompt(args.prompt)
        responses = generate_response(llm, [prompt], sampling_params)
        print(f"\nResponse: {responses[0]}")
    else:
        # Demo with sample medical queries
        sample_queries = [
            "Summarize the key findings from a chest X-ray showing bilateral infiltrates.",
            "What are the differential diagnoses for a patient presenting with acute chest pain?",
            "Explain the mechanism of action of metformin in type 2 diabetes.",
        ]

        sampling_params = create_sampling_params()
        prompts = [format_medical_prompt(q) for q in sample_queries]
        responses = generate_response(llm, prompts, sampling_params)

        print("\n" + "=" * 60)
        print("Sample Medical Queries and Responses")
        print("=" * 60)

        for query, response in zip(sample_queries, responses):
            print(f"\nQuery: {query}")
            print(f"Response: {response}")
            print("-" * 40)


if __name__ == "__main__":
    main()
