"""Task-based evaluation using lm-eval-harness or custom implementations"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoTokenizer
from gptqmodel import GPTQModel

logger = logging.getLogger(__name__)


def evaluate_with_lm_eval(
    model_path: str,
    tasks: List[str] = ["hellaswag", "arc_easy", "mmlu"],
    num_fewshot: int = 0,
    batch_size: int = 1,
    device: str = "cuda",
    trust_remote_code: bool = False
) -> Dict[str, Any]:
    """
    Evaluate model using lm-eval-harness

    Args:
        model_path: Path to the model
        tasks: List of task names
        num_fewshot: Number of few-shot examples
        batch_size: Batch size for evaluation
        device: Device to use
        trust_remote_code: Trust remote code

    Returns:
        Dictionary with task results
    """
    try:
        import lm_eval
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM

        logger.info(f"Running lm-eval on tasks: {tasks}")

        # Create model wrapper
        lm = HFLM(
            pretrained=model_path,
            device=device,
            batch_size=batch_size,
            trust_remote_code=trust_remote_code
        )

        # Run evaluation
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device=device
        )

        # Extract metrics
        task_metrics = {}
        for task in tasks:
            if task in results["results"]:
                task_results = results["results"][task]
                # Get the main metric for each task
                if "acc" in task_results:
                    task_metrics[task] = task_results["acc"]
                elif "acc_norm" in task_results:
                    task_metrics[task] = task_results["acc_norm"]
                elif "exact_match" in task_results:
                    task_metrics[task] = task_results["exact_match"]
                else:
                    task_metrics[task] = task_results

        return task_metrics

    except ImportError:
        logger.warning("lm-eval-harness not installed, using fallback evaluation")
        return evaluate_generation_quality(model_path, device=device)


def evaluate_generation_quality(
    model_path: str,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 32,
    temperature: float = 0.7,
    device: str = "cuda",
    trust_remote_code: bool = False
) -> Dict[str, Any]:
    """
    Evaluate generation quality with custom prompts

    Args:
        model_path: Path to the model
        prompts: List of prompts to use
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        device: Device to use
        trust_remote_code: Trust remote code

    Returns:
        Dictionary with generation metrics
    """
    if prompts is None:
        prompts = [
            "The capital of France is",
            "Machine learning is",
            "The most important invention of the 20th century was",
            "To solve climate change, we should",
            "The meaning of life is"
        ]

    logger.info("Evaluating generation quality")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPTQModel.load(
        model_path,
        device_map="auto",
        trust_remote_code=trust_remote_code
    )

    results = []
    total_time = 0
    total_tokens = 0

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        generation_time = time.time() - start_time

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        num_tokens = len(outputs[0]) - len(inputs.input_ids[0])

        results.append({
            "prompt": prompt,
            "generated": generated_text,
            "tokens_generated": num_tokens,
            "time_seconds": generation_time
        })

        total_time += generation_time
        total_tokens += num_tokens

    avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0

    return {
        "generation_samples": results[:3],  # Include first 3 samples
        "avg_tokens_per_second": avg_tokens_per_second,
        "total_generation_time": total_time,
        "total_tokens_generated": total_tokens
    }


def measure_inference_latency(
    model_path: str,
    batch_sizes: List[int] = [1, 4, 8],
    sequence_length: int = 512,
    num_iterations: int = 10,
    device: str = "cuda",
    trust_remote_code: bool = False
) -> Dict[str, Any]:
    """
    Measure inference latency at different batch sizes

    Args:
        model_path: Path to the model
        batch_sizes: List of batch sizes to test
        sequence_length: Input sequence length
        num_iterations: Number of iterations per batch size
        device: Device to use
        trust_remote_code: Trust remote code

    Returns:
        Dictionary with latency measurements
    """
    logger.info("Measuring inference latency")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code
    )

    model = GPTQModel.load(
        model_path,
        device_map="auto",
        trust_remote_code=trust_remote_code
    )
    model.eval()

    results = {}

    for batch_size in batch_sizes:
        # Create dummy input
        input_ids = torch.randint(
            0, tokenizer.vocab_size,
            (batch_size, sequence_length),
            device=device
        )

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids)

        # Measure
        torch.cuda.synchronize() if device == "cuda" else None
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(input_ids)

        torch.cuda.synchronize() if device == "cuda" else None
        total_time = time.time() - start_time

        avg_latency = total_time / num_iterations
        throughput = batch_size / avg_latency

        results[f"batch_{batch_size}"] = {
            "avg_latency_ms": avg_latency * 1000,
            "throughput_samples_per_sec": throughput,
            "total_time": total_time,
            "iterations": num_iterations
        }

        logger.info(f"Batch {batch_size}: {avg_latency*1000:.2f}ms avg latency")

    return results