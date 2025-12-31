"""
Model Validation Script for LLaMA3-8B-GPTQ Medical Model
=========================================================

This script evaluates the quantized model's performance and quality
on medical domain tasks.
"""

import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams


MODEL_ID = "nalrunyan/llama3-8b-gptq-4bit"


@dataclass
class ValidationResult:
    """Result of a single validation test."""

    test_id: str
    category: str
    task: str
    input_text: str
    expected_elements: list[str]
    generated_output: str
    elements_found: list[str]
    elements_missing: list[str]
    coverage_score: float
    latency_ms: float
    tokens_generated: int


@dataclass
class BenchmarkMetrics:
    """Overall benchmark metrics."""

    total_tests: int
    passed_tests: int
    avg_coverage_score: float
    avg_latency_ms: float
    avg_tokens_per_second: float
    category_scores: dict[str, float]
    total_time_seconds: float


def load_test_cases(test_file: str) -> dict:
    """Load test cases from JSON file."""

    with open(test_file) as f:
        data = json.load(f)
    return data


def format_medical_prompt(query: str) -> str:
    """Format prompt using LLaMA3 chat template."""

    system_prompt = """You are a medical AI assistant. Provide accurate, evidence-based medical information.
Be concise and professional. Always recommend verification by healthcare professionals."""

    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def check_expected_elements(output: str, expected_elements: list[str]) -> tuple[list[str], list[str]]:
    """Check which expected elements are present in the output."""

    output_lower = output.lower()
    found = []
    missing = []

    for element in expected_elements:
        # Check for element or common synonyms
        element_lower = element.lower()
        if element_lower in output_lower:
            found.append(element)
        else:
            # Check for partial matches (for multi-word elements)
            words = element_lower.split()
            if len(words) > 1 and all(w in output_lower for w in words):
                found.append(element)
            else:
                missing.append(element)

    return found, missing


def run_validation(
    llm: LLM,
    test_cases: list[dict],
    sampling_params: SamplingParams,
    verbose: bool = False,
) -> list[ValidationResult]:
    """Run validation on all test cases."""

    results = []

    for test in tqdm(test_cases, desc="Running validation"):
        prompt = format_medical_prompt(test["input"])

        # Measure latency
        start_time = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        generated_text = outputs[0].outputs[0].text
        tokens_generated = len(outputs[0].outputs[0].token_ids)

        # Check expected elements
        found, missing = check_expected_elements(generated_text, test["expected_elements"])
        coverage = len(found) / len(test["expected_elements"]) if test["expected_elements"] else 1.0

        result = ValidationResult(
            test_id=test["id"],
            category=test["category"],
            task=test["task"],
            input_text=test["input"],
            expected_elements=test["expected_elements"],
            generated_output=generated_text,
            elements_found=found,
            elements_missing=missing,
            coverage_score=coverage,
            latency_ms=latency_ms,
            tokens_generated=tokens_generated,
        )

        results.append(result)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Test: {test['id']} ({test['category']})")
            print(f"Coverage: {coverage:.1%} | Latency: {latency_ms:.0f}ms")
            print(f"Found: {found}")
            print(f"Missing: {missing}")

    return results


def calculate_metrics(results: list[ValidationResult], total_time: float) -> BenchmarkMetrics:
    """Calculate overall benchmark metrics."""

    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.coverage_score >= 0.6)

    avg_coverage = sum(r.coverage_score for r in results) / total_tests
    avg_latency = sum(r.latency_ms for r in results) / total_tests
    total_tokens = sum(r.tokens_generated for r in results)
    avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0

    # Calculate per-category scores
    categories = set(r.category for r in results)
    category_scores = {}
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        category_scores[cat] = sum(r.coverage_score for r in cat_results) / len(cat_results)

    return BenchmarkMetrics(
        total_tests=total_tests,
        passed_tests=passed_tests,
        avg_coverage_score=avg_coverage,
        avg_latency_ms=avg_latency,
        avg_tokens_per_second=avg_tokens_per_sec,
        category_scores=category_scores,
        total_time_seconds=total_time,
    )


def run_performance_benchmark(llm: LLM, num_prompts: int = 50) -> dict:
    """Run throughput and latency benchmark."""

    print(f"\nRunning performance benchmark with {num_prompts} prompts...")

    # Sample medical prompts for benchmarking
    benchmark_prompts = [
        "What are the symptoms of pneumonia?",
        "Explain the mechanism of beta blockers.",
        "List differential diagnoses for abdominal pain.",
        "What is the treatment for hypertension?",
        "Describe the pathophysiology of heart failure.",
    ]

    # Create batch of prompts
    prompts = [format_medical_prompt(benchmark_prompts[i % len(benchmark_prompts)]) for i in range(num_prompts)]

    sampling_params = SamplingParams(temperature=0.7, max_tokens=256, top_p=0.9)

    # Warmup
    print("Warming up...")
    _ = llm.generate(prompts[:5], sampling_params)

    # Benchmark
    print("Running benchmark...")
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    return {
        "num_prompts": num_prompts,
        "total_time_seconds": total_time,
        "throughput_prompts_per_sec": num_prompts / total_time,
        "total_output_tokens": total_tokens,
        "throughput_tokens_per_sec": total_tokens / total_time,
        "avg_latency_ms": (total_time / num_prompts) * 1000,
    }


def print_report(metrics: BenchmarkMetrics, perf_results: Optional[dict] = None):
    """Print validation report."""

    print("\n" + "=" * 70)
    print("VALIDATION REPORT - LLaMA3-8B-GPTQ-4bit Medical Model")
    print("=" * 70)

    print(f"\n{'QUALITY METRICS':^70}")
    print("-" * 70)
    print(f"Total Test Cases:      {metrics.total_tests}")
    print(f"Passed (>=60% coverage): {metrics.passed_tests} ({metrics.passed_tests/metrics.total_tests:.1%})")
    print(f"Average Coverage Score: {metrics.avg_coverage_score:.1%}")
    print(f"Average Latency:        {metrics.avg_latency_ms:.0f}ms")

    print(f"\n{'CATEGORY BREAKDOWN':^70}")
    print("-" * 70)
    for category, score in sorted(metrics.category_scores.items()):
        bar = "=" * int(score * 40)
        print(f"{category:25s} {score:6.1%} |{bar}")

    if perf_results:
        print(f"\n{'PERFORMANCE BENCHMARK':^70}")
        print("-" * 70)
        print(f"Throughput:             {perf_results['throughput_prompts_per_sec']:.2f} prompts/sec")
        print(f"Token Generation:       {perf_results['throughput_tokens_per_sec']:.0f} tokens/sec")
        print(f"Average Latency:        {perf_results['avg_latency_ms']:.0f}ms per prompt")

    print("\n" + "=" * 70)


def save_results(
    results: list[ValidationResult],
    metrics: BenchmarkMetrics,
    perf_results: Optional[dict],
    output_file: str,
):
    """Save results to JSON file."""

    output = {
        "metrics": asdict(metrics),
        "performance_benchmark": perf_results,
        "detailed_results": [asdict(r) for r in results],
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Validate LLaMA3-8B-GPTQ medical model")
    parser.add_argument("--model-id", type=str, default=MODEL_ID, help="HuggingFace model ID")
    parser.add_argument(
        "--test-file",
        type=str,
        default="medical_test_cases.json",
        help="Path to test cases JSON",
    )
    parser.add_argument("--output", type=str, default="validation_results.json", help="Output file for results")
    parser.add_argument("--gpu-memory", type=float, default=0.85, help="GPU memory utilization")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--skip-perf", action="store_true", help="Skip performance benchmark")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Load test cases
    print(f"Loading test cases from: {args.test_file}")
    test_data = load_test_cases(args.test_file)
    test_cases = test_data["test_cases"]
    print(f"Loaded {len(test_cases)} test cases")

    # Load model
    print(f"\nLoading model: {args.model_id}")
    llm = LLM(
        model=args.model_id,
        gpu_memory_utilization=args.gpu_memory,
        quantization="gptq",
        dtype="half",
        trust_remote_code=True,
    )

    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=args.max_tokens,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )

    # Run validation
    print("\nRunning validation tests...")
    start_time = time.perf_counter()
    results = run_validation(llm, test_cases, sampling_params, verbose=args.verbose)
    total_time = time.perf_counter() - start_time

    # Calculate metrics
    metrics = calculate_metrics(results, total_time)

    # Run performance benchmark
    perf_results = None
    if not args.skip_perf:
        perf_results = run_performance_benchmark(llm)

    # Print and save results
    print_report(metrics, perf_results)
    save_results(results, metrics, perf_results, args.output)


if __name__ == "__main__":
    main()
