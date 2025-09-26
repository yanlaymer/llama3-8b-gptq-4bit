#!/usr/bin/env python3
"""Comprehensive evaluation script for GPTQ quantized models"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from innova_llama3_gptq.evals import (
    evaluate_perplexity_suite,
    evaluate_with_lm_eval,
    measure_inference_latency,
    create_results_summary,
    save_results,
    generate_markdown_report
)
from scripts.utils import (
    setup_logging,
    get_hardware_info,
    get_git_info,
    create_timestamp,
    validate_model_dir,
    get_model_size,
    load_json
)


def load_eval_config(config_path: Path) -> dict:
    """Load evaluation configuration from YAML file"""
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        # Fallback to JSON if YAML not available
        return load_json(config_path)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GPTQ quantized models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to quantized model directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval.yaml",
        help="Path to evaluation config file"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to save results"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Override tasks to evaluate (e.g., hellaswag arc_easy)"
    )
    parser.add_argument(
        "--skip-perplexity",
        action="store_true",
        help="Skip perplexity evaluation"
    )
    parser.add_argument(
        "--skip-tasks",
        action="store_true",
        help="Skip task evaluation"
    )
    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="Skip latency measurement"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for evaluation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code in model"
    )

    args = parser.parse_args()
    logger = setup_logging("eval")

    # Validate model directory
    model_path = Path(args.model_dir)
    if not model_path.exists():
        logger.error(f"Model directory not found: {model_path}")
        return 1

    if not validate_model_dir(model_path):
        logger.error(f"Invalid model directory: {model_path}")
        return 1

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = load_eval_config(config_path)
        logger.info(f"Loaded config from {config_path}")
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config = {
            "perplexity": {"datasets": ["wikitext2"], "max_samples": 1000},
            "tasks": {"names": ["hellaswag"], "num_fewshot": 0},
            "latency": {"batch_sizes": [1, 4], "sequence_length": 512}
        }

    # Override tasks if specified
    if args.tasks:
        config["tasks"]["names"] = args.tasks

    # Set results directory
    if args.results_dir is None:
        timestamp = create_timestamp()
        args.results_dir = f"results/{timestamp}"

    results_path = Path(args.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Evaluating model: {model_path}")
    logger.info(f"Results will be saved to: {results_path}")

    # Gather model information
    model_info = {
        "path": str(model_path.absolute()),
        "size": get_model_size(model_path)
    }

    # Try to load quantization metadata
    quant_metadata_path = model_path / "quantization_metadata.json"
    if quant_metadata_path.exists():
        quant_metadata = load_json(quant_metadata_path)
        model_info.update(quant_metadata)

    # Get hardware info
    hardware_info = get_hardware_info()
    git_info = get_git_info()

    # Initialize results
    all_results = {
        "model": model_info,
        "hardware": hardware_info,
        "git": git_info
    }

    # Perplexity evaluation
    if not args.skip_perplexity:
        logger.info("Starting perplexity evaluation...")
        try:
            ppl_results = evaluate_perplexity_suite(
                model_path=str(model_path),
                datasets=config.get("perplexity", {}).get("datasets", ["wikitext2"]),
                max_samples_per_dataset=config.get("perplexity", {}).get("max_samples", 1000),
                device=args.device,
                trust_remote_code=args.trust_remote_code
            )
            all_results["perplexity"] = ppl_results
        except Exception as e:
            logger.error(f"Perplexity evaluation failed: {e}")
            all_results["perplexity"] = {"error": str(e)}

    # Task evaluation
    if not args.skip_tasks:
        logger.info("Starting task evaluation...")
        try:
            task_results = evaluate_with_lm_eval(
                model_path=str(model_path),
                tasks=config.get("tasks", {}).get("names", ["hellaswag"]),
                num_fewshot=config.get("tasks", {}).get("num_fewshot", 0),
                batch_size=args.batch_size,
                device=args.device,
                trust_remote_code=args.trust_remote_code
            )
            all_results["tasks"] = task_results
        except Exception as e:
            logger.error(f"Task evaluation failed: {e}")
            all_results["tasks"] = {"error": str(e)}

    # Latency measurement
    if not args.skip_latency:
        logger.info("Measuring inference latency...")
        try:
            latency_results = measure_inference_latency(
                model_path=str(model_path),
                batch_sizes=config.get("latency", {}).get("batch_sizes", [1, 4, 8]),
                sequence_length=config.get("latency", {}).get("sequence_length", 512),
                num_iterations=config.get("latency", {}).get("iterations", 10),
                device=args.device,
                trust_remote_code=args.trust_remote_code
            )
            all_results["latency"] = latency_results
        except Exception as e:
            logger.error(f"Latency measurement failed: {e}")
            all_results["latency"] = {"error": str(e)}

    # Create summary
    summary = create_results_summary(
        perplexity_results=all_results.get("perplexity", {}),
        task_results=all_results.get("tasks", {}),
        latency_results=all_results.get("latency", {}),
        model_info=model_info,
        hardware_info=hardware_info
    )

    # Save results
    logger.info("Saving results...")
    save_results(summary, results_path, format="json")
    save_results(summary, results_path, format="markdown")

    # Print summary
    logger.info("\n" + "="*50)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*50)

    if "summary_metrics" in summary:
        logger.info("\nKey Metrics:")
        for metric, value in summary["summary_metrics"].items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.3f}")
            else:
                logger.info(f"  {metric}: {value}")

    logger.info(f"\nFull results saved to: {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())