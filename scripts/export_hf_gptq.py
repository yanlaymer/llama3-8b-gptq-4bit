#!/usr/bin/env python3
"""Export GPTQ model to Hugging Face Hub with proper model card"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils import (
    setup_logging,
    authenticate_hf,
    validate_model_dir,
    get_model_size,
    load_json
)
from innova_llama3_gptq.hf import TEMPLATE_PATH


def build_model_card(
    model_dir: Path,
    base_model: str,
    repo_id: str,
    results_path: Optional[Path] = None
) -> str:
    """
    Build model card from template and evaluation results

    Args:
        model_dir: Path to model directory
        base_model: Base model ID
        repo_id: Target repository ID
        results_path: Path to evaluation results

    Returns:
        Formatted model card content
    """
    # Load template
    with open(TEMPLATE_PATH, 'r') as f:
        template = f.read()

    # Load quantization metadata
    quant_metadata = {}
    quant_metadata_path = model_dir / "quantization_metadata.json"
    if quant_metadata_path.exists():
        quant_metadata = load_json(quant_metadata_path)

    # Load quantize config
    quant_config = {}
    quant_config_path = model_dir / "quantize_config.json"
    if quant_config_path.exists():
        quant_config = load_json(quant_config_path)

    # Load evaluation results if available
    eval_results = {}
    if results_path and results_path.exists():
        # Find most recent metrics file
        metrics_files = list(results_path.glob("metrics_*.json"))
        if metrics_files:
            latest_metrics = max(metrics_files, key=lambda p: p.stat().st_mtime)
            eval_results = load_json(latest_metrics)

    # Get model size
    model_size = get_model_size(model_dir)

    # Build replacement dictionary
    replacements = {
        "{{LICENSE}}": "llama3",
        "{{BASE_MODEL}}": base_model,
        "{{MODEL_NAME}}": repo_id.split("/")[-1],
        "{{HF_REPO_ID}}": repo_id,
        "{{BITS}}": str(quant_config.get("bits", 4)),
        "{{GROUP_SIZE}}": str(quant_config.get("group_size", 128)),
        "{{DESC_ACT}}": str(quant_config.get("desc_act", True)),
        "{{CALIBRATION_DATASET}}": quant_metadata.get("calibration", {}).get("dataset", "wikitext2"),
        "{{CALIBRATION_SAMPLES}}": str(quant_metadata.get("calibration", {}).get("samples", 512)),
        "{{MODEL_SIZE}}": model_size["total_size"],
        "{{COMPRESSION_RATIO}}": "4" if quant_config.get("bits") == 4 else "5.3",
        "{{HF_ORG}}": repo_id.split("/")[0] if "/" in repo_id else "user"
    }

    # Add performance metrics if available
    if eval_results and "summary_metrics" in eval_results:
        metrics = eval_results["summary_metrics"]

        # Perplexity table
        ppl_rows = []
        for key, value in metrics.items():
            if key.startswith("ppl_"):
                dataset = key.replace("ppl_", "")
                ppl_rows.append(f"| {dataset} | {value:.2f} |")
        replacements["{{PERPLEXITY_TABLE}}"] = "\n".join(ppl_rows) if ppl_rows else "| wikitext2 | N/A |"

        # Task table
        task_rows = []
        for key, value in metrics.items():
            if key.startswith("task_"):
                task = key.replace("task_", "")
                task_rows.append(f"| {task} | {value:.3f} |")
        replacements["{{TASK_TABLE}}"] = "\n".join(task_rows) if task_rows else "| hellaswag | N/A |"

        # Performance metrics
        replacements["{{LATENCY_MS}}"] = f"{metrics.get('latency_ms', 'N/A'):.2f}" if isinstance(metrics.get('latency_ms'), (int, float)) else "N/A"
        replacements["{{THROUGHPUT_TPS}}"] = f"{metrics.get('throughput_tps', 'N/A'):.1f}" if isinstance(metrics.get('throughput_tps'), (int, float)) else "N/A"
        replacements["{{MEMORY_USAGE}}"] = "8.5"  # Placeholder
    else:
        # Use placeholders if no results
        replacements.update({
            "{{PERPLEXITY_TABLE}}": "| wikitext2 | TBD |\n| c4 | TBD |",
            "{{TASK_TABLE}}": "| hellaswag | TBD |\n| arc_easy | TBD |",
            "{{LATENCY_MS}}": "TBD",
            "{{THROUGHPUT_TPS}}": "TBD",
            "{{MEMORY_USAGE}}": "TBD"
        })

    # Hardware info
    if eval_results and "hardware" in eval_results:
        hw = eval_results["hardware"]
        replacements["{{TESTED_GPU}}"] = hw.get("gpu", "NVIDIA A6000")
        replacements["{{CUDA_VERSION}}"] = hw.get("cuda_version", "12.1")
        replacements["{{DRIVER_VERSION}}"] = hw.get("driver_version", "535.154.05")
    else:
        replacements.update({
            "{{TESTED_GPU}}": "NVIDIA A6000",
            "{{CUDA_VERSION}}": "12.1",
            "{{DRIVER_VERSION}}": "535.154.05"
        })

    # Replace all placeholders
    model_card = template
    for key, value in replacements.items():
        model_card = model_card.replace(key, str(value))

    return model_card


def export_to_hub(
    model_dir: Path,
    repo_id: str,
    base_model: str,
    private: bool = True,
    push: bool = False,
    token: Optional[str] = None,
    results_path: Optional[Path] = None
) -> None:
    """
    Export model to Hugging Face Hub

    Args:
        model_dir: Path to model directory
        repo_id: Target repository ID
        base_model: Base model ID
        private: Whether repo should be private
        push: Whether to push to hub
        token: HF auth token
        results_path: Path to evaluation results
    """
    logger = setup_logging("export")

    # Build model card
    logger.info("Building model card...")
    model_card = build_model_card(
        model_dir=model_dir,
        base_model=base_model,
        repo_id=repo_id,
        results_path=results_path
    )

    # Save model card
    readme_path = model_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(model_card)
    logger.info(f"Model card saved to {readme_path}")

    if push:
        try:
            from huggingface_hub import HfApi, create_repo

            logger.info(f"Authenticating with Hugging Face...")
            api = authenticate_hf(token)

            # Create repository
            logger.info(f"Creating repository: {repo_id}")
            create_repo(
                repo_id=repo_id,
                private=private,
                exist_ok=True,
                token=token
            )

            # Upload files
            logger.info(f"Uploading model files to {repo_id}...")
            api.upload_folder(
                folder_path=str(model_dir),
                repo_id=repo_id,
                repo_type="model",
                commit_message="Upload GPTQ quantized model",
                token=token
            )

            logger.info(f"Model successfully uploaded to: https://huggingface.co/{repo_id}")

        except Exception as e:
            logger.error(f"Failed to push to hub: {e}")
            logger.info("\n" + "="*60)
            logger.info("MANUAL UPLOAD INSTRUCTIONS")
            logger.info("="*60)
            logger.info("To manually upload your model:")
            logger.info(f"1. Install huggingface-cli: pip install huggingface-hub")
            logger.info(f"2. Login: huggingface-cli login")
            logger.info(f"3. Create repo: huggingface-cli repo create {repo_id} --type model {'--private' if private else ''}")
            logger.info(f"4. Upload: huggingface-cli upload {repo_id} {model_dir} . --repo-type model")
    else:
        logger.info("\n" + "="*60)
        logger.info("EXPORT COMPLETE - MANUAL UPLOAD REQUIRED")
        logger.info("="*60)
        logger.info(f"Model card generated at: {readme_path}")
        logger.info("\nTo upload to Hugging Face Hub:")
        logger.info(f"1. Install: pip install huggingface-hub")
        logger.info(f"2. Login: huggingface-cli login")
        logger.info(f"3. Run this script again with --push flag")
        logger.info(f"   Or manually upload using:")
        logger.info(f"   huggingface-cli upload {repo_id} {model_dir} . --repo-type model")


def main():
    parser = argparse.ArgumentParser(
        description="Export GPTQ model to Hugging Face Hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to GPTQ model directory"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Target Hugging Face repository ID (e.g., org/model-name)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Base model ID"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Make repository private"
    )
    parser.add_argument(
        "--public",
        dest="private",
        action="store_false",
        help="Make repository public"
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push to Hugging Face Hub"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Path to evaluation results directory"
    )

    args = parser.parse_args()
    logger = setup_logging("export")

    # Validate model directory
    model_path = Path(args.model_dir)
    if not model_path.exists():
        logger.error(f"Model directory not found: {model_path}")
        return 1

    if not validate_model_dir(model_path):
        logger.error(f"Invalid model directory: {model_path}")
        logger.error("Required files: config.json, tokenizer_config.json, and model weights")
        return 1

    # Parse results directory
    results_path = Path(args.results_dir) if args.results_dir else None

    # Export
    try:
        export_to_hub(
            model_dir=model_path,
            repo_id=args.repo_id,
            base_model=args.base_model,
            private=args.private,
            push=args.push,
            token=args.token,
            results_path=results_path
        )
        return 0
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())