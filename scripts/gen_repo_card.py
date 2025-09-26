#!/usr/bin/env python3
"""Generate repository model card from evaluation results"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.export_hf_gptq import build_model_card
from scripts.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Generate model card for GPTQ model"
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to model directory"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Base model ID"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="innova/llama3-8b-instruct-gptq",
        help="Repository ID"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Path to evaluation results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for model card (default: model-dir/README.md)"
    )

    args = parser.parse_args()
    logger = setup_logging("gen-card")

    model_path = Path(args.model_dir)
    if not model_path.exists():
        logger.error(f"Model directory not found: {model_path}")
        return 1

    results_path = Path(args.results_dir) if args.results_dir else None

    logger.info("Generating model card...")
    model_card = build_model_card(
        model_dir=model_path,
        base_model=args.base_model,
        repo_id=args.repo_id,
        results_path=results_path
    )

    output_path = Path(args.output) if args.output else model_path / "README.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(model_card)

    logger.info(f"Model card saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())