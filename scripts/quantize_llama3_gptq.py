#!/usr/bin/env python3
"""End-to-end GPTQ quantization script for Llama-3 models"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from innova_llama3_gptq import quantize_llama3_gptq
from scripts.utils import setup_logging, create_timestamp, get_hardware_info, save_json


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Llama-3 models using GPTQ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Hugging Face model ID or path (e.g., meta-llama/Meta-Llama-3-8B-Instruct)"
    )

    # Quantization arguments
    parser.add_argument(
        "--bits",
        type=int,
        choices=[3, 4, 8],
        default=4,
        help="Number of bits for quantization"
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Group size for quantization"
    )
    parser.add_argument(
        "--desc-act",
        action="store_true",
        default=True,
        help="Use activation order for quantization"
    )
    parser.add_argument(
        "--no-desc-act",
        dest="desc_act",
        action="store_false",
        help="Disable activation order"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext2",
        help="Calibration dataset (wikitext2, c4, ptb, or path to .jsonl)"
    )
    parser.add_argument(
        "--max-calib-samples",
        type=int,
        default=512,
        help="Maximum number of calibration samples"
    )

    # Output arguments
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for quantized model"
    )
    parser.add_argument(
        "--use-safetensors",
        action="store_true",
        default=True,
        help="Save model in safetensors format"
    )

    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for quantization"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code in model"
    )
    parser.add_argument(
        "--auth-token",
        type=str,
        default=None,
        help="Hugging Face authentication token"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging("quantize")

    # Set output directory if not specified
    if args.out_dir is None:
        timestamp = create_timestamp()
        model_name = args.model_id.split("/")[-1]
        args.out_dir = f"artifacts/gptq/{model_name}_{args.bits}bit_{timestamp}"

    logger.info("Starting GPTQ quantization")
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Configuration: {args.bits}-bit, group_size={args.group_size}")

    # Log hardware info
    hw_info = get_hardware_info()
    logger.info(f"Hardware: {hw_info}")

    try:
        # Run quantization
        output_dir = quantize_llama3_gptq(
            model_id=args.model_id,
            bits=args.bits,
            group_size=args.group_size,
            desc_act=args.desc_act,
            calib_dataset=args.dataset,
            max_calib_samples=args.max_calib_samples,
            out_dir=args.out_dir,
            use_safetensors=args.use_safetensors,
            seed=args.seed,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            auth_token=args.auth_token
        )

        logger.info(f"Quantization successful! Model saved to: {output_dir}")

        # Save hardware info
        save_json(hw_info, Path(output_dir) / "hardware_info.json")

        return 0

    except Exception as e:
        logger.error(f"Quantization failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())