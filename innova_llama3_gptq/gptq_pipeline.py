"""Core GPTQ quantization pipeline for Llama-3 models"""

import json
import logging
import os
import random
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

import torch
from accelerate import init_empty_weights
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class GPTQConfig:
    """GPTQ quantization configuration"""
    bits: int = 4
    group_size: int = 128
    desc_act: bool = True
    sym: bool = True
    true_sequential: bool = True
    use_cuda_fp16: bool = True
    model_seqlen: int = 2048
    batch_size: int = 1
    use_fast: bool = True
    use_triton: bool = False
    use_exllama: bool = True
    cache_examples_on_gpu: bool = False
    damp_percent: float = 0.01

    def to_quantize_config(self) -> BaseQuantizeConfig:
        """Convert to auto-gptq BaseQuantizeConfig"""
        return BaseQuantizeConfig(
            bits=self.bits,
            group_size=self.group_size,
            desc_act=self.desc_act,
            sym=self.sym,
            true_sequential=self.true_sequential,
            use_cuda_fp16=self.use_cuda_fp16,
            model_seqlen=self.model_seqlen,
            damp_percent=self.damp_percent
        )


class CalibrationDataset:
    """Manages calibration dataset loading and preprocessing"""

    DATASET_CONFIGS = {
        "wikitext2": {
            "name": "wikitext",
            "config": "wikitext-2-raw-v1",
            "split": "test",
            "text_column": "text"
        },
        "c4": {
            "name": "c4",
            "config": "en",
            "split": "validation",
            "text_column": "text"
        },
        "ptb": {
            "name": "ptb_text_only",
            "config": None,
            "split": "test",
            "text_column": "sentence"
        }
    }

    def __init__(
        self,
        dataset_name: str = "wikitext2",
        tokenizer: AutoTokenizer = None,
        max_samples: int = 512,
        max_length: int = 2048,
        seed: int = 42
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.max_length = max_length
        self.seed = seed
        random.seed(seed)

    def load_from_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Load calibration data from JSONL file"""
        examples = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
                if len(examples) >= self.max_samples:
                    break
        return examples

    def load_from_hub(self) -> List[Dict[str, Any]]:
        """Load calibration data from Hugging Face datasets"""
        if self.dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        config = self.DATASET_CONFIGS[self.dataset_name]
        dataset = load_dataset(
            config["name"],
            config.get("config"),
            split=config["split"]
        )

        text_column = config["text_column"]
        examples = []

        for item in tqdm(dataset, desc=f"Loading {self.dataset_name}"):
            text = item[text_column]
            if text and len(text.strip()) > 0:
                examples.append({"text": text})
            if len(examples) >= self.max_samples:
                break

        return examples

    def preprocess(self, examples: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        """Tokenize and prepare examples for quantization"""
        processed = []

        for example in tqdm(examples, desc="Preprocessing"):
            text = example.get("text", "")
            if not text:
                continue

            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=False
            )

            if tokens.input_ids.shape[1] >= 10:  # Skip very short sequences
                processed.append({
                    "input_ids": tokens.input_ids[0],
                    "attention_mask": tokens.attention_mask[0]
                })

        return processed


def quantize_llama3_gptq(
    model_id: str,
    bits: int = 4,
    group_size: int = 128,
    desc_act: bool = True,
    calib_dataset: str = "wikitext2",
    max_calib_samples: int = 512,
    out_dir: str = "./artifacts/gptq/run",
    use_safetensors: bool = True,
    seed: int = 42,
    device: str = "cuda",
    trust_remote_code: bool = False,
    auth_token: Optional[str] = None
) -> str:
    """
    Main GPTQ quantization pipeline for Llama-3 models

    Args:
        model_id: Hugging Face model ID or local path
        bits: Number of bits for quantization (3, 4, 8)
        group_size: Group size for quantization
        desc_act: Whether to use activation order
        calib_dataset: Calibration dataset name or path to JSONL
        max_calib_samples: Maximum number of calibration samples
        out_dir: Output directory for quantized model
        use_safetensors: Save in safetensors format
        seed: Random seed
        device: Device to use for quantization
        trust_remote_code: Trust remote code in model
        auth_token: Hugging Face auth token

    Returns:
        Path to quantized model directory
    """
    torch.manual_seed(seed)
    random.seed(seed)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting GPTQ quantization for {model_id}")
    logger.info(f"Config: {bits}-bit, group_size={group_size}, desc_act={desc_act}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=trust_remote_code,
        token=auth_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare calibration dataset
    logger.info(f"Preparing calibration dataset: {calib_dataset}")
    calib_data = CalibrationDataset(
        dataset_name=calib_dataset if not calib_dataset.endswith('.jsonl') else 'custom',
        tokenizer=tokenizer,
        max_samples=max_calib_samples,
        seed=seed
    )

    if calib_dataset.endswith('.jsonl'):
        examples = calib_data.load_from_file(calib_dataset)
    else:
        examples = calib_data.load_from_hub()

    calib_examples = calib_data.preprocess(examples)
    logger.info(f"Loaded {len(calib_examples)} calibration samples")

    # Setup quantization config
    quantize_config = GPTQConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act
    ).to_quantize_config()

    # Load and quantize model
    logger.info("Loading model for quantization...")
    model = AutoGPTQForCausalLM.from_pretrained(
        model_id,
        quantize_config=quantize_config,
        device_map="auto",
        use_safetensors=use_safetensors,
        trust_remote_code=trust_remote_code,
        token=auth_token
    )

    logger.info("Starting quantization...")
    model.quantize(
        calib_examples,
        batch_size=1,
        use_triton=False,
        cache_examples_on_gpu=False
    )

    # Save quantized model
    logger.info(f"Saving quantized model to {out_path}")
    model.save_quantized(
        out_path,
        use_safetensors=use_safetensors
    )

    # Save tokenizer
    tokenizer.save_pretrained(out_path)

    # Save generation config if exists
    try:
        from transformers import GenerationConfig
        gen_config = GenerationConfig.from_pretrained(
            model_id,
            token=auth_token
        )
        gen_config.save_pretrained(out_path)
    except:
        logger.warning("Could not save generation config")

    # Save quantization config
    quantize_config_dict = {
        "bits": bits,
        "group_size": group_size,
        "desc_act": desc_act,
        "sym": True,
        "true_sequential": True,
        "quant_method": "gptq",
        "model_name_or_path": model_id,
        "model_file_base_name": "model"
    }

    with open(out_path / "quantize_config.json", "w") as f:
        json.dump(quantize_config_dict, f, indent=2)

    # Save run metadata
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    ).stdout.strip()[:8] if subprocess.run(["git", "status"], capture_output=True).returncode == 0 else "unknown"

    metadata = {
        "model_id": model_id,
        "quantization": {
            "bits": bits,
            "group_size": group_size,
            "desc_act": desc_act,
            "method": "gptq"
        },
        "calibration": {
            "dataset": calib_dataset,
            "samples": len(calib_examples)
        },
        "timestamp": datetime.now().isoformat(),
        "git_sha": git_sha,
        "output_dir": str(out_path.absolute())
    }

    with open(out_path / "quantization_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Quantization complete! Model saved to {out_path}")
    return str(out_path)