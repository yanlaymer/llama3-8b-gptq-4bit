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
from gptqmodel import GPTQModel, QuantizeConfig
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
    # T4-specific optimizations
    offload_to_disk: bool = True
    v2: bool = False
    auto_gc: bool = True
    buffered_fwd: bool = True

    def to_quantize_config(self) -> QuantizeConfig:
        """Convert to gptqmodel QuantizeConfig"""
        return QuantizeConfig(
            bits=self.bits,
            group_size=self.group_size,
            desc_act=self.desc_act,
            sym=self.sym,
            true_sequential=self.true_sequential,
            damp_percent=self.damp_percent
        )

    @classmethod
    def for_t4_gpu(cls) -> "GPTQConfig":
        """Create T4-optimized configuration"""
        return cls(
            bits=4,
            group_size=128,
            desc_act=True,
            sym=True,
            true_sequential=True,
            use_cuda_fp16=True,
            model_seqlen=4096,  # Increased for better calibration
            batch_size=1,
            use_fast=True,
            use_triton=False,
            use_exllama=True,
            cache_examples_on_gpu=False,  # Critical for T4
            damp_percent=0.01,
            offload_to_disk=True,  # Saves ~73.5% CPU memory
            v2=False,  # v2 requires 2-4x more VRAM
            auto_gc=True,
            buffered_fwd=True
        )


class CalibrationDataset:
    """Manages calibration dataset loading and preprocessing"""

    DATASET_CONFIGS = {
        # General domain datasets
        "wikitext2": {
            "name": "wikitext",
            "config": "wikitext-2-raw-v1",
            "split": "test",
            "text_column": "text",
            "format_fn": None
        },
        "c4": {
            "name": "c4",
            "config": "en",
            "split": "validation",
            "text_column": "text",
            "format_fn": None
        },
        "ptb": {
            "name": "ptb_text_only",
            "config": None,
            "split": "test",
            "text_column": "sentence",
            "format_fn": None
        },
        # Medical domain datasets
        "pubmedqa": {
            "name": "qiaojin/PubMedQA",
            "config": "pqa_labeled",
            "split": "train",
            "text_column": None,  # Custom formatting needed
            "format_fn": "format_pubmedqa"
        },
        "medqa": {
            "name": "bigbio/med_qa",
            "config": "med_qa_en_bigbio_qa",
            "split": "train",
            "text_column": None,
            "format_fn": "format_medqa"
        },
        "pmc_patients": {
            "name": "AGBonnet/augmented-clinical-notes",
            "config": None,
            "split": "train",
            "text_column": "text",
            "format_fn": None
        },
        "asclepius_notes": {
            "name": "starmpcc/Asclepius-Synthetic-Clinical-Notes",
            "config": None,
            "split": "train",
            "text_column": "text",
            "format_fn": None
        }
    }

    def __init__(
        self,
        dataset_name: str = "wikitext2",
        tokenizer: AutoTokenizer = None,
        max_samples: int = 512,
        max_length: int = 2048,
        min_length: int = 256,
        seed: int = 42
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.max_length = max_length
        self.min_length = min_length
        self.seed = seed
        random.seed(seed)

    @staticmethod
    def format_pubmedqa(item: Dict[str, Any]) -> str:
        """Format PubMedQA item into calibration text"""
        question = item.get("QUESTION", "")
        long_answer = item.get("LONG_ANSWER", "")
        # Combine question and answer for medical reasoning context
        return f"{question}\n\n{long_answer}"

    @staticmethod
    def format_medqa(item: Dict[str, Any]) -> str:
        """Format MedQA item into calibration text"""
        question = item.get("question", "")
        choices_dict = item.get("choices", [])

        # Format choices
        if isinstance(choices_dict, list):
            choices_text = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices_dict)])
        else:
            choices_text = ""

        # Combine question and choices
        return f"{question}\n\nOptions:\n{choices_text}" if choices_text else question

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

        text_column = config.get("text_column")
        format_fn_name = config.get("format_fn")
        examples = []

        # Get format function if specified
        format_fn = None
        if format_fn_name:
            format_fn = getattr(self, format_fn_name, None)

        # Try to load more than needed in case some are filtered out
        target_samples = min(self.max_samples * 2, len(dataset))

        for item in tqdm(dataset.select(range(target_samples)), desc=f"Loading {self.dataset_name}"):
            # Extract text using either format function or text column
            if format_fn:
                text = format_fn(item)
            elif text_column:
                text = item[text_column]
            else:
                logger.warning(f"No text_column or format_fn for {self.dataset_name}")
                continue

            if text and len(text.strip()) > 50:  # Filter very short texts
                examples.append({"text": text})
            if len(examples) >= self.max_samples:
                break

        if len(examples) < self.max_samples:
            logger.warning(f"Only loaded {len(examples)} samples, expected {self.max_samples}")

        return examples

    def preprocess(self, examples: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        """Tokenize and prepare examples for quantization"""
        processed = []
        concatenated_texts = []

        # Concatenate shorter texts to reach target length
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

            # Only include sequences that meet minimum length requirement
            if tokens.input_ids.shape[1] >= self.min_length:
                processed.append({
                    "input_ids": tokens.input_ids[0],
                    "attention_mask": tokens.attention_mask[0]
                })
            elif tokens.input_ids.shape[1] >= 50:  # Still useful for concatenation
                concatenated_texts.append(text)

        # If we don't have enough long sequences, concatenate shorter ones
        if len(processed) < self.max_samples // 2 and concatenated_texts:
            logger.info(f"Concatenating shorter texts to create longer sequences...")

            combined_text = ""
            for text in concatenated_texts:
                combined_text += text + " "
                if len(combined_text) > self.max_length * 2:  # Enough for one sequence
                    tokens = self.tokenizer(
                        combined_text,
                        return_tensors="pt",
                        max_length=self.max_length,
                        truncation=True,
                        padding=False
                    )

                    if tokens.input_ids.shape[1] >= self.min_length:
                        processed.append({
                            "input_ids": tokens.input_ids[0],
                            "attention_mask": tokens.attention_mask[0]
                        })

                    combined_text = ""

                    if len(processed) >= self.max_samples:
                        break

        avg_length = sum(p["input_ids"].shape[0] for p in processed) / len(processed) if processed else 0
        logger.info(f"Processed {len(processed)} sequences with average length {avg_length:.1f}")

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
    auth_token: Optional[str] = None,
    use_t4_optimizations: bool = False
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
    import gc
    import os

    torch.manual_seed(seed)
    random.seed(seed)

    # Setup memory optimizations for T4 or low-VRAM environments
    if use_t4_optimizations or "KAGGLE_KERNEL_RUN_TYPE" in os.environ or "COLAB_GPU" in os.environ:
        logger.info("🚀 Applying T4/Kaggle/Colab memory optimizations...")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🔧 GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB VRAM)")
        torch.cuda.empty_cache()

    logger.info(f"Starting GPTQ quantization for {model_id}")
    logger.info(f"Config: {bits}-bit, group_size={group_size}, desc_act={desc_act}")

    # Apply T4 optimizations if needed
    if use_t4_optimizations or (torch.cuda.is_available() and gpu_memory <= 16.5):
        logger.info("📝 Using T4-optimized settings for low VRAM")
        use_t4_optimizations = True

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
        max_length=4096 if use_t4_optimizations else 2048,
        min_length=256,
        seed=seed
    )

    if calib_dataset.endswith('.jsonl'):
        examples = calib_data.load_from_file(calib_dataset)
    else:
        examples = calib_data.load_from_hub()

    calib_examples = calib_data.preprocess(examples)
    logger.info(f"Loaded {len(calib_examples)} calibration samples")

    # Setup quantization config
    if use_t4_optimizations:
        quantize_config = GPTQConfig.for_t4_gpu().to_quantize_config()
        logger.info("📊 Using T4-optimized quantization config")
    else:
        quantize_config = GPTQConfig(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act
        ).to_quantize_config()

    # Load and quantize model
    logger.info("Loading model for quantization...")
    model = GPTQModel.load(
        model_id,
        quantize_config=quantize_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        token=auth_token
    )

    logger.info("Starting quantization...")

    # Memory management during quantization
    torch.cuda.empty_cache()
    gc.collect()

    try:
        # Use optimized quantization parameters
        quantization_kwargs = {
            "batch_size": 1,
            "calibration_dataset_min_length": 256 if use_t4_optimizations else 10,
            "auto_gc": True if use_t4_optimizations else False
        }

        # Add cache_examples_on_gpu=False for T4 optimizations
        if use_t4_optimizations:
            quantization_kwargs["cache_examples_on_gpu"] = False
            logger.info("💾 Using cache_examples_on_gpu=False for T4 optimization")

        model.quantize(calib_examples, **quantization_kwargs)

        # Clear memory after quantization
        torch.cuda.empty_cache()
        gc.collect()

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("🚨 GPU out of memory during quantization!")
            logger.error("💡 Try reducing max_calib_samples or using cache_examples_on_gpu=False")
            raise
        else:
            raise

    # Save quantized model
    logger.info(f"Saving quantized model to {out_path}")
    model.save(out_path)

    # Clear memory after saving
    torch.cuda.empty_cache()
    gc.collect()

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


def quantize_llama3_gptq_t4_optimized(
    model_id: str,
    out_dir: str = "./artifacts/gptq/t4_optimized",
    max_calib_samples: int = 256,
    auth_token: Optional[str] = None
) -> str:
    """
    T4-optimized GPTQ quantization with memory-efficient settings

    Perfect for Kaggle, Colab, and other T4 GPU environments

    Args:
        model_id: Hugging Face model ID
        out_dir: Output directory
        max_calib_samples: Number of calibration samples (reduced for speed)
        auth_token: HuggingFace token

    Returns:
        Path to quantized model
    """
    return quantize_llama3_gptq(
        model_id=model_id,
        bits=4,
        group_size=128,
        desc_act=True,
        calib_dataset="wikitext2",
        max_calib_samples=max_calib_samples,
        out_dir=out_dir,
        use_safetensors=True,
        seed=42,
        device="cuda",
        trust_remote_code=False,
        auth_token=auth_token,
        use_t4_optimizations=True
    )