"""Perplexity evaluation for quantized models"""

import logging
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_name: str = "wikitext2",
    max_samples: Optional[int] = None,
    max_length: int = 2048,
    batch_size: int = 1,
    device: str = "cuda"
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute perplexity on a dataset

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        dataset_name: Dataset to use (wikitext2, c4, ptb)
        max_samples: Maximum number of samples to evaluate
        max_length: Maximum sequence length
        batch_size: Batch size for evaluation
        device: Device to use

    Returns:
        Tuple of (perplexity, metrics_dict)
    """
    model.eval()

    # Load dataset
    if dataset_name == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text_column = "text"
    elif dataset_name == "c4":
        dataset = load_dataset("c4", "en", split="validation", streaming=True)
        text_column = "text"
        if max_samples:
            dataset = dataset.take(max_samples)
    elif dataset_name == "ptb":
        dataset = load_dataset("ptb_text_only", split="test")
        text_column = "sentence"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Prepare texts
    texts = []
    for item in dataset:
        text = item[text_column]
        if text and len(text.strip()) > 0:
            texts.append(text)
            if max_samples and len(texts) >= max_samples:
                break

    logger.info(f"Evaluating perplexity on {len(texts)} samples from {dataset_name}")

    # Tokenize and compute perplexity
    total_loss = 0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing perplexity"):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            encodings = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            ).to(device)

            # Skip if sequence is too short
            if encodings.input_ids.shape[1] < 2:
                continue

            # Forward pass
            outputs = model(
                input_ids=encodings.input_ids,
                attention_mask=encodings.attention_mask,
                labels=encodings.input_ids
            )

            # Accumulate loss
            batch_loss = outputs.loss.item()
            batch_tokens = encodings.attention_mask.sum().item()

            total_loss += batch_loss * batch_tokens
            total_tokens += batch_tokens
            num_batches += 1

    # Calculate perplexity
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    metrics = {
        "perplexity": perplexity,
        "loss": avg_loss,
        "total_tokens": total_tokens,
        "num_samples": len(texts),
        "num_batches": num_batches,
        "dataset": dataset_name
    }

    return perplexity, metrics


def evaluate_perplexity_suite(
    model_path: str,
    datasets: list = ["wikitext2", "c4"],
    max_samples_per_dataset: int = 1000,
    device: str = "cuda",
    trust_remote_code: bool = False
) -> Dict[str, Any]:
    """
    Evaluate perplexity on multiple datasets

    Args:
        model_path: Path to the model
        datasets: List of dataset names
        max_samples_per_dataset: Max samples per dataset
        device: Device to use
        trust_remote_code: Trust remote code

    Returns:
        Dictionary with perplexity results
    """
    logger.info(f"Loading model from {model_path}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=trust_remote_code
    )

    results = {}

    for dataset_name in datasets:
        try:
            logger.info(f"Evaluating on {dataset_name}")
            ppl, metrics = compute_perplexity(
                model=model,
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                max_samples=max_samples_per_dataset,
                device=device
            )
            results[dataset_name] = metrics
            logger.info(f"{dataset_name} perplexity: {ppl:.2f}")
        except Exception as e:
            logger.error(f"Failed to evaluate on {dataset_name}: {e}")
            results[dataset_name] = {"error": str(e)}

    return results