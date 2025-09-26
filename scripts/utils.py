"""Shared utilities for scripts"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import psutil
import torch
from huggingface_hub import HfApi, login


def setup_logging(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Setup logger with consistent formatting"""
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def get_hardware_info() -> Dict[str, Any]:
    """Get current hardware information"""
    info = {
        "cpu": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "platform": sys.platform
    }

    if torch.cuda.is_available():
        info.update({
            "gpu": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        })

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            info["driver_version"] = result.stdout.strip()
    except:
        pass

    return info


def get_git_info() -> Dict[str, str]:
    """Get current git repository information"""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()

        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()

        return {"sha": sha[:8], "branch": branch}
    except:
        return {"sha": "unknown", "branch": "unknown"}


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Save dictionary to JSON file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load JSON file"""
    with open(filepath, "r") as f:
        return json.load(f)


def authenticate_hf(token: Optional[str] = None) -> HfApi:
    """Authenticate with Hugging Face Hub"""
    token = token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    return HfApi()


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_model_size(model_dir: Path) -> Dict[str, Any]:
    """Calculate model size information"""
    total_size = 0
    file_count = 0
    extensions = {}

    for file_path in model_dir.rglob("*"):
        if file_path.is_file():
            size = file_path.stat().st_size
            total_size += size
            file_count += 1

            ext = file_path.suffix.lower()
            if ext:
                extensions[ext] = extensions.get(ext, 0) + size

    return {
        "total_size_bytes": total_size,
        "total_size": format_size(total_size),
        "file_count": file_count,
        "extensions": {k: format_size(v) for k, v in extensions.items()}
    }


def validate_model_dir(model_dir: Path) -> bool:
    """Validate that directory contains a valid model"""
    required_files = [
        "config.json",
        "tokenizer_config.json"
    ]

    for file in required_files:
        if not (model_dir / file).exists():
            return False

    # Check for model weights
    has_weights = any([
        list(model_dir.glob("*.safetensors")),
        list(model_dir.glob("*.bin")),
        list(model_dir.glob("*.pt"))
    ])

    return has_weights


def create_timestamp() -> str:
    """Create timestamp string for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")