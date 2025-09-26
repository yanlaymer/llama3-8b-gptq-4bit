"""Hugging Face integration utilities"""

from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent / "repo_card_template.md"
PROCESSING_CONFIG_PATH = Path(__file__).parent / "processing.json"

__all__ = ["TEMPLATE_PATH", "PROCESSING_CONFIG_PATH"]