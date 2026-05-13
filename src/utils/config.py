# src/utils/config.py
"""
Centralised config loader.
Reads configs/preprocessing.yaml and merges .env secrets.
All other modules import get_config() from here — never read files directly.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env once at import time
load_dotenv()

_CONFIG_PATH = Path("configs/preprocessing.yaml")
_cached_config: dict[str, Any] | None = None


def get_config(config_path: Path = _CONFIG_PATH) -> dict[str, Any]:
    """Load and cache the YAML config. Call this anywhere you need settings."""
    global _cached_config
    
    if _cached_config is not None:
        return _cached_config
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with config_path.open("r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f) or {}
    _cached_config = config
    
    return config


def get_env(key: str, default: str | None = None) -> str | None:
    """Fetch a secret from environment variables (loaded from .env)."""
    return os.getenv(key, default)