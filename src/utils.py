"""
src/utils.py
Shared utility functions: seeding, device resolution, config loading,
pretty logging, and path helpers.
"""
from __future__ import annotations

import os
import random
import logging
from pathlib import Path
from typing import Union

import numpy as np
import yaml
from omegaconf import OmegaConf, DictConfig
from rich.logging import RichHandler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a rich-formatted logger."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # deterministic may be slower; toggle via env var
            if os.environ.get("DETERMINISTIC", "0") == "1":
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def resolve_device(device_str: str = "auto") -> "torch.device":
    """Resolve device string to torch.device."""
    import torch
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Union[str, Path]) -> DictConfig:
    """
    Load a YAML config, optionally merging a 'defaults' key for base config
    inheritance (simple one-level resolution).
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    defaults = raw.pop("defaults", [])
    cfg = OmegaConf.create(raw)

    for d in defaults:
        base_path = path.parent / f"{d}.yaml"
        if base_path.exists():
            with open(base_path) as f:
                base_raw = yaml.safe_load(f)
            base_raw.pop("defaults", None)
            base_cfg = OmegaConf.create(base_raw)
            cfg = OmegaConf.merge(base_cfg, cfg)

    return cfg


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it does not exist; return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_root() -> Path:
    """Return the repository root (parent of src/)."""
    return Path(__file__).resolve().parent.parent
