"""
tests/test_phase1_skeleton.py
Unit tests for Phase 1: project structure, config loading, utility functions.
Run: pytest tests/test_phase1_skeleton.py -v
"""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure src is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from src.utils import set_seed, resolve_device, load_config, ensure_dir, project_root


# ---------------------------------------------------------------------------
# Directory structure tests
# ---------------------------------------------------------------------------
REQUIRED_DIRS = [
    "configs",
    "src",
    "src/data",
    "src/models",
    "src/losses",
    "src/metrics",
    "src/training",
    "src/inference",
    "src/api",
    "tests",
    "outputs/checkpoints",
    "outputs/logs",
    "outputs/results",
    "outputs/figures",
    "scripts",
    "notebooks",
]

@pytest.mark.parametrize("rel_dir", REQUIRED_DIRS)
def test_required_directory_exists(rel_dir: str):
    p = ROOT / rel_dir
    assert p.exists(), f"Missing required directory: {p}"
    assert p.is_dir(), f"Expected a directory, got a file: {p}"


REQUIRED_FILES = [
    "requirements.txt",
    "README.md",
    "pyproject.toml",
    ".gitignore",
    "configs/base.yaml",
    "configs/unet_resnet34.yaml",
    "configs/deeplabv3plus_efficientnet.yaml",
    "configs/segformer_b2.yaml",
    "configs/dual_branch.yaml",
    "configs/classifiers.yaml",
    "configs/synthetic_gen.yaml",
    "src/utils.py",
    "scripts/check_env.py",
]

@pytest.mark.parametrize("rel_file", REQUIRED_FILES)
def test_required_file_exists(rel_file: str):
    p = ROOT / rel_file
    assert p.exists(), f"Missing required file: {p}"
    assert p.is_file(), f"Expected a file: {p}"


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

def test_set_seed_does_not_raise():
    set_seed(42)


def test_resolve_device_cpu():
    import torch
    dev = resolve_device("cpu")
    assert dev == torch.device("cpu")


def test_resolve_device_auto():
    import torch
    dev = resolve_device("auto")
    assert dev in (torch.device("cpu"), torch.device("cuda"))


def test_project_root():
    root = project_root()
    assert root.is_dir()
    assert (root / "src").is_dir()


def test_ensure_dir_creates_path(tmp_path: Path):
    new_dir = tmp_path / "a" / "b" / "c"
    result = ensure_dir(new_dir)
    assert result.exists()
    assert result.is_dir()


def test_load_base_config():
    cfg = load_config(ROOT / "configs/base.yaml")
    assert "project" in cfg
    assert "training" in cfg
    assert "paths" in cfg
    assert cfg.training.seed == 42


def test_load_unet_config_inherits_base():
    cfg = load_config(ROOT / "configs/unet_resnet34.yaml")
    # Should inherit base keys
    assert "training" in cfg
    assert "paths" in cfg
    # Should have model-specific keys
    assert cfg.model.arch == "unet"
    assert cfg.model.encoder == "resnet34"


def test_load_dual_branch_config():
    cfg = load_config(ROOT / "configs/dual_branch.yaml")
    assert cfg.model.name == "dual_branch_forgery_net"
    assert cfg.model.type == "multitask"


def test_load_synthetic_gen_config():
    cfg = load_config(ROOT / "configs/synthetic_gen.yaml")
    assert cfg.synthetic.n_source_docs >= 10000
    assert cfg.synthetic.target_total >= 50000
    assert "splicing" in cfg.synthetic.tamper_types


def test_disclaimer_in_base_config():
    cfg = load_config(ROOT / "configs/base.yaml")
    disclaimer = str(cfg.project.disclaimer).lower()
    assert "forensic" in disclaimer or "legal" in disclaimer
