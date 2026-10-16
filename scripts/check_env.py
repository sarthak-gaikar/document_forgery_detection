#!/usr/bin/env python3
"""
scripts/check_env.py
====================
Phase 1 – Environment Verification Script

Checks Python version, package imports, CUDA availability,
disk space, and key directory structure. Prints a rich summary table.
Run: python scripts/check_env.py
"""
from __future__ import annotations

import sys
import os
import platform
import importlib
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Minimal bootstrap: allow running without 'rich' installed yet
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    class _FakeConsole:
        def print(self, *a, **kw):
            print(*a)
        def rule(self, t="", **kw):
            print("-" * 60, t)
    console = _FakeConsole()

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------

def check_python() -> tuple[bool, str]:
    v = sys.version_info
    ok = v >= (3, 10)
    return ok, f"Python {v.major}.{v.minor}.{v.micro}"


def check_package(pkg: str, import_name: str | None = None) -> tuple[bool, str]:
    name = import_name or pkg.replace("-", "_")
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "?")
        return True, ver
    except ImportError:
        return False, "NOT INSTALLED"


def check_cuda() -> tuple[bool, str]:
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            return True, f"{dev} ({mem} GB)"
        return False, "CPU only (CUDA not available) – OK for testing"
    except ImportError:
        return False, "torch not installed"


def check_dir(rel: str) -> tuple[bool, str]:
    p = ROOT / rel
    return p.exists(), str(p)


def check_disk() -> tuple[bool, str]:
    total, used, free = shutil.disk_usage(ROOT)
    free_gb = free / (1024**3)
    ok = free_gb >= 20
    return ok, f"{free_gb:.1f} GB free (need >= 20 GB)"


# ---------------------------------------------------------------------------
# Package manifest to check
# ---------------------------------------------------------------------------
PACKAGES = [
    ("torch", None),
    ("torchvision", None),
    ("segmentation_models_pytorch", None),
    ("timm", None),
    ("transformers", None),
    ("datasets", None),
    ("opencv-python", "cv2"),
    ("albumentations", None),
    ("Pillow", "PIL"),
    ("PyMuPDF", "fitz"),
    ("Faker", "faker"),
    ("reportlab", None),
    ("lmdb", None),
    ("numpy", None),
    ("pandas", None),
    ("scikit-learn", "sklearn"),
    ("scipy", None),
    ("matplotlib", None),
    ("seaborn", None),
    ("tensorboard", None),
    ("omegaconf", None),
    ("einops", None),
    ("rich", None),
    ("fastapi", None),
    ("streamlit", None),
    ("pytest", None),
]

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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    console.rule("[bold cyan]Tampered-Text Detection  –  Environment Check")
    console.print(f"Platform : {platform.system()} {platform.release()}")
    console.print(f"Arch     : {platform.machine()}")
    console.print(f"Root     : {ROOT}\n")

    all_ok = True

    # ---- Python version ---------------------------------------------------
    ok, msg = check_python()
    status = "[green]OK[/green]" if (ok and HAS_RICH) else ("OK" if ok else "FAIL")
    console.print(f"  Python  : {msg}  {'[green]OK[/green]' if ok and HAS_RICH else ('OK' if ok else '[red]FAIL - need 3.10+[/red]')}")
    if not ok:
        all_ok = False

    # ---- CUDA -------------------------------------------------------------
    ok, msg = check_cuda()
    console.print(f"  CUDA    : {msg}")

    # ---- Packages (table) -------------------------------------------------
    if HAS_RICH:
        tbl = Table(title="Package Status", show_header=True, header_style="bold magenta")
        tbl.add_column("Package", style="cyan", width=32)
        tbl.add_column("Version / Status", width=36)
        tbl.add_column("OK?", width=6)
    else:
        print(f"\n{'Package':<32} {'Version':<30} {'OK?':<6}")
        print("-" * 72)

    pkg_fails = []
    for pkg, imp in PACKAGES:
        ok, ver = check_package(pkg, imp)
        if not ok:
            pkg_fails.append(pkg)
            all_ok = False
        if HAS_RICH:
            tbl.add_row(pkg, ver, "[green]YES[/green]" if ok else "[red]NO[/red]")
        else:
            print(f"{pkg:<32} {ver:<30} {'YES' if ok else 'NO':<6}")

    if HAS_RICH:
        console.print(tbl)

    # ---- Directory structure ----------------------------------------------
    if HAS_RICH:
        dtbl = Table(title="Required Directories", show_header=True, header_style="bold blue")
        dtbl.add_column("Path", style="cyan", width=36)
        dtbl.add_column("Exists?", width=8)
    else:
        print(f"\n{'Directory':<36} {'Exists?':<8}")
        print("-" * 46)

    dir_fails = []
    for d in REQUIRED_DIRS:
        ok, full = check_dir(d)
        if not ok:
            dir_fails.append(d)
        if HAS_RICH:
            dtbl.add_row(d, "[green]YES[/green]" if ok else "[red]NO[/red]")
        else:
            print(f"{d:<36} {'YES' if ok else 'NO':<8}")

    if HAS_RICH:
        console.print(dtbl)

    # ---- Disk space -------------------------------------------------------
    ok, msg = check_disk()
    console.print(f"\n  Disk    : {msg}")
    if not ok:
        console.print("  [yellow]WARNING: Low disk space. Dataset + checkpoints need ~50+ GB.[/yellow]")

    # ---- Summary ----------------------------------------------------------
    console.rule()
    if pkg_fails:
        console.print(f"[red]Missing packages: {pkg_fails}[/red]")
        console.print("[yellow]Run:  pip install -r requirements.txt[/yellow]")
    if dir_fails:
        console.print(f"[red]Missing directories: {dir_fails}[/red]")
    if all_ok and not dir_fails:
        console.print("[bold green]All checks passed! Environment is ready.[/bold green]")
    else:
        console.print("[bold red]Some checks failed – see above.[/bold red]")
    console.rule()

    return 0 if (all_ok and not dir_fails) else 1


if __name__ == "__main__":
    sys.exit(main())
