"""Stable project paths shared by build, simulation, and analysis scripts.

Defaults are anchored to this repository instead of the process working
directory. Large run data can be redirected with environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent


def _configured_dir(variable: str, default: Path) -> Path:
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else default


BUILD_DIR = _configured_dir("IYPT16_BUILD_DIR", PROJECT_ROOT / "build")
RUNS_DIR = _configured_dir("IYPT16_RUNS_DIR", PROJECT_ROOT / "runs")
RESULTS_DIR = _configured_dir("IYPT16_RESULTS_DIR", PROJECT_ROOT / "results")
SCRATCH_DIR = _configured_dir("IYPT16_SCRATCH_DIR", PROJECT_ROOT / "scratch")

CRITICAL_SLOWING_RUNS_DIR = RUNS_DIR / "critical-slowing"
SOMEGA_FP64_LOG_DIR = CRITICAL_SLOWING_RUNS_DIR / "logs" / "somega-fp64"
SOMEGA_DD_LOG_DIR = CRITICAL_SLOWING_RUNS_DIR / "logs" / "somega-dd"
SR_FP64_LOG_DIR = CRITICAL_SLOWING_RUNS_DIR / "logs" / "sr-fp64"

# Read-only compatibility locations retained during the staged migration.
LEGACY_PROJECT_TMP = PROJECT_ROOT / "tmp"
LEGACY_WORKSPACE_TMP = WORKSPACE_ROOT / "tmp"


def first_existing(*paths: Path) -> Path:
    """Return the first existing path, or the first candidate if none exist."""
    if not paths:
        raise ValueError("at least one path is required")
    return next((path for path in paths if path.exists()), paths[0])


def extension_dir(kind: str) -> Path:
    """Return the isolated build directory for one extension configuration."""
    supported = {"fp32", "fp64", "dd", "dump-fp32", "dump-fp64"}
    if kind not in supported:
        raise ValueError(f"unknown extension kind {kind!r}; expected one of {sorted(supported)}")
    return BUILD_DIR / "extensions" / kind


def fp64_module_dir() -> Path:
    """Locate an FP64 extension, including legacy locations during migration."""
    return first_existing(
        extension_dir("fp64"),
        LEGACY_PROJECT_TMP / "fp64",
        LEGACY_WORKSPACE_TMP / "fp64",
        PROJECT_ROOT,
    )


def dump_module_dir(fp64: bool = False) -> Path:
    """Locate a trajectory-dump extension, including legacy locations."""
    kind = "dump-fp64" if fp64 else "dump-fp32"
    legacy_name = "dump_fp64" if fp64 else "dump_fp32"
    return first_existing(
        extension_dir(kind),
        BUILD_DIR / "extensions" / "dump-legacy",
        LEGACY_PROJECT_TMP / legacy_name,
        LEGACY_WORKSPACE_TMP / legacy_name,
        PROJECT_ROOT,
    )
