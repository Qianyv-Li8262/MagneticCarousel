# SPDX-License-Identifier: AGPL-3.0-only
"""Validate and locate CMake-built CUDA scan modules."""

from __future__ import annotations

import json
from pathlib import Path

from project_paths import BUILD_DIR, PROJECT_ROOT
from write_build_config import sha256
from write_cuda_build_config import source_files


def load_cuda_artifact(scan: str, precision: str, field_mode: str, cache_mode: int) -> tuple[Path, dict]:
    name = f"cuda_scan_{scan}_{precision}"
    directory = BUILD_DIR / "cuda-scans"
    manifest_path = directory / f"{name}.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"GPU build missing: {manifest_path}; build CMake target {name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {"schema_version": 1, "scan": scan, "precision": precision,
                "field_mode": field_mode, "lut_cache_mode": cache_mode if field_mode == "lut" else 0}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"GPU build configuration differs from this scan; reconfigure and build {name}")
    binary = directory / f"{name}.cubin"
    if manifest.get("binary") != binary.name or not binary.is_file() or sha256(binary) != manifest.get("binary_sha256"):
        raise RuntimeError(f"GPU cubin is missing or differs from its build manifest: {binary}")
    recorded = manifest.get("source_sha256", {})
    for path in source_files(PROJECT_ROOT, scan):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if not path.is_file() or recorded.get(relative) != sha256(path):
            raise RuntimeError(f"GPU source changed since {name} was built: {relative}; rebuild the CMake target")
    if field_mode == "lut":
        lut = Path(manifest["lut"])
        if not lut.is_file() or sha256(lut) != manifest.get("lut_sha256"):
            raise RuntimeError(f"GPU LUT changed since {name} was built: {lut}; rebuild the CMake target")
    return binary, manifest
