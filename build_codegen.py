# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Qianyv-Li8262

from pathlib import Path
import shutil
import subprocess
from typing import Literal


def _write_if_changed(path: Path, contents: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == contents:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8", newline="\n")
    temporary.replace(path)


CoefficientTarget = Literal["cpu", "nvrtc"]


def generate_field_coefficients(root: Path, clang: str, target: CoefficientTarget) -> Path:
    kernels = root / "kernels"
    headers = kernels / "headers"
    source = kernels / "tools" / "generate_field_coefficients.cpp"
    generated = kernels / "generated"
    tool_build = root / "build" / "tools"
    generated.mkdir(parents=True, exist_ok=True)
    tool_build.mkdir(parents=True, exist_ok=True)

    executable = tool_build / "generate_field_coefficients.exe"
    subprocess.run(
        [clang, "-std=c++17", "-O2", str(source), f"-I{headers}", "-o", str(executable)],
        check=True,
    )

    if target == "cpu":
        output = generated / "field_coefficients_cpu.generated.hpp"
    elif target == "nvrtc":
        output = generated / "field_coefficients_nvrtc.generated.cuh"
    else:
        raise ValueError(f"unknown coefficient target: {target!r}")

    result = subprocess.run(
        [str(executable), f"--{target}"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    _write_if_changed(output, result.stdout)
    return output


def COEFFS_INIT(target: CoefficientTarget) -> None:
    """Regenerate exactly one target's coefficient header before compilation."""
    clang = shutil.which("clang++")
    if clang is None:
        raise RuntimeError("clang++ was not found in PATH")
    generate_field_coefficients(Path(__file__).resolve().parent, clang, target)
