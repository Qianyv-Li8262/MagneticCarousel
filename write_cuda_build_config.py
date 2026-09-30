# SPDX-License-Identifier: AGPL-3.0-only
"""Stage a CMake-built CUDA cubin and record the inputs used to build it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from write_build_config import literal, sha256


def source_files(root: Path, scan: str) -> list[Path]:
    kernel = root / "kernels"
    files = [kernel / f"scan_{scan}.cu", *sorted((kernel / "headers").rglob("*.hpp"))]
    if scan == "initial":
        files.append(kernel / "scan_initial_common.hpp")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--scan", choices=("sr_somega", "initial"), required=True)
    parser.add_argument("--precision", choices=("fp32_kahan", "fp64"), required=True)
    parser.add_argument("--field-mode", choices=("exact", "lut"), required=True)
    parser.add_argument("--lut-cache-mode", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--architectures", required=True)
    parser.add_argument("--cuda-version", required=True)
    parser.add_argument("--lut", type=Path)
    args = parser.parse_args()

    root = args.source_dir.resolve()
    field_source = (root / "kernels/headers/hysteresis_config_field.hpp").read_text(encoding="utf-8")
    parameters = {
        key: literal(field_source, key, "int")
        for key in ("magnet_number", "kFieldFourierModeCount", "kSplineNodes")
    }
    if args.field_mode == "lut" and (args.lut is None or not args.lut.is_file()):
        raise FileNotFoundError("LUT mode requires the corresponding generated binary table")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.binary, args.output)
    data = {
        "schema_version": 1,
        "binary": args.output.name,
        "binary_sha256": sha256(args.output),
        "scan": args.scan,
        "precision": args.precision,
        "field_mode": args.field_mode,
        "lut_cache_mode": args.lut_cache_mode,
        "architectures": args.architectures,
        "cuda_version": args.cuda_version,
        "field_parameters": parameters,
        "source_sha256": {
            path.relative_to(root).as_posix(): sha256(path)
            for path in source_files(root, args.scan)
        },
    }
    if args.lut is not None:
        data["lut"] = str(args.lut.resolve())
        data["lut_sha256"] = sha256(args.lut)
    args.output.with_suffix(".json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
