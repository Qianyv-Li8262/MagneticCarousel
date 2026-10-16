# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Qianyv-Li8262

"""Write a build-time manifest beside a 2D CPU extension."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def literal(source: str, name: str, kind: str) -> str:
    matches = re.findall(
        rf"^\s*CONFIG_CONST\s+{kind}\s+{name}\s*=\s*([^;]+);",
        source, re.MULTILINE,
    )
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} in the C++ header; found {len(matches)}")
    return matches[0].strip()


def field_parameters(source: str) -> dict:
    values = {key: literal(source, key, "int")
              for key in ("magnet_number", "kFieldFourierModeCount", "kSplineNodes")}
    for macro, label in (("IYPT16_FIELD_HEIGHT", "height"),
                         ("IYPT16_LUT_LOWER_BOUND_RATIO", "lut_lower_bound_ratio"),
                         ("IYPT16_LUT_UPPER_BOUND_RATIO", "lut_upper_bound_ratio")):
        match = re.search(rf"^#define {macro}\s+([^\s]+)", source, re.MULTILINE)
        if match is None:
            raise ValueError(f"{macro} is missing from the field header")
        values[label] = match.group(1)
    return values


def inclined_disk_parameters(source: str) -> dict:
    enabled = literal(source, "kEnableInclinedDisk", "bool")
    if enabled not in ("true", "false"):
        raise ValueError("kEnableInclinedDisk must be true or false")
    values = {"enabled": enabled == "true"}
    if values["enabled"]:
        for name, label in (("kTiltAxisAzimuthRad", "axis_azimuth_rad"),
                            ("kTiltAngleRad", "angle_rad")):
            value = literal(source, name, "real_t")
            values[label] = float(value.removeprefix("REAL_LITERAL(").removesuffix(")"))
    return values


def eddy_current_parameters(source: str) -> dict:
    match = re.search(r"^#define IYPT16_ENABLE_EDDY_CURRENTS\s+(0|1|false|true)\s*$", source, re.MULTILINE)
    if match is None:
        raise ValueError("IYPT16_ENABLE_EDDY_CURRENTS must be 0 or 1 in the config header")
    enabled = match.group(1) in ("1", "true")
    modes = 4 * int(literal(source, "kEddyCurrentModeCount", "int"))
    return {"enabled": enabled, "mode_count": modes, "state_size": 15 + (3 * modes if enabled else 0),
            "avx_exp_mode": literal(source, "kAvxExpMode", "AvxExpMode"),
            "avx_exp_order": int(literal(source, "kAvxExpExpansionOrder", "int"))}


def write_manifest(path: Path, data: dict) -> None:
    contents = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if not path.is_file() or path.read_text(encoding="utf-8") != contents:
        path.write_text(contents, encoding="utf-8")
        print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp64", "fp32-kahan"), required=True)
    parser.add_argument("--field-mode", choices=("exact", "lut"), required=True)
    parser.add_argument("--magnet-model", choices=("point", "cylinder"), default="point")
    parser.add_argument("--magnet-radius")
    parser.add_argument("--magnet-thickness")
    parser.add_argument("--lut", type=Path)
    parser.add_argument("--lut-cache-mode", choices=(0, 1, 2), type=int, required=True)
    args = parser.parse_args()

    headers = args.source_dir / "kernels" / "headers"
    field_header = headers / "hysteresis_config_field.hpp"
    model_header = headers / "hysteresis_config.hpp"
    field_source = field_header.read_text(encoding="utf-8")
    model_source = model_header.read_text(encoding="utf-8")
    magnet_number = int(literal(field_source, "magnet_number", "int"))
    if magnet_number < 2 or magnet_number % 2:
        raise ValueError("magnet_number must be an even integer >= 2")
    model_keys = ("Mu_r", "Alpha_JA", "C_JA", "K", "Beta", "Eta", "Mu",
                  "kRollingMuLow", "kRollingMuHigh", "kRollingTransitionOmega",
                  "kRollingOmegaEpsilon", "kRollingLambda",
                  "kStepRefinementRecursionDepth", "kAttitudeNormalizeMode",
                  "kDebugChkEight", "kDebugChkTen", "kFinerStepCheck", "kEnableRollingResistance")
    field_values = field_parameters(field_source)
    model_values = {
        key: literal(model_source, key, "int" if key == "kStepRefinementRecursionDepth"
                     else "bool" if key in ("kDebugChkEight", "kDebugChkTen", "kFinerStepCheck", "kEnableRollingResistance")
                     else "AttitudeNormalizeMode" if key == "kAttitudeNormalizeMode"
                     else "real_t")
        for key in model_keys
    }
    data = {
        "schema_version": 1,
        "binary": args.binary.name,
        "binary_sha256": sha256(args.binary),
        "precision": args.precision,
        "field_mode": args.field_mode,
        "magnet_model": args.magnet_model,
        "magnet_geometry": ({"radius": args.magnet_radius, "thickness": args.magnet_thickness}
                            if args.magnet_model == "cylinder" else None),
        "lut_cache_mode": args.lut_cache_mode,
        "magnet_number": magnet_number,
        "field_parameters": field_values,
        "model_parameters": model_values,
        "inclined_disk": inclined_disk_parameters(model_source),
        "eddy_currents": eddy_current_parameters(model_source),
        "source_sha256": {
            str(path.relative_to(args.source_dir)).replace("\\", "/"): sha256(path)
            for path in (field_header, model_header, headers / "2dmodel-hysteresis.hpp",
                         headers / "hysteresis_types.hpp", headers / "hysteresis_physics.hpp",
                         headers / "eddy_currents.hpp", headers / "state_io.hpp",
                         headers / "numeric_extensions/avx2.hpp",
                         args.source_dir / "kernels/generated/eddy_current_coefficients.generated.hpp",
                         args.source_dir / "kernels/scan_initial_cpu.cpp", args.source_dir / "kernels/2dmodel-hysteresis.cpp",
                         args.source_dir / "kernels/bindings/pybind.cpp")
        },
    }
    if args.field_mode == "lut":
        if args.lut is None or not args.lut.is_file():
            raise FileNotFoundError("LUT mode requires the selected generated table")
        data["lut"] = str(args.lut.resolve())
        data["lut_sha256"] = sha256(args.lut)
    output = args.binary.parent / "build_config.json"
    write_manifest(output, data)


if __name__ == "__main__":
    main()
