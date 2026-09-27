# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Qianyv-Li8262

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

import numpy
import pybind11

from project_paths import extension_dir
from build_codegen import generate_field_coefficients


parser = argparse.ArgumentParser(description="Build the CPU hysteresis extension")
parser.add_argument("--fp64", action="store_true", help="build with double-precision real_t")
parser.add_argument("--dd", action="store_true", help="build with 106-bit double-double real_t")
parser.add_argument("--kahan", action="store_true", help="build FP32 with Kahan-compensated persistent state")
parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="place the extension in this directory instead of next to the sources",
)
args = parser.parse_args()
if sum((args.fp64, args.dd, args.kahan)) > 1:
    parser.error("--fp64, --dd and --kahan are mutually exclusive")

root = Path(__file__).resolve().parent
kernels = root / "kernels"
headers = kernels / "headers"
clang = shutil.which("clang++")
if clang is None:
    raise RuntimeError("clang++ was not found in PATH")

generate_field_coefficients(root, clang, "cpu")

suffix = sysconfig.get_config_var("EXT_SUFFIX")
if not suffix:
    raise RuntimeError("Python did not report an extension-module suffix")

python_include = Path(sysconfig.get_paths()["include"])
python_lib = Path(sys.prefix) / "libs" / f"python{sys.version_info.major}{sys.version_info.minor}.lib"
if not python_lib.is_file():
    raise RuntimeError(f"Python import library was not found: {python_lib}")

build_kind = "dd" if args.dd else "fp64" if args.fp64 else "fp32-kahan" if args.kahan else "fp32"
output_dir = args.output_dir.resolve() if args.output_dir is not None else extension_dir(build_kind)
output_dir.mkdir(parents=True, exist_ok=True)
module_name = "hysteresis_dd" if args.dd else "hysteresis"
output = output_dir / f"{module_name}{suffix}"
command = [
    clang,
    "-std=c++17",
    "-O2",
    *(["-DUSE_FP64"] if args.fp64 else []),
    *(["-DUSE_KAHAN"] if args.kahan else []),
    *(["-DUSE_DD", f"-DMODULE_NAME={module_name}"] if args.dd else []),
    "-shared",
    str(kernels / "bindings" / "pybind.cpp"),
    str(kernels / "1dmodel-hysteresis.cpp"),
    *([str(kernels / "hysteresis_critical_slow_cpu.cpp")] if args.fp64 or args.dd else []),
    *([str(kernels / "hysteresis_phase_boundary_cpu.cpp")] if args.fp64 else []),
    f"-I{headers}",
    f"-I{pybind11.get_include()}",
    f"-I{numpy.get_include()}",
    f"-I{python_include}",
    str(python_lib),
    "-o",
    str(output),
]
subprocess.run(command, check=True)
print(output)
