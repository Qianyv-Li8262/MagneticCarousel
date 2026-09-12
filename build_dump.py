import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

import numpy
import pybind11


parser = argparse.ArgumentParser(description="Build the CPU-only traced hysteresis extension")
parser.add_argument("--fp64", action="store_true", help="build with double-precision real_t")
parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="place the extension in this directory instead of next to the sources",
)
args = parser.parse_args()

root = Path(__file__).resolve().parent
clang = shutil.which("clang++")
if clang is None:
    raise RuntimeError("clang++ was not found in PATH")

suffix = sysconfig.get_config_var("EXT_SUFFIX")
if not suffix:
    raise RuntimeError("Python did not report an extension-module suffix")

python_include = Path(sysconfig.get_paths()["include"])
python_lib = Path(sys.prefix) / "libs" / f"python{sys.version_info.major}{sys.version_info.minor}.lib"
if not python_lib.is_file():
    raise RuntimeError(f"Python import library was not found: {python_lib}")

output_dir = args.output_dir.resolve() if args.output_dir is not None else root
output_dir.mkdir(parents=True, exist_ok=True)
output = output_dir / f"hysteresis_dump{suffix}"
command = [
    clang,
    "-std=c++17",
    "-O2",
    *(["-DUSE_FP64"] if args.fp64 else []),
    "-shared",
    str(root / "pybind_dump.cpp"),
    str(root / "1dmodel-hysteresis.cpp"),
    f"-I{pybind11.get_include()}",
    f"-I{numpy.get_include()}",
    f"-I{python_include}",
    str(python_lib),
    "-o",
    str(output),
]
subprocess.run(command, check=True)
print(output)
