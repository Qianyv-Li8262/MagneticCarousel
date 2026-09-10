from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

import numpy
import pybind11


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

output = root / f"hysteresis{suffix}"
command = [
    clang,
    "-std=c++17",
    "-O2",
    "-shared",
    str(root / "pybind.cpp"),
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
