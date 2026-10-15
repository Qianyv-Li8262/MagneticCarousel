"""Build horizontal-only audit; preserve physical rolling law across disk RPM."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import sysconfig
import numpy
import pybind11

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'scratch/material-response-20261013-source'
OUT=ROOT/'scratch/horizontal-period-20261015-source'
shutil.copytree(SOURCE/'kernels',OUT/'kernels',dirs_exist_ok=True)
p=OUT/'kernels/headers/hysteresis_config.hpp'
s=p.read_text().replace('kEnableInclinedDisk = true','kEnableInclinedDisk = false')
for key in ['kRollingTransitionOmega','kRollingOmegaEpsilon']:
    s=s.replace('CONFIG_CONST real_t '+key,'inline real_t '+key)
p.write_text(s)
p=OUT/'kernels/2dmodel-hysteresis.cpp'
s=p.read_text()+'''
void configure_rolling_time_scale(real_t ratio) {
    if (!real_isfinite(ratio) || ratio<=0)
        throw std::invalid_argument("invalid rolling time scale");
    kRollingTransitionOmega=REAL_LITERAL(1.5)*ratio;
    kRollingOmegaEpsilon=REAL_LITERAL(0.002)*ratio;
}
'''
p.write_text(s)
p=OUT/'kernels/bindings/pybind.cpp';s=p.read_text()
s=s.replace('namespace py = pybind11;','namespace py = pybind11;\nvoid configure_rolling_time_scale(real_t);')
s=s.replace('    m.def("configure_material"','    m.def("configure_rolling_time_scale",&configure_rolling_time_scale);\n    m.def("configure_material"',1)
p.write_text(s)
lut=ROOT/'scratch/laser-tilt-20261009-source/field-lut/field_lut_fp64.bin'
binary=OUT/('horizontal_period_audit'+sysconfig.get_config_var('EXT_SUFFIX'))
cmd=[shutil.which('clang++'),'-std=c++17','-O3','-shared','-fuse-ld=lld','-DUSE_FP64',
     '-DMODULE_NAME=horizontal_period_audit','-DFIELD_EVAL_MODE=1','-DLUT_CACHE_MODE=0',
     f'-DIYPT16_FP64_LUT_PATH="{lut.as_posix()}"','-D_DLL','-D_MT','-Xclang','--dependent-lib=msvcrt',
     str(OUT/'kernels/2dmodel-hysteresis.cpp'),str(OUT/'kernels/bindings/pybind.cpp'),
     f'-I{OUT/"kernels/headers"}',f'-I{pybind11.get_include()}',f'-I{numpy.get_include()}',
     f'-I{sysconfig.get_paths()["include"]}','-ID:/cpplibs/vcpkg/installed/x64-windows/include',
     str(Path(sys.prefix)/'libs/python310.lib'),'-o',str(binary)]
subprocess.run(cmd,check=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'build_manifest.json').write_text(json.dumps(dict(binary=str(binary),binary_sha256=sha(binary),
    lut=str(lut),lut_sha256=sha(lut),source_sha256={str(p.relative_to(OUT)):sha(p) for p in (OUT/'kernels').rglob('*') if p.is_file()},
    changes='Compile-time tilt disabled. No added normal-spin damping. Runtime scaling of rolling transition and regularization rates for constant SI rolling law across RPM.'),indent=2))
print(binary)
