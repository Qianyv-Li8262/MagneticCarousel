"""Freeze Mu=0.2 in the independent spin/rolling/material trial model."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,sysconfig
import numpy,pybind11
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'scratch/horizontal-spin-period-20261015-source'
OUT=ROOT/'scratch/friction02-period-20261015-source'
shutil.copytree(SOURCE/'kernels',OUT/'kernels',dirs_exist_ok=True)
p=OUT/'kernels/headers/hysteresis_config.hpp';s=p.read_text()
assert s.count('Mu = REAL_LITERAL(0.02)')==1
s=s.replace('Mu = REAL_LITERAL(0.02)','Mu = REAL_LITERAL(0.2)');p.write_text(s)
p=OUT/'kernels/2dmodel-hysteresis.cpp'
p.write_text(p.read_text()+'\nreal_t fixed_contact_friction(){return Mu;}\n')
p=OUT/'kernels/bindings/pybind.cpp';s=p.read_text()
s=s.replace('namespace py = pybind11;','namespace py = pybind11;\nreal_t fixed_contact_friction();')
s=s.replace('    m.def("configure_material"','    m.attr("contact_friction")=static_cast<double>(fixed_contact_friction());\n    m.def("configure_material"',1)
p.write_text(s)
lut=ROOT/'scratch/laser-tilt-20261009-source/field-lut/field_lut_fp64.bin'
binary=OUT/('friction02_audit'+sysconfig.get_config_var('EXT_SUFFIX'))
cmd=[shutil.which('clang++'),'-std=c++17','-O3','-shared','-fuse-ld=lld','-DUSE_FP64',
     '-DMODULE_NAME=friction02_audit','-DFIELD_EVAL_MODE=1','-DLUT_CACHE_MODE=0',
     f'-DIYPT16_FP64_LUT_PATH="{lut.as_posix()}"','-D_DLL','-D_MT','-Xclang','--dependent-lib=msvcrt',
     str(OUT/'kernels/2dmodel-hysteresis.cpp'),str(OUT/'kernels/bindings/pybind.cpp'),
     f'-I{OUT/"kernels/headers"}',f'-I{pybind11.get_include()}',f'-I{numpy.get_include()}',
     f'-I{sysconfig.get_paths()["include"]}','-ID:/cpplibs/vcpkg/installed/x64-windows/include',
     str(Path(sys.prefix)/'libs/python310.lib'),'-o',str(binary)]
subprocess.run(cmd,check=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'build_manifest.json').write_text(json.dumps(dict(binary=str(binary),binary_sha256=sha(binary),
    contact_friction=.2,lut=str(lut),lut_sha256=sha(lut),
    source_sha256={str(p.relative_to(OUT)):sha(p) for p in (OUT/'kernels').rglob('*') if p.is_file()},
    changes='Mu changed from 0.02 to fixed 0.2. Same coefficient enters kinetic friction and rolling feasibility. Horizontal disk, fixed moment/height; independent passive Z damping retained.'),indent=2))
print(binary)
