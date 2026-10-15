"""Compile an isolated fit with the user's measured rolling/contact law frozen."""
from pathlib import Path
import hashlib,json,re,shutil,subprocess,sys,sysconfig
import numpy,pybind11
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'scratch/friction02-period-20261015-source'
OUT=ROOT/'scratch/measured-rolling-period-20261015-source'
shutil.copytree(SOURCE/'kernels',OUT/'kernels',dirs_exist_ok=True)
p=OUT/'kernels/headers/hysteresis_config.hpp';s=p.read_text()
formal=(ROOT/'kernels/headers/hysteresis_config.hpp').read_text()
start,end=formal.index('// Mechanical parameters'),formal.index('// Fixed tilt axis')
block=formal[start:end]
for name in ['kRollingTransitionOmega','kRollingOmegaEpsilon']:
    block=block.replace('CONFIG_CONST real_t '+name,'inline real_t '+name)
s=s[:s.index('// Mechanical parameters')]+block+s[s.index('// Fixed tilt axis'):]
assert 'kEnableInclinedDisk = false' in s
p.write_text(s)
p=OUT/'kernels/2dmodel-hysteresis.cpp';s=p.read_text()
pattern=r'void configure_rolling_resistance\(real_t low, real_t high, real_t lambda\) \{.*?\n\}'
s,n=re.subn(pattern,'''void configure_rolling_resistance(real_t low, real_t high, real_t lambda) {
    if (low!=kRollingMuLow || high!=kRollingMuHigh || lambda!=kRollingLambda)
        throw std::invalid_argument("Measured rolling resistance is fixed; do not tune it");
}''',s,flags=re.S);assert n==1
p.write_text(s)
p=OUT/'kernels/bindings/pybind.cpp';s=p.read_text()
s=s.replace('    m.attr("contact_friction")=', '''    m.attr("measured_rolling_parameters")=py::make_tuple(0.00490,0.00490,1.5,0.002,0.0);
    m.attr("inclined_disk_enabled")=false;
    m.attr("contact_friction")=''',1)
p.write_text(s)
lut=ROOT/'scratch/laser-tilt-20261009-source/field-lut/field_lut_fp64.bin'
binary=OUT/('measured_rolling_audit'+sysconfig.get_config_var('EXT_SUFFIX'))
cmd=[shutil.which('clang++'),'-std=c++17','-O3','-shared','-fuse-ld=lld','-DUSE_FP64',
     '-DMODULE_NAME=measured_rolling_audit','-DFIELD_EVAL_MODE=1','-DLUT_CACHE_MODE=0',
     f'-DIYPT16_FP64_LUT_PATH="{lut.as_posix()}"','-D_DLL','-D_MT','-Xclang','--dependent-lib=msvcrt',
     str(OUT/'kernels/2dmodel-hysteresis.cpp'),str(OUT/'kernels/bindings/pybind.cpp'),
     f'-I{OUT/"kernels/headers"}',f'-I{pybind11.get_include()}',f'-I{numpy.get_include()}',
     f'-I{sysconfig.get_paths()["include"]}','-ID:/cpplibs/vcpkg/installed/x64-windows/include',
     str(Path(sys.prefix)/'libs/python310.lib'),'-o',str(binary)]
subprocess.run(cmd,check=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'build_manifest.json').write_text(json.dumps(dict(binary=str(binary),binary_sha256=sha(binary),
    contact_friction=.2,measured_rolling=dict(enabled=True,low=.00490,high=.00490,transition=1.5,epsilon=.002,linear=0.),
    tilt_enabled=False,lut=str(lut),lut_sha256=sha(lut),
    source_sha256={str(p.relative_to(OUT)):sha(p) for p in (OUT/'kernels').rglob('*') if p.is_file()},
    changes='Measured rolling and Mu=0.2 fixed; horizontal disk. Same fixed magnet moment/height. Only uncalibrated material and existing independent normal-spin damping may be fitted.'),indent=2))
print(binary)
