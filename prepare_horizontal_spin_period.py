"""Isolated horizontal JA/rolling model with independent passive Z damping."""
from pathlib import Path
import hashlib, json, shutil, subprocess, sys, sysconfig
import numpy, pybind11
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'scratch/horizontal-period-20261015-source'
OUT=ROOT/'scratch/horizontal-spin-period-20261015-source'
shutil.copytree(SOURCE/'kernels',OUT/'kernels',dirs_exist_ok=True)
p=OUT/'kernels/headers/hysteresis_config.hpp'
s=p.read_text();assert 'kEnableInclinedDisk = false' in s
p.write_text(s+'\ninline real_t kNormalSpinLambda = REAL_LITERAL(0.0);\n')
p=OUT/'kernels/headers/2dmodel-hysteresis.hpp';s=p.read_text()
start=s.index('DEVICE INLINE Vec3 add_gaussian_rolling_resistance(')
end=s.index('// MARK: RHS eval',start)
part=s[start:end]
part=part.replace('{\n    if constexpr','{\n    const Vec3 damped_magnetic = magnetic - make_Vec3(REAL_LITERAL(0.0), REAL_LITERAL(0.0), kNormalSpinLambda * static_cast<real_t>(state.omega.z));\n    if constexpr',1)
assert part.count('return magnetic')==3
part=part.replace('return magnetic','return damped_magnetic')
s=s[:start]+part+s[end:];p.write_text(s)
p=OUT/'kernels/2dmodel-hysteresis.cpp'
p.write_text(p.read_text()+'''
void configure_normal_spin_damping(real_t lambda) {
    if (!real_isfinite(lambda) || lambda<0)
        throw std::invalid_argument("normal spin damping must be finite and nonnegative");
    kNormalSpinLambda=lambda;
}
''')
p=OUT/'kernels/bindings/pybind.cpp';s=p.read_text()
s=s.replace('namespace py = pybind11;','namespace py = pybind11;\nvoid configure_normal_spin_damping(real_t);')
s=s.replace('    m.def("configure_material"','    m.def("configure_normal_spin_damping",&configure_normal_spin_damping);\n    m.def("configure_material"',1)
p.write_text(s)
lut=ROOT/'scratch/laser-tilt-20261009-source/field-lut/field_lut_fp64.bin'
binary=OUT/('horizontal_spin_audit'+sysconfig.get_config_var('EXT_SUFFIX'))
cmd=[shutil.which('clang++'),'-std=c++17','-O3','-shared','-fuse-ld=lld','-DUSE_FP64',
     '-DMODULE_NAME=horizontal_spin_audit','-DFIELD_EVAL_MODE=1','-DLUT_CACHE_MODE=0',
     f'-DIYPT16_FP64_LUT_PATH="{lut.as_posix()}"','-D_DLL','-D_MT','-Xclang','--dependent-lib=msvcrt',
     str(OUT/'kernels/2dmodel-hysteresis.cpp'),str(OUT/'kernels/bindings/pybind.cpp'),
     f'-I{OUT/"kernels/headers"}',f'-I{pybind11.get_include()}',f'-I{numpy.get_include()}',
     f'-I{sysconfig.get_paths()["include"]}','-ID:/cpplibs/vcpkg/installed/x64-windows/include',
     str(Path(sys.prefix)/'libs/python310.lib'),'-o',str(binary)]
subprocess.run(cmd,check=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'build_manifest.json').write_text(json.dumps(dict(binary=str(binary),binary_sha256=sha(binary),
    lut=str(lut),lut_sha256=sha(lut),source_sha256={str(p.relative_to(OUT)):sha(p) for p in (OUT/'kernels').rglob('*') if p.is_file()},
    changes='Horizontal disk, fixed moment/height, passive independent Z torque -c_z omega_z; lambda_z=1/(Omega tau_s). Planar rolling unchanged.'),indent=2))
print(binary)
