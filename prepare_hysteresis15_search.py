"""Build isolated runtime rolling-law trials from the current formal source."""
from pathlib import Path
import hashlib,json,re,shutil,subprocess,sys,sysconfig
import numpy,pybind11
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'scratch/hysteresis15-search-source'
shutil.copytree(ROOT/'kernels',OUT/'kernels',dirs_exist_ok=True)
p=OUT/'kernels/headers/hysteresis_config.hpp';s=p.read_text()
for name in ['kRollingMuLow','kRollingMuHigh','kRollingTransitionOmega','kRollingOmegaEpsilon','kRollingLambda','Mu_r','Alpha_JA','C_JA','K','Xi_JA']:
 s=re.sub(r'^CONFIG_CONST real_t '+name+r'\s*=',r'inline real_t '+name+' =',s,flags=re.M)
s += '\ninline real_t kTrialRollingExponent=REAL_LITERAL(2.0);\n'
p.write_text(s)
p=OUT/'kernels/headers/2dmodel-hysteresis.hpp'
s=p.read_text().replace('std::exp(-static_cast<double>(rate2))','std::exp(-std::pow(static_cast<double>(rate2),static_cast<double>(kTrialRollingExponent)/2.))')
s=s.replace('constexpr real_t','const real_t')
p.write_text(s)
p=OUT/'kernels/2dmodel-hysteresis.cpp'
p.write_text(p.read_text().replace('constexpr real_t','const real_t')+"""
void configure_material(real_t mur,real_t alpha,real_t c,real_t k) {
 if (!(mur>1 && alpha>0 && c>=0 && c<=1 && k>0) || !real_isfinite(mur+alpha+c+k)) throw std::invalid_argument("invalid material");
 Mu_r=mur;Alpha_JA=alpha;C_JA=c;K=k;Xi_JA=1-1/(1+alpha*(mur-1));
}
void configure_trial_rolling(real_t low,real_t high,real_t transition,real_t exponent) {
 if (!(low>=0 && high>=0 && transition>0)) throw std::invalid_argument("invalid rolling law");
 kTrialRollingExponent=exponent;kRollingMuLow=low;kRollingMuHigh=high;kRollingTransitionOmega=transition;kRollingLambda=0;
}
""")
p=OUT/'kernels/bindings/pybind.cpp';s=p.read_text()
s=s.replace('namespace py = pybind11;', 'namespace py = pybind11;\nvoid configure_material(real_t,real_t,real_t,real_t);\nvoid configure_trial_rolling(real_t,real_t,real_t,real_t);')
s=s.replace('PYBIND11_MODULE(MODULE_NAME, m)\n{','PYBIND11_MODULE(MODULE_NAME, m)\n{\n    m.def("configure_material", &configure_material);\n    m.def("configure_trial_rolling", &configure_trial_rolling);')
assert 'm.def("configure_trial_rolling"' in s
p.write_text(s)
lut=ROOT/'build/release/field-lut/field_lut_fp64.bin'
binary=OUT/('hysteresis15_trial'+sysconfig.get_config_var('EXT_SUFFIX'))
cmd=[shutil.which('clang++'),'-std=c++17','-O3','-shared','-fuse-ld=lld','-DUSE_FP64','-DMODULE_NAME=hysteresis15_trial','-DFIELD_EVAL_MODE=1','-DLUT_CACHE_MODE=0',f'-DIYPT16_FP64_LUT_PATH="{lut.as_posix()}"','-D_DLL','-D_MT','-Xclang','--dependent-lib=msvcrt',str(OUT/'kernels/2dmodel-hysteresis.cpp'),str(OUT/'kernels/bindings/pybind.cpp'),f'-I{OUT/"kernels/headers"}',f'-I{pybind11.get_include()}',f'-I{numpy.get_include()}',f'-I{sysconfig.get_paths()["include"]}','-ID:/cpplibs/vcpkg/installed/x64-windows/include',str(Path(sys.prefix)/'libs/python310.lib'),'-o',str(binary)]
subprocess.run(cmd,check=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'manifest.json').write_text(json.dumps(dict(binary=str(binary),binary_sha256=sha(binary),lut=str(lut),lut_sha256=sha(lut),formal_build=json.loads((ROOT/'build/release/extensions/fp64/build_config.json').read_text()),source_sha256={str(p.relative_to(OUT)):sha(p) for p in (OUT/'kernels').rglob('*') if p.is_file()},changes='Material Mu_r/Alpha_JA/C_JA/K and rolling low/high/transition/exponent runtime configurable; Xi recomputed. Formal Beta=0.0854985/contact/geometry preserved. No extra normal-spin damping.'),indent=2))
print(binary)

