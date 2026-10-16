"""Rebuild the isolated low-height fitting executable; production build is untouched."""
from pathlib import Path
import os,json,shutil,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'scratch/video56c6-fit-20261016-source'
for folder in ['headers','generated','tools']:
 shutil.copytree(ROOT/'kernels'/folder,OUT/'kernels'/folder,dirs_exist_ok=True)
p=OUT/'kernels/headers/hysteresis_config_field.hpp';p.write_text(p.read_text().replace('#define IYPT16_FIELD_HEIGHT 2.264','#define IYPT16_FIELD_HEIGHT 1.8285714285714286'))
p=OUT/'kernels/headers/hysteresis_config.hpp';s=p.read_text()
for key in ['Mu_r','Alpha_JA','C_JA','K','Xi_JA']:s=s.replace('CONFIG_CONST real_t '+key,'inline real_t '+key)
p.write_text(s)
p=OUT/'kernels/headers/2dmodel-hysteresis.hpp';s=p.read_text()
for key in ['E_','D_','c','a']:s=s.replace('constexpr real_t '+key+' =','const real_t '+key+' =')
p.write_text(s)
p=OUT/'kernels/generated/eddy_current_coefficients.generated.hpp';p.write_text(p.read_text().replace('alignas(32) inline constexpr double','alignas(32) inline double'))
p=OUT/'kernels/tools/generate_eddy_coefficients.cpp';s=p.read_text().replace('if (argc!=3)','if (argc!=4)').replace('const Real mu(Mu_r),sigma','const Real mu(std::stod(argv[3])),sigma');p.write_text(s)
env=os.environ.copy();preset=json.loads((ROOT/'CMakeUserPresets.json').read_text())['configurePresets'][0]
for k,v in preset['environment'].items():env[k]=v.replace('$penv{PATH}',os.environ['PATH'])
compiler=preset['cacheVariables']['CMAKE_CXX_COMPILER'];include=['-I'+str(OUT/'kernels/headers'),'-ID:/cpplibs/vcpkg/installed/x64-windows/include']
def compile(source,name,flags):subprocess.run([compiler,'-std=c++17','-O3','-DUSE_FP64',*include,*flags,str(source),'-o',str(OUT/name)],env=env,check=True)
compile(ROOT/'tests/native/fit_video56c6.cpp','fit_video56c6.exe',['-mavx2','-ffp-contract=off','-DFIELD_EVAL_MODE=1'])
compile(OUT/'kernels/tools/generate_eddy_coefficients.cpp','generate_eddy_fit.exe',[])
compile(OUT/'kernels/tools/generate_cylinder_field_lut.cpp','generate_lut_parallel.exe',['-fopenmp','-DIYPT16_CYLINDER_RADIUS=0.8','-DIYPT16_CYLINDER_THICKNESS=0.857142857142857'])
env['OMP_NUM_THREADS']='8'
subprocess.run([str(OUT/'generate_lut_parallel.exe'),str(OUT/'field_lut_fp32.bin'),str(OUT/'field_lut_fp64.bin')],env=env,check=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=dict(h_mm=32,beta=.0569993,eddy_modes=12,production_untouched=True,source_hashes={str(p.relative_to(OUT)):sha(p) for p in (OUT/'kernels').rglob('*') if p.is_file()},harness_sha256=sha(ROOT/'tests/native/fit_video56c6.cpp'),binary_sha256=sha(OUT/'fit_video56c6.exe'),lut_sha256=sha(OUT/'field_lut_fp64.bin'))
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
