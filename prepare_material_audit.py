"""Isolated CPU material parameter diagnostics; preserves production formulas."""
from pathlib import Path
import shutil,re,subprocess,sys,sysconfig,json,hashlib
import numpy,pybind11
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'scratch/laser-tilt-20261009-source'
stationary='--stationary' in sys.argv
OUT=ROOT/('scratch/material-response-20261013-static-source' if stationary else 'scratch/material-response-20261013-source');OUT.mkdir(exist_ok=True)
shutil.copytree(SOURCE/'kernels',OUT/'kernels',dirs_exist_ok=True)
h=OUT/'kernels/headers/hysteresis_config.hpp';s=h.read_text()
for name in ['Mu_r','Alpha_JA','C_JA','K','Xi_JA']:
 s=re.sub(r'CONFIG_CONST real_t '+name+r'\s*=',r'inline real_t '+name+' =',s)
h.write_text(s)
if stationary:
 with h.open('a') as out:out.write('\ninline real_t audit_disk_rate=REAL_LITERAL(1.0);\n')
for p in [OUT/'kernels/headers/2dmodel-hysteresis.hpp',OUT/'kernels/2dmodel-hysteresis.cpp']:
 s=p.read_text();s=s.replace('constexpr real_t','const real_t');p.write_text(s)
if stationary:
 p=OUT/'kernels/headers/2dmodel-hysteresis.hpp';s=p.read_text()
 s=s.replace('in.speed.x + in.pos.y, in.speed.y - in.pos.x','in.speed.x + audit_disk_rate * in.pos.y, in.speed.y - audit_disk_rate * in.pos.x')
 s=s.replace('relative_velocity.y - h.y','relative_velocity.y - audit_disk_rate * h.y').replace('relative_velocity.y + h.x','relative_velocity.y + audit_disk_rate * h.x')
 s=s.replace('out.down_disk_angle = REAL_LITERAL(1.0);','out.down_disk_angle = audit_disk_rate;');p.write_text(s)
p=OUT/'kernels/2dmodel-hysteresis.cpp'
with p.open('a') as out:out.write('''
void configure_material(real_t mur,real_t alpha,real_t c,real_t k) {
    if (!(mur>1 && alpha>0 && c>=0 && c<=1 && k>0) || !real_isfinite(mur+alpha+c+k))
        throw std::invalid_argument("invalid material parameters");
    Mu_r=mur;Alpha_JA=alpha;C_JA=c;K=k;
    Xi_JA=1-1/(1+alpha*(mur-1));
}
std::array<real_t,24> material_diagnostics(const State &s,const SolverParams &p) {
    const Vec3 qw=vec_cast<real_t>(rotate(s.attitude,s.q));
    const auto f=eval_mag_field_and_force(p,s.pos,qw,s.down_disk_angle,nullptr,trajectory_lut());
    const real_t D=(Mu_r-1)/(Mu_r+2),E=(1-C_JA)/((1-Xi_JA)*(Mu_r+2));
    const Vec3 m=3*(D*f.h-E*qw),tau=-p.gamma_m*E*cross(qw,f.h);
    return {f.h.x,f.h.y,f.h.z,m.x,m.y,m.z,f.Force.x,f.Force.y,f.Force.z,
            tau.x,tau.y,tau.z,qw.x,qw.y,qw.z,D,E,Xi_JA,
            f.hGrad3d.component[0],f.hGrad3d.component[1],f.hGrad3d.component[2],
            f.hGrad3d.component[3],f.hGrad3d.component[4],-f.hGrad3d.component[0]-f.hGrad3d.component[3]};
}
''')
p=OUT/'kernels/bindings/pybind.cpp';s=p.read_text();s=s.replace('namespace py = pybind11;','namespace py = pybind11;\nvoid configure_material(real_t,real_t,real_t,real_t);\nstd::array<real_t,24> material_diagnostics(const State&,const SolverParams&);')
if stationary:
 pcpp=OUT/'kernels/2dmodel-hysteresis.cpp'
 with pcpp.open('a') as out:out.write('\nvoid configure_disk_rate(real_t x){audit_disk_rate=x;}\n')
 s=s.replace('namespace py = pybind11;','namespace py = pybind11;\nvoid configure_disk_rate(real_t);')
 s=s.replace('    m.def("configure_laser_tilt"','    m.def("configure_disk_rate",&configure_disk_rate);\n    m.def("configure_laser_tilt"')
s=s.replace('    m.def("configure_laser_tilt"','''    m.def("configure_material",&configure_material);
    m.def("material_diagnostics",[](const py::array_t<real_t,py::array::c_style|py::array::forcecast>& a,const SolverParams& p) {
        const auto d=material_diagnostics(state_from_numpy(a),p);py::array_t<real_t> out(24);
        for (int i=0;i<24;++i) out.mutable_at(i)=d[i];return out;
    });
    m.def("configure_laser_tilt"''');p.write_text(s)
lut=(SOURCE/'field-lut/field_lut_fp64.bin').resolve();binary=OUT/('material_audit'+sysconfig.get_config_var('EXT_SUFFIX'))
cmd=[shutil.which('clang++'),'-std=c++17','-O3','-shared','-fuse-ld=lld','-DUSE_FP64','-DMODULE_NAME=material_audit','-DFIELD_EVAL_MODE=1','-DLUT_CACHE_MODE=0',f'-DIYPT16_FP64_LUT_PATH="{lut.as_posix()}"','-D_DLL','-D_MT','-Xclang','--dependent-lib=msvcrt',str(OUT/'kernels/2dmodel-hysteresis.cpp'),str(OUT/'kernels/bindings/pybind.cpp'),f'-I{OUT/"kernels/headers"}',f'-I{pybind11.get_include()}',f'-I{numpy.get_include()}',f'-I{sysconfig.get_paths()["include"]}','-ID:/cpplibs/vcpkg/installed/x64-windows/include',str(Path(sys.prefix)/'libs/python310.lib'),'-o',str(binary)]
subprocess.run(cmd,check=True)
hashes={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'kernels').rglob('*') if p.is_file()}
(OUT/'audit_build.json').write_text(json.dumps(dict(binary=str(binary),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),lut=str(lut),lut_sha256=hashlib.sha256(lut.read_bytes()).hexdigest(),source_sha256=hashes,changes='Material CONFIG_CONST variables made runtime configurable; dependent local constexpr changed to const; diagnostics only. Magnetic force and JA equations unchanged.'),indent=2))
print(binary)
