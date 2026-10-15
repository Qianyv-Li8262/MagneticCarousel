"""Isolated diagnostic extension permitting replacement of height-specific LUT."""
from pathlib import Path
import shutil,subprocess,sys,sysconfig,json,hashlib
import numpy,pybind11
ROOT=Path(__file__).resolve().parent;SRC=ROOT/'scratch/material-response-20261013-source';OUT=ROOT/'scratch/height-response-20261013-source';OUT.mkdir(exist_ok=True);shutil.copytree(SRC/'kernels',OUT/'kernels',dirs_exist_ok=True)
p=OUT/'kernels/2dmodel-hysteresis.cpp';s=p.read_text();s=s.replace('namespace {','std::vector<LUTNode> audit_height_nodes;\nnamespace {',1)
needle='const LUTNode *trajectory_lut()\n{';assert needle in s;s=s.replace(needle,needle+'\n    if (!audit_height_nodes.empty()) return audit_height_nodes.data();',1)
s+='''
void configure_height_table(const std::string& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    const std::size_t count=kFieldFourierModeCount*kSplineNodes;
    if (!f || f.tellg()!=static_cast<std::streamoff>(count*sizeof(LUTNode)))
        throw std::runtime_error("height LUT size mismatch");
    std::vector<LUTNode> nodes(count);f.seekg(0);f.read(reinterpret_cast<char*>(nodes.data()),count*sizeof(LUTNode));
    if (!f) throw std::runtime_error("height LUT read failed");
    audit_height_nodes=std::move(nodes);
}
''';p.write_text(s)
p=OUT/'kernels/bindings/pybind.cpp';s=p.read_text();s=s.replace('namespace py = pybind11;','namespace py = pybind11;\nvoid configure_height_table(const std::string&);');s=s.replace('    m.def("configure_material"','    m.def("configure_height_table",&configure_height_table);\n    m.def("configure_material"',1);s=s.replace('py::class_<SolverParams>(m, "SolverParams")','py::class_<SolverParams>(m, "SolverParams", py::module_local())');p.write_text(s)
lut=ROOT/'scratch/laser-tilt-20261009-source/field-lut/field_lut_fp64.bin';binary=OUT/('height_audit'+sysconfig.get_config_var('EXT_SUFFIX'))
cmd=[shutil.which('clang++'),'-std=c++17','-O3','-shared','-fuse-ld=lld','-DUSE_FP64','-DMODULE_NAME=height_audit','-DFIELD_EVAL_MODE=1','-DLUT_CACHE_MODE=0',f'-DIYPT16_FP64_LUT_PATH="{lut.as_posix()}"','-D_DLL','-D_MT','-Xclang','--dependent-lib=msvcrt',str(OUT/'kernels/2dmodel-hysteresis.cpp'),str(OUT/'kernels/bindings/pybind.cpp'),f'-I{OUT/"kernels/headers"}',f'-I{pybind11.get_include()}',f'-I{numpy.get_include()}',f'-I{sysconfig.get_paths()["include"]}','-ID:/cpplibs/vcpkg/installed/x64-windows/include',str(Path(sys.prefix)/'libs/python310.lib'),'-o',str(binary)]
subprocess.run(cmd,check=True);(OUT/'build_info.json').write_text(json.dumps(dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),changes='Runtime replacement of the complete height-specific field/gradient LUT; mechanical and JA equations unchanged. Height is encoded in LUT, not an independent force gain.'),indent=2));print(binary)
