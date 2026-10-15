"""Isolate the current fixed-Q235 core for sequential laser-tilt comparisons."""
from pathlib import Path
import shutil,json,re
root=Path(__file__).resolve().parent
dest=root/'scratch'/'laser-tilt-20261009-source'
dest.mkdir(parents=True,exist_ok=True)
shutil.copytree(root/'kernels',dest/'kernels',dirs_exist_ok=True)
field_header=dest/'kernels/headers/hysteresis_config_field.hpp'
field=field_header.read_text(encoding='utf-8').replace('#define IYPT16_LUT_LOWER_BOUND_RATIO 0.5','#define IYPT16_LUT_LOWER_BOUND_RATIO 0.2').replace('#define IYPT16_LUT_UPPER_BOUND_RATIO 1.5','#define IYPT16_LUT_UPPER_BOUND_RATIO 2.5').replace('kSplineNodes = 204','kSplineNodes = 600')
field=re.sub(r'static_assert\(kFieldFourierModeCount \* kSplineNodes \* 4 \* sizeof\(double\) <= 32 \* 1024,\s*"FP64 field LUT exceeds the 32 KiB shared-memory budget"\);','// CPU-only audit: no CUDA shared-memory limit applies.',field)
field_header.write_text(field,encoding='utf-8')
header=dest/'kernels/headers/hysteresis_config.hpp'
s=header.read_text(encoding='utf-8').replace('kEnableInclinedDisk = false','kEnableInclinedDisk = true').replace('kEnableRollingResistance = false','kEnableRollingResistance = true')
for name in ['Beta','kTiltAngleRad','kTiltAxisAzimuthRad','kRollingMuLow','kRollingMuHigh','kRollingLambda']:
    s=s.replace('CONFIG_CONST real_t '+name,'inline real_t '+name)
header.write_text(s,encoding='utf-8')
cpp=dest/'kernels/2dmodel-hysteresis.cpp'
s=cpp.read_text(encoding='utf-8')
s+='''
void configure_magnet_moment_scale(real_t scale) {
    if (!real_isfinite(scale) || scale<=0)
        throw std::invalid_argument("magnet moment scale must be finite and positive");
    Beta=REAL_LITERAL(0.056999326368103)*scale;
}
void configure_laser_tilt(real_t angle, real_t azimuth) {
    if (!real_isfinite(angle+azimuth) || real_fabs(angle)>REAL_LITERAL(0.02))
        throw std::invalid_argument("invalid laser tilt");
    kTiltAngleRad=angle; kTiltAxisAzimuthRad=azimuth;
}
void configure_rolling_resistance(real_t low, real_t high, real_t lambda) {
    if (!real_isfinite(low+high+lambda) || low<0 || high<0 || lambda<0)
        throw std::invalid_argument("invalid rolling resistance");
    kRollingMuLow=low; kRollingMuHigh=high; kRollingLambda=lambda;
}
Vec3 laser_magnetic_force(const State &state, const SolverParams &params) {
    const Vec3 q_world=vec_cast<real_t>(rotate(state.attitude,state.q));
    return eval_mag_field_and_force(params,state.pos,q_world,state.down_disk_angle,
                                   nullptr,trajectory_lut()).Force;
}
'''
cpp.write_text(s,encoding='utf-8')
binding=dest/'kernels/bindings/pybind.cpp'
s=binding.read_text(encoding='utf-8').replace('namespace py = pybind11;','namespace py = pybind11;\nvoid configure_magnet_moment_scale(real_t);\nvoid configure_laser_tilt(real_t,real_t);\nvoid configure_rolling_resistance(real_t,real_t,real_t);\nVec3 laser_magnetic_force(const State&,const SolverParams&);')
s=s.replace('    m.doc() = ','''    m.def("configure_laser_tilt", &configure_laser_tilt);
    m.def("configure_magnet_moment_scale", &configure_magnet_moment_scale);
    m.def("configure_rolling_resistance", &configure_rolling_resistance);
    m.def("laser_magnetic_force", [](const py::array_t<real_t,py::array::c_style|py::array::forcecast>& a,const SolverParams& p) {
        const auto f=laser_magnetic_force(state_from_numpy(a),p);
        py::array_t<real_t> result(3);result.mutable_at(0)=f.x;result.mutable_at(1)=f.y;result.mutable_at(2)=f.z;return result;
    });
    m.doc() = ''',1)
binding.write_text(s,encoding='utf-8')
lut=(dest/'field-lut/field_lut_fp64.bin').as_posix()
(dest/'field-lut').mkdir(exist_ok=True)
(dest/'CMakeLists.txt').write_text(f'''cmake_minimum_required(VERSION 3.24)
project(laser_tilt_audit LANGUAGES CXX)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${{Python_EXECUTABLE}}" -m pybind11 --cmakedir OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_package(pybind11 CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS multiprecision)
add_executable(laser_lut_generator kernels/tools/generate_cylinder_field_lut.cpp)
target_compile_features(laser_lut_generator PRIVATE cxx_std_17)
target_include_directories(laser_lut_generator PRIVATE kernels/headers)
target_link_libraries(laser_lut_generator PRIVATE Boost::multiprecision)
target_compile_definitions(laser_lut_generator PRIVATE IYPT16_CYLINDER_RADIUS=0.8 IYPT16_CYLINDER_THICKNESS=0.857142857142857)
target_compile_options(laser_lut_generator PRIVATE -fopenmp)
target_link_options(laser_lut_generator PRIVATE -fopenmp)
pybind11_add_module(hysteresis MODULE NO_EXTRAS kernels/2dmodel-hysteresis.cpp kernels/bindings/pybind.cpp)
target_compile_features(hysteresis PRIVATE cxx_std_17)
target_include_directories(hysteresis PRIVATE kernels/headers)
target_link_libraries(hysteresis PRIVATE Boost::multiprecision)
target_compile_definitions(hysteresis PRIVATE USE_FP64 MODULE_NAME=hysteresis FIELD_EVAL_MODE=1 LUT_CACHE_MODE=0 IYPT16_FP64_LUT_PATH="{lut}")
option(LASER_DISABLE_ROLLING "Disable added Gaussian rolling resistance" OFF)
if(LASER_DISABLE_ROLLING)
    target_compile_definitions(hysteresis PRIVATE IYPT16_DISABLE_GAUSSIAN_ROLLING)
endif()
''',encoding='utf-8')
preset=json.loads((root/'CMakeUserPresets.json').read_text(encoding='utf-8'))
preset['configurePresets']=preset['configurePresets'][:1]
preset['configurePresets'][0]['binaryDir']='${sourceDir}/build'
(dest/'CMakeUserPresets.json').write_text(json.dumps(preset,indent=2),encoding='utf-8')
print(dest)
