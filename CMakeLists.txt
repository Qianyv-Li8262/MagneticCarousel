cmake_minimum_required(VERSION 3.24...4.4)

project(
    iypt16_hysteresis
    VERSION 0.1.0
    LANGUAGES CXX
)

find_package(
    Python
    COMPONENTS Interpreter Development.Module
    REQUIRED
)

find_package(pybind11 CONFIG REQUIRED)

set(KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels")
set(HEADER_DIR "${KERNEL_DIR}/headers")
set(GENERATED_DIR "${KERNEL_DIR}/generated")
set(EXTENSION_DIR "${CMAKE_BINARY_DIR}/extensions")

add_executable(
    generate_field_coefficients
    "${KERNEL_DIR}/tools/generate_field_coefficients.cpp"
)

target_compile_features(
    generate_field_coefficients
    PRIVATE cxx_std_17
)

target_include_directories(
    generate_field_coefficients
    PRIVATE "${HEADER_DIR}"
)



set(
    CPU_COEFFICIENTS
    "${GENERATED_DIR}/field_coefficients_cpu.generated.hpp"
)

add_custom_command(
    OUTPUT "${CPU_COEFFICIENTS}"

    COMMAND
        "${CMAKE_COMMAND}"
        "-DGENERATOR=$<TARGET_FILE:generate_field_coefficients>"
        "-DOUTPUT=${CPU_COEFFICIENTS}"
        "-DTARGET=cpu"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunCoefficientGenerator.cmake"

    DEPENDS
        generate_field_coefficients
        "${KERNEL_DIR}/tools/generate_field_coefficients.cpp"
        "${HEADER_DIR}/hysteresis_config.hpp"

    COMMENT "Generating CPU field coefficients"
    VERBATIM
)

add_custom_target(
    generate_cpu_coefficients
    DEPENDS "${CPU_COEFFICIENTS}"
)

set(
    NVRTC_COEFFICIENTS
    "${GENERATED_DIR}/field_coefficients_nvrtc.generated.cuh"
)

add_custom_command(
    OUTPUT "${NVRTC_COEFFICIENTS}"

    COMMAND
        "${CMAKE_COMMAND}"
        "-DGENERATOR=$<TARGET_FILE:generate_field_coefficients>"
        "-DOUTPUT=${NVRTC_COEFFICIENTS}"
        "-DTARGET=nvrtc"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunCoefficientGenerator.cmake"

    DEPENDS
        generate_field_coefficients
        "${KERNEL_DIR}/tools/generate_field_coefficients.cpp"
        "${HEADER_DIR}/hysteresis_config.hpp"

    COMMENT "Generating NVRTC field coefficients"
    VERBATIM
)

add_custom_target(
    generate_nvrtc_coefficients ALL
    DEPENDS "${NVRTC_COEFFICIENTS}"
)

add_custom_target(
    generate_coefficients
    DEPENDS
        generate_cpu_coefficients
        generate_nvrtc_coefficients
)

# ---------------------------------------------------------------------------
# FP32
# ---------------------------------------------------------------------------

pybind11_add_module(
    hysteresis_fp32
    MODULE
    NO_EXTRAS
    "${KERNEL_DIR}/bindings/pybind.cpp"
    "${KERNEL_DIR}/1dmodel-hysteresis.cpp"
)

add_dependencies(
    hysteresis_fp32
    generate_cpu_coefficients
)

target_compile_features(
    hysteresis_fp32
    PRIVATE cxx_std_17
)



target_compile_definitions(
    hysteresis_fp32
    PRIVATE MODULE_NAME=hysteresis
)

target_include_directories(
    hysteresis_fp32
    PRIVATE "${HEADER_DIR}"
)

set_target_properties(
    hysteresis_fp32
    PROPERTIES
        OUTPUT_NAME "hysteresis"
        LIBRARY_OUTPUT_DIRECTORY "${EXTENSION_DIR}/fp32"
        RUNTIME_OUTPUT_DIRECTORY "${EXTENSION_DIR}/fp32"
)

# ---------------------------------------------------------------------------
# Kahan FP32 (CPU reproduction of the default CUDA state arithmetic)
# ---------------------------------------------------------------------------

pybind11_add_module(
    hysteresis_fp32_kahan
    MODULE
    NO_EXTRAS
    "${KERNEL_DIR}/bindings/pybind.cpp"
    "${KERNEL_DIR}/1dmodel-hysteresis.cpp"
)

add_dependencies(
    hysteresis_fp32_kahan
    generate_cpu_coefficients
)

target_compile_features(
    hysteresis_fp32_kahan
    PRIVATE cxx_std_17
)

target_compile_definitions(
    hysteresis_fp32_kahan
    PRIVATE
        USE_KAHAN
        MODULE_NAME=hysteresis
)

target_include_directories(
    hysteresis_fp32_kahan
    PRIVATE "${HEADER_DIR}"
)

set_target_properties(
    hysteresis_fp32_kahan
    PROPERTIES
        OUTPUT_NAME "hysteresis"
        LIBRARY_OUTPUT_DIRECTORY "${EXTENSION_DIR}/fp32-kahan"
        RUNTIME_OUTPUT_DIRECTORY "${EXTENSION_DIR}/fp32-kahan"
)

# ---------------------------------------------------------------------------
# FP64
# ---------------------------------------------------------------------------

pybind11_add_module(
    hysteresis_fp64
    MODULE
    NO_EXTRAS
    "${KERNEL_DIR}/bindings/pybind.cpp"
    "${KERNEL_DIR}/1dmodel-hysteresis.cpp"
    "${KERNEL_DIR}/hysteresis_critical_slow_cpu.cpp"
    "${KERNEL_DIR}/hysteresis_phase_boundary_cpu.cpp"
)

add_dependencies(
    hysteresis_fp64
    generate_cpu_coefficients
)

target_compile_features(
    hysteresis_fp64
    PRIVATE cxx_std_17
)



target_compile_definitions(
    hysteresis_fp64
    PRIVATE
        USE_FP64
        MODULE_NAME=hysteresis
)

target_include_directories(
    hysteresis_fp64
    PRIVATE "${HEADER_DIR}"
)

set_target_properties(
    hysteresis_fp64
    PROPERTIES
        OUTPUT_NAME "hysteresis"
        LIBRARY_OUTPUT_DIRECTORY "${EXTENSION_DIR}/fp64"
        RUNTIME_OUTPUT_DIRECTORY "${EXTENSION_DIR}/fp64"
)

# ---------------------------------------------------------------------------
# Double-double
# ---------------------------------------------------------------------------

pybind11_add_module(
    hysteresis_dd
    MODULE
    NO_EXTRAS
    "${KERNEL_DIR}/bindings/pybind.cpp"
    "${KERNEL_DIR}/1dmodel-hysteresis.cpp"
    "${KERNEL_DIR}/hysteresis_critical_slow_cpu.cpp"
)

add_dependencies(
    hysteresis_dd
    generate_cpu_coefficients
)

target_compile_features(
    hysteresis_dd
    PRIVATE cxx_std_17
)



target_compile_definitions(
    hysteresis_dd
    PRIVATE
        USE_DD
        MODULE_NAME=hysteresis_dd
)

target_include_directories(
    hysteresis_dd
    PRIVATE "${HEADER_DIR}"
)

set_target_properties(
    hysteresis_dd
    PROPERTIES
        OUTPUT_NAME "hysteresis_dd"
        LIBRARY_OUTPUT_DIRECTORY "${EXTENSION_DIR}/dd"
        RUNTIME_OUTPUT_DIRECTORY "${EXTENSION_DIR}/dd"
)

# ---------------------------------------------------------------------------
# Trajectory dump — FP32
# ---------------------------------------------------------------------------

pybind11_add_module(
    hysteresis_dump_fp32
    MODULE
    NO_EXTRAS
    "${KERNEL_DIR}/bindings/pybind_dump.cpp"
    "${KERNEL_DIR}/1dmodel-hysteresis.cpp"
)

add_dependencies(
    hysteresis_dump_fp32
    generate_cpu_coefficients
)

target_compile_features(
    hysteresis_dump_fp32
    PRIVATE cxx_std_17
)


target_include_directories(
    hysteresis_dump_fp32
    PRIVATE "${HEADER_DIR}"
)

set_target_properties(
    hysteresis_dump_fp32
    PROPERTIES
        OUTPUT_NAME "hysteresis_dump"
        LIBRARY_OUTPUT_DIRECTORY "${EXTENSION_DIR}/dump-fp32"
        RUNTIME_OUTPUT_DIRECTORY "${EXTENSION_DIR}/dump-fp32"
)

# ---------------------------------------------------------------------------
# Trajectory dump — FP64
# ---------------------------------------------------------------------------

pybind11_add_module(
    hysteresis_dump_fp64
    MODULE
    NO_EXTRAS
    "${KERNEL_DIR}/bindings/pybind_dump.cpp"
    "${KERNEL_DIR}/1dmodel-hysteresis.cpp"
)

add_dependencies(
    hysteresis_dump_fp64
    generate_cpu_coefficients
)

target_compile_features(
    hysteresis_dump_fp64
    PRIVATE cxx_std_17
)


target_compile_definitions(
    hysteresis_dump_fp64
    PRIVATE USE_FP64
)

target_include_directories(
    hysteresis_dump_fp64
    PRIVATE "${HEADER_DIR}"
)

set_target_properties(
    hysteresis_dump_fp64
    PROPERTIES
        OUTPUT_NAME "hysteresis_dump"
        LIBRARY_OUTPUT_DIRECTORY "${EXTENSION_DIR}/dump-fp64"
        RUNTIME_OUTPUT_DIRECTORY "${EXTENSION_DIR}/dump-fp64"
)
