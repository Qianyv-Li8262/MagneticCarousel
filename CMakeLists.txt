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

find_package(
    Boost 1.91
    CONFIG
    REQUIRED
    COMPONENTS multiprecision
)

include(CTest)

set(KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels")
set(HEADER_DIR "${KERNEL_DIR}/headers")
set(GENERATED_DIR "${KERNEL_DIR}/generated")
set(EXTENSION_DIR "${CMAKE_BINARY_DIR}/extensions")

add_executable(
    generate_field_lut
    "${KERNEL_DIR}/tools/generate_field_lut.cpp"
)

target_compile_features(
    generate_field_lut
    PRIVATE cxx_std_17
)

target_include_directories(
    generate_field_lut
    PRIVATE "${HEADER_DIR}"
)

target_link_libraries(
    generate_field_lut
    PRIVATE Boost::multiprecision
)

set(FIELD_LUT_DIR "${CMAKE_BINARY_DIR}/field-lut")
set(FIELD_LUT_FP32 "${FIELD_LUT_DIR}/field_lut_fp32.bin")
set(FIELD_LUT_FP64 "${FIELD_LUT_DIR}/field_lut_fp64.bin")

add_custom_command(
    OUTPUT
        "${FIELD_LUT_FP32}"
        "${FIELD_LUT_FP64}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${FIELD_LUT_DIR}"
    COMMAND
        "$<TARGET_FILE:generate_field_lut>"
        "${FIELD_LUT_FP32}"
        "${FIELD_LUT_FP64}"
    DEPENDS
        generate_field_lut
        "${KERNEL_DIR}/tools/generate_field_lut.cpp"
        "${HEADER_DIR}/hysteresis_config_field.hpp"
    COMMENT "Generating FP32 and FP64 magnetic field spline LUTs"
    VERBATIM
)

add_custom_target(
    generate_field_lut_files
    ALL
    DEPENDS
        "${FIELD_LUT_FP32}"
        "${FIELD_LUT_FP64}"
)

if(BUILD_TESTING)
    add_executable(
        test_boost_multiprecision
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/test_boost_multiprecision.cpp"
    )

    target_compile_features(
        test_boost_multiprecision
        PRIVATE cxx_std_17
    )

    target_link_libraries(
        test_boost_multiprecision
        PRIVATE Boost::multiprecision
    )

    add_test(
        NAME boost_multiprecision
        COMMAND test_boost_multiprecision
    )

    add_executable(
        test_lut_interpolation
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/test_lut_interpolation.cpp"
    )

    target_compile_features(test_lut_interpolation PRIVATE cxx_std_17)
    target_include_directories(test_lut_interpolation PRIVATE "${HEADER_DIR}")

    add_test(
        NAME lut_interpolation
        COMMAND test_lut_interpolation
    )

    add_executable(
        test_fourier_field_reconstruction
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/test_fourier_field_reconstruction.cpp"
    )

    target_compile_features(test_fourier_field_reconstruction PRIVATE cxx_std_17)
    target_compile_definitions(test_fourier_field_reconstruction PRIVATE FIELD_EVAL_MODE=1)
    target_include_directories(test_fourier_field_reconstruction PRIVATE "${HEADER_DIR}")

    add_test(
        NAME fourier_field_reconstruction
        COMMAND test_fourier_field_reconstruction
    )
    foreach(precision IN ITEMS fp32 fp64)
        set(projection_test test_state_projection_${precision}_kahan)
        add_executable(${projection_test} "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/test_state_projection.cpp")
        target_compile_features(${projection_test} PRIVATE cxx_std_17)
        target_include_directories(${projection_test} PRIVATE "${HEADER_DIR}")
        target_compile_definitions(${projection_test} PRIVATE USE_KAHAN)
        if(precision STREQUAL "fp64")
            target_compile_definitions(${projection_test} PRIVATE USE_FP64)
        endif()
        add_test(NAME state_projection_${precision}_kahan COMMAND ${projection_test})
    endforeach()
endif()

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
