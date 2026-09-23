cmake_minimum_required(VERSION 3.24)
project(iypt16_hysteresis_2d LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Use pybind11 from the same Python environment as the extension.
if(NOT pybind11_DIR)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )
endif()
find_package(pybind11 CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS multiprecision)
include(CTest)

set(KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels")
set(IYPT16_FIELD_MODE "exact" CACHE STRING "Magnetic field evaluation: exact or lut")
set_property(CACHE IYPT16_FIELD_MODE PROPERTY STRINGS exact lut)
if(NOT IYPT16_FIELD_MODE STREQUAL "exact" AND NOT IYPT16_FIELD_MODE STREQUAL "lut")
    message(FATAL_ERROR "IYPT16_FIELD_MODE must be exact or lut")
endif()

add_executable(generate_field_lut
    "${KERNEL_DIR}/tools/generate_field_lut.cpp"
)
target_compile_features(generate_field_lut PRIVATE cxx_std_17)
target_include_directories(generate_field_lut PRIVATE "${KERNEL_DIR}/headers")
target_link_libraries(generate_field_lut PRIVATE Boost::multiprecision)

if(IYPT16_FIELD_MODE STREQUAL "lut")
    set(FIELD_LUT_DIR "${CMAKE_BINARY_DIR}/field-lut")
    set(FIELD_LUT_FP32 "${FIELD_LUT_DIR}/field_lut_fp32.bin")
    set(FIELD_LUT_FP64 "${FIELD_LUT_DIR}/field_lut_fp64.bin")
    add_custom_command(
        OUTPUT "${FIELD_LUT_FP32}" "${FIELD_LUT_FP64}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${FIELD_LUT_DIR}"
        COMMAND "$<TARGET_FILE:generate_field_lut>" "${FIELD_LUT_FP32}" "${FIELD_LUT_FP64}"
        DEPENDS generate_field_lut
                "${KERNEL_DIR}/tools/generate_field_lut.cpp"
                "${KERNEL_DIR}/headers/hysteresis_config_field.hpp"
        VERBATIM
    )
    add_custom_target(generate_field_lut_files DEPENDS "${FIELD_LUT_FP32}" "${FIELD_LUT_FP64}")
endif()

pybind11_add_module(hysteresis_fp64 MODULE NO_EXTRAS
    "${KERNEL_DIR}/2dmodel-hysteresis.cpp"
    "${KERNEL_DIR}/bindings/pybind.cpp"
)
target_compile_features(hysteresis_fp64 PRIVATE cxx_std_17)
target_compile_definitions(hysteresis_fp64 PRIVATE USE_FP64 MODULE_NAME=hysteresis)
if(IYPT16_FIELD_MODE STREQUAL "lut")
    target_compile_definitions(hysteresis_fp64 PRIVATE FIELD_EVAL_MODE=1)
    target_compile_definitions(hysteresis_fp64 PRIVATE IYPT16_FP64_LUT_PATH="${FIELD_LUT_FP64}")
    add_dependencies(hysteresis_fp64 generate_field_lut_files)
endif()
target_include_directories(hysteresis_fp64 PRIVATE "${KERNEL_DIR}/headers")
set_target_properties(hysteresis_fp64 PROPERTIES
    OUTPUT_NAME hysteresis
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/extensions/fp64"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/extensions/fp64"
)

if(BUILD_TESTING)
    function(add_native_test name)
        add_executable(${name} "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/${name}.cpp")
        target_compile_features(${name} PRIVATE cxx_std_17)
        target_include_directories(${name} PRIVATE "${KERNEL_DIR}/headers")
        # These tests use assert; keep the checks active in Release builds.
        if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            target_compile_options(${name} PRIVATE /UNDEBUG)
        else()
            target_compile_options(${name} PRIVATE -UNDEBUG)
        endif()
        add_test(NAME ${name} COMMAND ${name})
    endfunction()

    add_native_test(test_boost_multiprecision)
    target_link_libraries(test_boost_multiprecision PRIVATE Boost::multiprecision)
    add_native_test(test_lut_interpolation)
    add_native_test(test_fourier_field_reconstruction)
    target_compile_definitions(test_fourier_field_reconstruction PRIVATE FIELD_EVAL_MODE=1)
    add_native_test(test_fwd_lut_domain)
    target_compile_definitions(test_fwd_lut_domain PRIVATE FIELD_EVAL_MODE=1)
    add_native_test(test_rolling_step)

    foreach(precision IN ITEMS fp32 fp64)
        set(name test_state_projection_${precision}_kahan)
        add_executable(${name} "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/test_state_projection.cpp")
        target_compile_features(${name} PRIVATE cxx_std_17)
        target_include_directories(${name} PRIVATE "${KERNEL_DIR}/headers")
        target_compile_definitions(${name} PRIVATE USE_KAHAN)
        if(precision STREQUAL "fp64")
            target_compile_definitions(${name} PRIVATE USE_FP64)
        endif()
        if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            target_compile_options(${name} PRIVATE /UNDEBUG)
        else()
            target_compile_options(${name} PRIVATE -UNDEBUG)
        endif()
        add_test(NAME ${name} COMMAND ${name})
    endforeach()
endif()
