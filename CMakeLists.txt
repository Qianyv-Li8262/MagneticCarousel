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
# scan_initial_cpu.cpp is included by 2dmodel-hysteresis.cpp because the model
# header defines non-inline evolution functions. List it as a source for build
# tracking while keeping a single translation unit for those definitions.
set_source_files_properties("${KERNEL_DIR}/scan_initial_cpu.cpp" PROPERTIES HEADER_FILE_ONLY TRUE)
set(IYPT16_FIELD_MODE "exact" CACHE STRING "Magnetic field evaluation: exact or lut")
set_property(CACHE IYPT16_FIELD_MODE PROPERTY STRINGS exact lut)
set(IYPT16_LUT_CACHE_MODE "0" CACHE STRING "LUT cache: 0=off, 1=nodes, 2=spline coefficients")
set_property(CACHE IYPT16_LUT_CACHE_MODE PROPERTY STRINGS 0 1 2)
set(IYPT16_FIELD_LUT_THREADS "24" CACHE STRING "CPU threads used to generate the field LUT")
option(IYPT16_ENABLE_CUDA "Build native GPU scan cubins" ON)
if(NOT IYPT16_FIELD_MODE STREQUAL "exact" AND NOT IYPT16_FIELD_MODE STREQUAL "lut")
    message(FATAL_ERROR "IYPT16_FIELD_MODE must be exact or lut")
endif()

if(IYPT16_ENABLE_CUDA)
    if(CMAKE_VERSION VERSION_LESS 3.27)
        message(FATAL_ERROR "IYPT16_ENABLE_CUDA requires CMake 3.27 or newer for CUDA_CUBIN_COMPILATION")
    endif()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures for GPU scan cubins")
    endif()
    enable_language(CUDA)
endif()
if(NOT IYPT16_LUT_CACHE_MODE MATCHES "^[012]$")
    message(FATAL_ERROR "IYPT16_LUT_CACHE_MODE must be 0, 1, or 2")
endif()
if(NOT IYPT16_FIELD_LUT_THREADS MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "IYPT16_FIELD_LUT_THREADS must be a positive integer")
endif()

add_executable(generate_field_lut
    "${KERNEL_DIR}/tools/generate_field_lut.cpp"
)
target_compile_features(generate_field_lut PRIVATE cxx_std_17)
target_include_directories(generate_field_lut PRIVATE "${KERNEL_DIR}/headers")
target_link_libraries(generate_field_lut PRIVATE Boost::multiprecision)
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Clang's Windows driver finds its bundled libomp, although FindOpenMP
    # currently fails to identify the import library in this toolchain.
    target_compile_options(generate_field_lut PRIVATE -fopenmp)
    target_link_options(generate_field_lut PRIVATE -fopenmp)
else()
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_link_libraries(generate_field_lut PRIVATE OpenMP::OpenMP_CXX)
endif()

if(IYPT16_FIELD_MODE STREQUAL "lut" OR BUILD_TESTING)
    set(FIELD_LUT_DIR "${CMAKE_BINARY_DIR}/field-lut")
    set(FIELD_LUT_FP32 "${FIELD_LUT_DIR}/field_lut_fp32.bin")
    set(FIELD_LUT_FP64 "${FIELD_LUT_DIR}/field_lut_fp64.bin")
    add_custom_command(
        OUTPUT "${FIELD_LUT_FP32}" "${FIELD_LUT_FP64}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${FIELD_LUT_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E env
                "OMP_NUM_THREADS=${IYPT16_FIELD_LUT_THREADS}"
                "OMP_DYNAMIC=FALSE" "OMP_PLACES=cores" "OMP_PROC_BIND=spread"
                "$<TARGET_FILE:generate_field_lut>" "${FIELD_LUT_FP32}" "${FIELD_LUT_FP64}"
        DEPENDS generate_field_lut
                "${KERNEL_DIR}/tools/generate_field_lut.cpp"
                "${KERNEL_DIR}/headers/hysteresis_config_field.hpp"
        VERBATIM
    )
    add_custom_target(generate_field_lut_files DEPENDS "${FIELD_LUT_FP32}" "${FIELD_LUT_FP64}")
endif()

if(IYPT16_ENABLE_CUDA)
    set(CUDA_SCAN_DIR "${CMAKE_BINARY_DIR}/cuda-scans")
    if(IYPT16_FIELD_MODE STREQUAL "lut")
        set(cuda_field_mode 1)
        set(cuda_cache_mode "${IYPT16_LUT_CACHE_MODE}")
    else()
        set(cuda_field_mode 0)
        set(cuda_cache_mode 0)
    endif()

    foreach(scan IN ITEMS sr_somega initial)
        foreach(precision IN ITEMS fp32_kahan fp64)
            set(scan_target "cuda_scan_${scan}_${precision}")
            set(object_target "${scan_target}_objects")
            set(cubin_path "${CUDA_SCAN_DIR}/${scan_target}.cubin")
            set(manifest_path "${CUDA_SCAN_DIR}/${scan_target}.json")
            add_library(${object_target} OBJECT "${KERNEL_DIR}/scan_${scan}.cu")
            set_target_properties(${object_target} PROPERTIES
                CUDA_CUBIN_COMPILATION ON
                CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
                CUDA_STANDARD 17
                CUDA_STANDARD_REQUIRED ON
                EXCLUDE_FROM_ALL TRUE
            )
            target_include_directories(${object_target} PRIVATE
                "${KERNEL_DIR}" "${KERNEL_DIR}/headers")
            target_compile_definitions(${object_target} PRIVATE
                FIELD_EVAL_MODE=${cuda_field_mode}
                LUT_CACHE_MODE=${cuda_cache_mode})
            if(precision STREQUAL "fp64")
                target_compile_definitions(${object_target} PRIVATE USE_FP64)
            else()
                target_compile_definitions(${object_target} PRIVATE USE_KAHAN)
            endif()
            if(scan STREQUAL "sr_somega")
                target_compile_options(${object_target} PRIVATE -lineinfo)
            endif()

            set(stage_dependencies
                "$<TARGET_OBJECTS:${object_target}>"
                "${CMAKE_CURRENT_SOURCE_DIR}/write_cuda_build_config.py")
            set(stage_args)
            if(IYPT16_FIELD_MODE STREQUAL "lut")
                list(APPEND stage_dependencies generate_field_lut_files "${FIELD_LUT_FP32}" "${FIELD_LUT_FP64}")
                if(precision STREQUAL "fp64")
                    list(APPEND stage_args --lut "${FIELD_LUT_FP64}")
                else()
                    list(APPEND stage_args --lut "${FIELD_LUT_FP32}")
                endif()
            endif()
            add_custom_command(
                OUTPUT "${cubin_path}" "${manifest_path}"
                COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/write_cuda_build_config.py"
                    --binary "$<TARGET_OBJECTS:${object_target}>"
                    --output "${cubin_path}"
                    --source-dir "${CMAKE_CURRENT_SOURCE_DIR}"
                    --scan "${scan}"
                    --precision "${precision}"
                    --field-mode "${IYPT16_FIELD_MODE}"
                    --lut-cache-mode "${cuda_cache_mode}"
                    --architectures "${CMAKE_CUDA_ARCHITECTURES}"
                    --cuda-version "${CMAKE_CUDA_COMPILER_VERSION}"
                    ${stage_args}
                DEPENDS ${object_target} ${stage_dependencies}
                VERBATIM
            )
            add_custom_target(${scan_target} ALL DEPENDS "${cubin_path}" "${manifest_path}")
        endforeach()
    endforeach()
endif()

pybind11_add_module(hysteresis_fp64 MODULE NO_EXTRAS
    "${KERNEL_DIR}/2dmodel-hysteresis.cpp"
    "${KERNEL_DIR}/scan_initial_cpu.cpp"
    "${KERNEL_DIR}/bindings/pybind.cpp"
)
target_compile_features(hysteresis_fp64 PRIVATE cxx_std_17)
target_compile_definitions(hysteresis_fp64 PRIVATE USE_FP64 MODULE_NAME=hysteresis)
if(IYPT16_FIELD_MODE STREQUAL "lut")
    target_compile_definitions(hysteresis_fp64 PRIVATE FIELD_EVAL_MODE=1)
    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(hysteresis_fp64 PRIVATE /ULUT_CACHE_MODE /DLUT_CACHE_MODE=${IYPT16_LUT_CACHE_MODE})
    else()
        target_compile_options(hysteresis_fp64 PRIVATE -ULUT_CACHE_MODE -DLUT_CACHE_MODE=${IYPT16_LUT_CACHE_MODE})
    endif()
    target_compile_definitions(hysteresis_fp64 PRIVATE IYPT16_FP64_LUT_PATH="${FIELD_LUT_FP64}")
    add_dependencies(hysteresis_fp64 generate_field_lut_files)
endif()
target_include_directories(hysteresis_fp64 PRIVATE "${KERNEL_DIR}/headers")
set_target_properties(hysteresis_fp64 PROPERTIES
    OUTPUT_NAME hysteresis
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/extensions/fp64"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/extensions/fp64"
)

# Kahan-compensated FP32 build used by animate_one_trajectory.py --kahan.
# Keep the same import name as the FP64 module; the isolated output directory
# lets the Python loader choose the requested precision explicitly.
pybind11_add_module(hysteresis_fp32_kahan MODULE NO_EXTRAS
    "${KERNEL_DIR}/2dmodel-hysteresis.cpp"
    "${KERNEL_DIR}/scan_initial_cpu.cpp"
    "${KERNEL_DIR}/bindings/pybind.cpp"
)
target_compile_features(hysteresis_fp32_kahan PRIVATE cxx_std_17)
target_compile_definitions(hysteresis_fp32_kahan PRIVATE USE_KAHAN MODULE_NAME=hysteresis)
if(IYPT16_FIELD_MODE STREQUAL "lut")
    target_compile_definitions(hysteresis_fp32_kahan PRIVATE FIELD_EVAL_MODE=1)
    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(hysteresis_fp32_kahan PRIVATE /ULUT_CACHE_MODE /DLUT_CACHE_MODE=${IYPT16_LUT_CACHE_MODE})
    else()
        target_compile_options(hysteresis_fp32_kahan PRIVATE -ULUT_CACHE_MODE -DLUT_CACHE_MODE=${IYPT16_LUT_CACHE_MODE})
    endif()
    target_compile_definitions(hysteresis_fp32_kahan PRIVATE IYPT16_FP64_LUT_PATH="${FIELD_LUT_FP32}")
    add_dependencies(hysteresis_fp32_kahan generate_field_lut_files)
endif()
target_include_directories(hysteresis_fp32_kahan PRIVATE "${KERNEL_DIR}/headers")
set_target_properties(hysteresis_fp32_kahan PROPERTIES
    OUTPUT_NAME hysteresis
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/extensions/fp32-kahan"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/extensions/fp32-kahan"
)

# Record the actual extension built in each precision directory. POST_BUILD
# keeps the manifest tied to the binary rather than to CMake configure time.
foreach(target IN ITEMS hysteresis_fp64 hysteresis_fp32_kahan)
    set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/write_build_config.py")
    if(target STREQUAL "hysteresis_fp64")
        set(manifest_precision "fp64")
    else()
        set(manifest_precision "fp32-kahan")
    endif()
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/write_build_config.py"
                --binary "$<TARGET_FILE:${target}>"
                --source-dir "${CMAKE_CURRENT_SOURCE_DIR}"
                --precision "${manifest_precision}"
                --field-mode "${IYPT16_FIELD_MODE}"
                --lut-cache-mode "${IYPT16_LUT_CACHE_MODE}"
        VERBATIM
    )
endforeach()

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
    add_native_test(test_field_lut_vs_exact)
    target_compile_definitions(test_field_lut_vs_exact PRIVATE USE_FP64 FIELD_EVAL_MODE=1
        IYPT16_TEST_LUT_PATH="${FIELD_LUT_FP64}")
    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(test_field_lut_vs_exact PRIVATE /ULUT_CACHE_MODE /DLUT_CACHE_MODE=2)
    else()
        target_compile_options(test_field_lut_vs_exact PRIVATE -ULUT_CACHE_MODE -DLUT_CACHE_MODE=2)
    endif()
    add_dependencies(test_field_lut_vs_exact generate_field_lut_files)
    add_native_test(test_fwd_lut_domain)
    target_compile_definitions(test_fwd_lut_domain PRIVATE FIELD_EVAL_MODE=1)
    add_native_test(test_rolling_step)
    add_native_test(test_magnetic_force_direction)


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
