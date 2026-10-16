cmake_minimum_required(VERSION 3.24)
project(iypt16_hysteresis_2d LANGUAGES CXX)

# CPU State arithmetic and software exponentials use AVX2; CUDA uses scalar arrays.
if(MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>")
else()
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-mavx2>")
endif()

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
set(EDDY_GENERATED_DIR "${KERNEL_DIR}/generated")
set(EDDY_COEFFICIENT_HEADER "${EDDY_GENERATED_DIR}/eddy_current_coefficients.generated.hpp")
set(EDDY_COEFFICIENT_REPORT "${CMAKE_CURRENT_BINARY_DIR}/generated/eddy_current_coefficients.report.json")
set(EDDY_CODEGEN_STAMP "${CMAKE_CURRENT_BINARY_DIR}/generated/eddy_current_coefficients.stamp")
add_executable(generate_eddy_coefficients_tool "${KERNEL_DIR}/tools/generate_eddy_coefficients.cpp")
target_compile_features(generate_eddy_coefficients_tool PRIVATE cxx_std_17)
target_compile_definitions(generate_eddy_coefficients_tool PRIVATE USE_FP64)
target_include_directories(generate_eddy_coefficients_tool PRIVATE "${KERNEL_DIR}/headers")
target_link_libraries(generate_eddy_coefficients_tool PRIVATE Boost::multiprecision)
add_custom_command(
    OUTPUT "${EDDY_CODEGEN_STAMP}"
    BYPRODUCTS "${EDDY_COEFFICIENT_HEADER}" "${EDDY_COEFFICIENT_REPORT}"
    COMMAND "$<TARGET_FILE:generate_eddy_coefficients_tool>" "${EDDY_COEFFICIENT_HEADER}" "${EDDY_COEFFICIENT_REPORT}"
    COMMAND "${CMAKE_COMMAND}" -E touch "${EDDY_CODEGEN_STAMP}"
    DEPENDS generate_eddy_coefficients_tool
        "${KERNEL_DIR}/tools/eddy_codegen_math.hpp" "${KERNEL_DIR}/tools/eddy_codegen_config.hpp"
        "${KERNEL_DIR}/headers/hysteresis_config.hpp" "${KERNEL_DIR}/headers/hysteresis_physics.hpp"
        "${KERNEL_DIR}/headers/hysteresis_types.hpp"
    COMMENT "Generating compressed eddy-current modes with Boost.Multiprecision"
    VERBATIM)
add_custom_target(generate_eddy_coefficients ALL
    DEPENDS "${EDDY_CODEGEN_STAMP}" "${EDDY_COEFFICIENT_HEADER}" "${EDDY_COEFFICIENT_REPORT}")
# scan_initial_cpu.cpp is included by 2dmodel-hysteresis.cpp because the model
# header defines non-inline evolution functions. List it as a source for build
# tracking while keeping a single translation unit for those definitions.
set_source_files_properties("${KERNEL_DIR}/scan_initial_cpu.cpp" PROPERTIES HEADER_FILE_ONLY TRUE)
set(IYPT16_FIELD_MODE "exact" CACHE STRING "Magnetic field evaluation: exact or lut")
set_property(CACHE IYPT16_FIELD_MODE PROPERTY STRINGS exact lut)
set(IYPT16_MAGNET_MODEL "cylinder" CACHE STRING "Magnet source geometry: point or cylinder")
set_property(CACHE IYPT16_MAGNET_MODEL PROPERTY STRINGS point cylinder)
# Diameter 28 mm, full thickness 15 mm; k = 4 / 70 mm.
set(IYPT16_MAGNET_RADIUS "0.8" CACHE STRING "Finite magnet radius, normalized by k")
set(IYPT16_MAGNET_THICKNESS "0.857142857142857" CACHE STRING "Finite magnet thickness, normalized by k")
if(NOT IYPT16_MAGNET_MODEL MATCHES "^(point|cylinder)$")
    message(FATAL_ERROR "IYPT16_MAGNET_MODEL must be point or cylinder")
endif()
if(IYPT16_MAGNET_MODEL STREQUAL "cylinder")
    if(IYPT16_MAGNET_RADIUS STREQUAL "" OR IYPT16_MAGNET_THICKNESS STREQUAL "")
        message(FATAL_ERROR "cylinder requires IYPT16_MAGNET_RADIUS and IYPT16_MAGNET_THICKNESS (k-normalized lengths)")
    endif()
    if(NOT IYPT16_FIELD_MODE STREQUAL "lut")
        message(STATUS "Finite cylinders require LUT: overriding IYPT16_FIELD_MODE=${IYPT16_FIELD_MODE} to lut")
        set(IYPT16_FIELD_MODE lut CACHE STRING "Magnetic field evaluation: exact or lut" FORCE)
    endif()
endif()
# Test-only cylinder dimensions when the selected production model is point.
set(cylinder_radius "${IYPT16_MAGNET_RADIUS}")
set(cylinder_thickness "${IYPT16_MAGNET_THICKNESS}")
if(cylinder_radius STREQUAL "")
    set(cylinder_radius 0.5)
endif()
if(cylinder_thickness STREQUAL "")
    set(cylinder_thickness 0.5)
endif()
foreach(dimension IN ITEMS cylinder_radius cylinder_thickness)
    if(NOT "${${dimension}}" MATCHES "^[+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?$")
        message(FATAL_ERROR "${dimension} must be a positive numeric literal")
    endif()
    if(${dimension} LESS_EQUAL 0)
        message(FATAL_ERROR "${dimension} must be positive")
    endif()
endforeach()
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
function(configure_lut_generator target)
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_include_directories(${target} PRIVATE "${KERNEL_DIR}/headers")
    target_link_libraries(${target} PRIVATE Boost::multiprecision)
    if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        # Clang's Windows driver finds its bundled libomp, although FindOpenMP
        # currently fails to identify the import library in this toolchain.
        target_compile_options(${target} PRIVATE -fopenmp)
        target_link_options(${target} PRIVATE -fopenmp)
    else()
        find_package(OpenMP REQUIRED COMPONENTS CXX)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endfunction()
configure_lut_generator(generate_field_lut)
if(IYPT16_MAGNET_MODEL STREQUAL "cylinder" OR BUILD_TESTING)
    add_executable(generate_cylinder_field_lut "${KERNEL_DIR}/tools/generate_cylinder_field_lut.cpp")
    configure_lut_generator(generate_cylinder_field_lut)
    target_compile_definitions(generate_cylinder_field_lut PRIVATE
        IYPT16_CYLINDER_RADIUS=${cylinder_radius}
        IYPT16_CYLINDER_THICKNESS=${cylinder_thickness})
endif()

if(IYPT16_FIELD_MODE STREQUAL "lut" OR BUILD_TESTING)
    set(FIELD_LUT_DIR "${CMAKE_BINARY_DIR}/field-lut")
    set(FIELD_LUT_FP32 "${FIELD_LUT_DIR}/field_lut_fp32.bin")
    set(FIELD_LUT_FP64 "${FIELD_LUT_DIR}/field_lut_fp64.bin")
    foreach(model IN ITEMS point cylinder)
        if(NOT BUILD_TESTING AND NOT IYPT16_MAGNET_MODEL STREQUAL model)
            continue()
        endif()
        if(model STREQUAL "point")
            set(generator_target generate_field_lut)
        else()
            set(generator_target generate_cylinder_field_lut)
        endif()
        set(FIELD_LUT_${model}_FP32 "${FIELD_LUT_DIR}/field_lut_${model}_fp32.bin")
        set(FIELD_LUT_${model}_FP64 "${FIELD_LUT_DIR}/field_lut_${model}_fp64.bin")
        add_custom_command(
            OUTPUT "${FIELD_LUT_${model}_FP32}" "${FIELD_LUT_${model}_FP64}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${FIELD_LUT_DIR}"
            COMMAND "${CMAKE_COMMAND}" -E env
                "OMP_NUM_THREADS=${IYPT16_FIELD_LUT_THREADS}"
                "OMP_DYNAMIC=FALSE" "OMP_PLACES=cores" "OMP_PROC_BIND=spread"
                "$<TARGET_FILE:${generator_target}>" "${FIELD_LUT_${model}_FP32}" "${FIELD_LUT_${model}_FP64}"
            DEPENDS ${generator_target}
                "${KERNEL_DIR}/tools/field_lut_common.hpp"
                "${KERNEL_DIR}/headers/hysteresis_config_field.hpp"
            VERBATIM)
        add_custom_target(generate_field_lut_${model}_files
            DEPENDS "${FIELD_LUT_${model}_FP32}" "${FIELD_LUT_${model}_FP64}")
    endforeach()
    # Keep established runtime paths; only the selected model's data is staged.
    add_custom_command(OUTPUT "${FIELD_LUT_FP32}" "${FIELD_LUT_FP64}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${FIELD_LUT_${IYPT16_MAGNET_MODEL}_FP32}" "${FIELD_LUT_FP32}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${FIELD_LUT_${IYPT16_MAGNET_MODEL}_FP64}" "${FIELD_LUT_FP64}"
        DEPENDS "${FIELD_LUT_${IYPT16_MAGNET_MODEL}_FP32}" "${FIELD_LUT_${IYPT16_MAGNET_MODEL}_FP64}"
        VERBATIM)
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
            add_dependencies(${object_target} generate_eddy_coefficients)
            set_target_properties(${object_target} PROPERTIES
                CUDA_CUBIN_COMPILATION ON
                CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
                CUDA_STANDARD 17
                CUDA_STANDARD_REQUIRED ON
                EXCLUDE_FROM_ALL TRUE
            )
            target_include_directories(${object_target} PRIVATE
                "${KERNEL_DIR}" "${KERNEL_DIR}/headers" "${EDDY_GENERATED_DIR}")
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
                "${CMAKE_CURRENT_SOURCE_DIR}/write_cuda_build_config.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/write_build_config.py")
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
                OUTPUT "${cubin_path}"
                COMMAND "${CMAKE_COMMAND}" -E make_directory "${CUDA_SCAN_DIR}"
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "$<TARGET_OBJECTS:${object_target}>" "${cubin_path}"
                DEPENDS ${object_target} "$<TARGET_OBJECTS:${object_target}>"
                VERBATIM)
            add_custom_command(
                OUTPUT "${manifest_path}"
                COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/write_cuda_build_config.py"
                    --binary "${cubin_path}"
                    --output "${cubin_path}"
                    --source-dir "${CMAKE_CURRENT_SOURCE_DIR}"
                    --scan "${scan}"
                    --precision "${precision}"
                    --field-mode "${IYPT16_FIELD_MODE}"
                    --magnet-model "${IYPT16_MAGNET_MODEL}"
                    --magnet-radius "${cylinder_radius}"
                    --magnet-thickness "${cylinder_thickness}"
                    --lut-cache-mode "${cuda_cache_mode}"
                    --architectures "${CMAKE_CUDA_ARCHITECTURES}"
                    --cuda-version "${CMAKE_CUDA_COMPILER_VERSION}"
                    ${stage_args}
                DEPENDS "${cubin_path}" ${stage_dependencies}
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
foreach(target IN ITEMS hysteresis_fp64 hysteresis_fp32_kahan)
    add_dependencies(${target} generate_eddy_coefficients)
    target_include_directories(${target} PRIVATE "${EDDY_GENERATED_DIR}")
endforeach()
set_target_properties(hysteresis_fp32_kahan PROPERTIES
    OUTPUT_NAME hysteresis
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/extensions/fp32-kahan"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/extensions/fp32-kahan"
)

# Manifests are independent build outputs: editing metadata writers or replacing
# a LUT refreshes JSON without relinking the extension. Binary/header dependencies
# ensure computation changes are built before recording their metadata.
add_custom_target(build_metadata)
foreach(target IN ITEMS hysteresis_fp64 hysteresis_fp32_kahan)
    if(target STREQUAL "hysteresis_fp64")
        set(manifest_precision "fp64")
        set(cpu_manifest_lut "${FIELD_LUT_FP64}")
        set(manifest_directory "${CMAKE_BINARY_DIR}/extensions/fp64")
    else()
        set(manifest_precision "fp32-kahan")
        set(cpu_manifest_lut "${FIELD_LUT_FP32}")
        set(manifest_directory "${CMAKE_BINARY_DIR}/extensions/fp32-kahan")
    endif()
    set(cpu_manifest_args)
    set(cpu_manifest_dependencies)
    if(IYPT16_FIELD_MODE STREQUAL "lut")
        list(APPEND cpu_manifest_args --lut "${cpu_manifest_lut}")
        list(APPEND cpu_manifest_dependencies "${cpu_manifest_lut}")
    endif()
    set(cpu_manifest "${manifest_directory}/build_config.json")
    set(cpu_manifest_command "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/write_build_config.py"
                --binary "$<TARGET_FILE:${target}>"
                --source-dir "${CMAKE_CURRENT_SOURCE_DIR}"
                --precision "${manifest_precision}"
                --field-mode "${IYPT16_FIELD_MODE}"
                --magnet-model "${IYPT16_MAGNET_MODEL}"
                --magnet-radius "${cylinder_radius}"
                --magnet-thickness "${cylinder_thickness}"
                --lut-cache-mode "${IYPT16_LUT_CACHE_MODE}"
                ${cpu_manifest_args})
    # Preserve direct extension-target builds: a new binary also gets a manifest.
    # The independent output below refreshes metadata when no link is needed.
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${cpu_manifest_command} VERBATIM)
    add_custom_command(OUTPUT "${cpu_manifest}"
        COMMAND ${cpu_manifest_command}
        DEPENDS ${target} "$<TARGET_FILE:${target}>" ${cpu_manifest_dependencies}
                "${CMAKE_CURRENT_SOURCE_DIR}/write_build_config.py"
                "${KERNEL_DIR}/headers/hysteresis_config_field.hpp"
                "${KERNEL_DIR}/headers/hysteresis_physics.hpp"
                "${KERNEL_DIR}/headers/hysteresis_config.hpp"
        VERBATIM)
    add_custom_target(${target}_manifest ALL DEPENDS "${cpu_manifest}")
    add_dependencies(build_metadata ${target}_manifest)
endforeach()
if(IYPT16_ENABLE_CUDA)
    foreach(scan IN ITEMS sr_somega initial)
        foreach(precision IN ITEMS fp32_kahan fp64)
            add_dependencies(build_metadata cuda_scan_${scan}_${precision})
        endforeach()
    endforeach()
endif()

if(BUILD_TESTING)
    # Keep an enabled binding isolated from the production switch, so tests can
    # verify that Python chunks preserve all 36 history values while production
    # remains compiled with eddy currents disabled for regression comparisons.
    pybind11_add_module(hysteresis_eddy_enabled_test MODULE NO_EXTRAS
        "${KERNEL_DIR}/2dmodel-hysteresis.cpp" "${KERNEL_DIR}/bindings/pybind.cpp")
    target_compile_features(hysteresis_eddy_enabled_test PRIVATE cxx_std_17)
    target_compile_definitions(hysteresis_eddy_enabled_test PRIVATE USE_FP64
        MODULE_NAME=hysteresis_eddy_enabled_test IYPT16_ENABLE_EDDY_CURRENTS=1)
    target_include_directories(hysteresis_eddy_enabled_test PRIVATE "${KERNEL_DIR}/headers")
    add_dependencies(hysteresis_eddy_enabled_test generate_eddy_coefficients)
    if(IYPT16_FIELD_MODE STREQUAL "lut")
        target_compile_definitions(hysteresis_eddy_enabled_test PRIVATE FIELD_EVAL_MODE=1
            IYPT16_FP64_LUT_PATH="${FIELD_LUT_FP64}")
        add_dependencies(hysteresis_eddy_enabled_test generate_field_lut_files)
    endif()
    set_target_properties(hysteresis_eddy_enabled_test PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/native-tests"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/native-tests")
    add_test(NAME test_eddy_binding_chunks COMMAND "${Python_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_eddy_binding_chunks.py"
        "$<TARGET_FILE:hysteresis_eddy_enabled_test>")
    foreach(enabled IN ITEMS 0 1)
        foreach(precision IN ITEMS fp64 fp32_kahan)
            set(name test_eddy_integration_${enabled}_${precision})
            add_executable(${name} "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/test_eddy_integration.cpp")
            target_compile_features(${name} PRIVATE cxx_std_17)
            target_include_directories(${name} PRIVATE "${KERNEL_DIR}/headers")
            target_compile_definitions(${name} PRIVATE IYPT16_ENABLE_EDDY_CURRENTS=${enabled})
            if(precision STREQUAL "fp64")
                target_compile_definitions(${name} PRIVATE USE_FP64)
            else()
                target_compile_definitions(${name} PRIVATE USE_KAHAN)
            endif()
            add_dependencies(${name} generate_eddy_coefficients)
            add_test(NAME ${name} COMMAND ${name})
        endforeach()
    endforeach()
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
    add_native_test(test_eddy_coefficients)
    target_compile_definitions(test_eddy_coefficients PRIVATE USE_FP64)
    target_include_directories(test_eddy_coefficients PRIVATE "${EDDY_GENERATED_DIR}")
    add_dependencies(test_eddy_coefficients generate_eddy_coefficients)
    if(IYPT16_ENABLE_CUDA)
        add_library(test_eddy_coefficients_cuda OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/test_eddy_coefficients_cuda.cu")
        set_target_properties(test_eddy_coefficients_cuda PROPERTIES CUDA_CUBIN_COMPILATION ON
            CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
        target_compile_definitions(test_eddy_coefficients_cuda PRIVATE USE_FP64)
        target_include_directories(test_eddy_coefficients_cuda PRIVATE "${KERNEL_DIR}/headers" "${EDDY_GENERATED_DIR}")
        add_dependencies(test_eddy_coefficients_cuda generate_eddy_coefficients)
    endif()
    add_native_test(test_lut_interpolation)
    add_native_test(test_solver_params)
    add_executable(test_solver_params_fp64 "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/test_solver_params.cpp")
    target_compile_features(test_solver_params_fp64 PRIVATE cxx_std_17)
    target_include_directories(test_solver_params_fp64 PRIVATE "${KERNEL_DIR}/headers")
    target_compile_definitions(test_solver_params_fp64 PRIVATE USE_FP64)
    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(test_solver_params_fp64 PRIVATE /UNDEBUG)
    else()
        target_compile_options(test_solver_params_fp64 PRIVATE -UNDEBUG)
    endif()
    add_test(NAME test_solver_params_fp64 COMMAND test_solver_params_fp64)
    add_native_test(test_fourier_field_reconstruction)
    target_compile_definitions(test_fourier_field_reconstruction PRIVATE FIELD_EVAL_MODE=1)
    foreach(model IN ITEMS point cylinder)
        if(model STREQUAL IYPT16_MAGNET_MODEL)
            set(field_test test_field_lut_vs_exact)
        else()
            set(field_test test_field_lut_vs_${model})
        endif()
        add_executable(${field_test} "${CMAKE_CURRENT_SOURCE_DIR}/tests/native/test_field_lut_vs_exact.cpp")
        target_compile_features(${field_test} PRIVATE cxx_std_17)
        target_include_directories(${field_test} PRIVATE "${KERNEL_DIR}/headers" "${KERNEL_DIR}/tools")
        target_link_libraries(${field_test} PRIVATE Boost::multiprecision)
        add_test(NAME test_field_lut_vs_${model} COMMAND ${field_test})
        target_compile_definitions(${field_test} PRIVATE USE_FP64 FIELD_EVAL_MODE=1
            IYPT16_TEST_LUT_PATH="${FIELD_LUT_${model}_FP64}")
        if(model STREQUAL "cylinder")
            target_compile_definitions(${field_test} PRIVATE IYPT16_MAGNET_MODEL=1
                IYPT16_CYLINDER_RADIUS=${cylinder_radius} IYPT16_CYLINDER_THICKNESS=${cylinder_thickness})
        else()
            target_compile_definitions(${field_test} PRIVATE IYPT16_MAGNET_MODEL=0)
        endif()
        if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            target_compile_options(${field_test} PRIVATE /UNDEBUG /ULUT_CACHE_MODE /DLUT_CACHE_MODE=2)
        else()
            target_compile_options(${field_test} PRIVATE -UNDEBUG -ULUT_CACHE_MODE -DLUT_CACHE_MODE=2)
        endif()
        add_dependencies(${field_test} generate_field_lut_${model}_files)
    endforeach()
    add_native_test(test_fwd_lut_domain)
    target_compile_definitions(test_fwd_lut_domain PRIVATE FIELD_EVAL_MODE=1)
    add_native_test(test_rolling_step)
    add_native_test(test_inclined_disk)
    add_native_test(test_magnetic_force_direction)
    # Existing analytic tests describe the zero-added-rolling-resistance limit.
    target_compile_definitions(test_rolling_step PRIVATE IYPT16_DISABLE_GAUSSIAN_ROLLING)
    target_compile_definitions(test_inclined_disk PRIVATE IYPT16_DISABLE_GAUSSIAN_ROLLING)
    add_native_test(test_gaussian_rolling)


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
