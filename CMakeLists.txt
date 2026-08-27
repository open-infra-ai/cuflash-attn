cmake_minimum_required(VERSION 3.18)

# Set the project default before enabling CUDA. CMake initializes
# CMAKE_CUDA_ARCHITECTURES during project(), so a fallback placed afterwards
# would preserve nvcc's legacy sm_52 default instead of this supported set.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86" CACHE STRING "CUDA architectures")
endif()

project(cuflash VERSION 0.6.0 LANGUAGES CXX CUDA)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)
include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Export compile_commands.json for clangd LSP support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Options
option(BUILD_TESTS "Build tests" ON)
option(ENABLE_RAPIDCHECK "Enable RapidCheck property tests" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(ENABLE_FAST_MATH "Enable --use_fast_math (faster but less precise)" OFF)
option(ENABLE_ASAN "Enable AddressSanitizer (debug builds only)" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer (debug builds only)" OFF)

# Source files - unified FP32/FP16/BF16 templates
set(SOURCES
    src/api/flash_attention_api.cu
    src/forward/flash_attention_forward_typed.cu
    src/forward/flash_attention_forward_wmma.cu
    src/forward/flash_decoding.cu
    src/backward/flash_attention_backward_typed.cu
    src/kernels/tile_io.cu
    src/kernels/matmul.cu
    src/kernels/online_softmax.cu
)

# Library
add_library(cuflash ${SOURCES})
target_link_libraries(cuflash PUBLIC CUDA::cudart)
target_include_directories(cuflash PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CUDAToolkit_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:${CUDAToolkit_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:include>
)
# Private include directories for internal kernels
target_include_directories(cuflash PRIVATE
    ${CMAKE_SOURCE_DIR}/src/kernels
    ${CMAKE_SOURCE_DIR}/src
)
set_target_properties(cuflash PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME cuflash
    CUDA_SEPARABLE_COMPILATION ON
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Define CUFLASH_BUILDING_SHARED when building as shared library
if(BUILD_SHARED_LIBS)
    target_compile_definitions(cuflash PRIVATE CUFLASH_BUILDING_SHARED=1)
endif()

# Generate version header
configure_file(
    ${CMAKE_SOURCE_DIR}/include/cuflash/version.h.in
    ${CMAKE_BINARY_DIR}/include/cuflash/version.h
    @ONLY
)

# Add generated include directory
target_include_directories(cuflash PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
)

# Set CUDA flags
# Note: Optimization level (-O2/-O3) is controlled by CMAKE_BUILD_TYPE.
target_compile_options(cuflash PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion
    >
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --threads=0
        -Xcompiler=-Wall,-Wextra,-Wpedantic
        $<$<BOOL:${ENABLE_FAST_MATH}>:--use_fast_math>
    >
)

# Sanitizer support (optional, for debugging)
if(ENABLE_ASAN OR ENABLE_UBSAN)
    set(SANITIZER_FLAGS "")
    if(ENABLE_ASAN)
        list(APPEND SANITIZER_FLAGS "-fsanitize=address")
    endif()
    if(ENABLE_UBSAN)
        list(APPEND SANITIZER_FLAGS "-fsanitize=undefined")
    endif()
    
    target_compile_options(cuflash PRIVATE ${SANITIZER_FLAGS} -fno-omit-frame-pointer)
    target_link_options(cuflash PRIVATE ${SANITIZER_FLAGS})
    
    message(STATUS "Sanitizers enabled: ${SANITIZER_FLAGS}")
endif()

# Tests
if(BUILD_TESTS)
    include(FetchContent)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        f8d7d77c06936315286eb55f8de22cd23c188571  # v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    if(ENABLE_RAPIDCHECK)
        FetchContent_Declare(
            rapidcheck
            GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git
            GIT_TAG        ff6af6fc683159deb51c543b065eba14dfcf329b
            GIT_SHALLOW    FALSE
        )
        set(RC_ENABLE_GTEST ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(rapidcheck)
    endif()

    add_executable(cuflash_tests
        tests/test_main.cpp
        tests/unit/test_online_softmax.cu
        tests/unit/test_forward.cu
        tests/unit/test_flash_decoding.cu
        tests/unit/test_backward.cu
        tests/unit/test_causal_mask.cu
        tests/unit/test_error_handling.cu
        tests/unit/test_dtype.cu
        tests/unit/test_numerical_stability.cu
        tests/unit/test_stress_edge_cases.cu
        tests/unit/test_tile_io.cu
        tests/unit/test_matmul.cu
    )
    target_link_libraries(cuflash_tests PRIVATE cuflash gtest)
    target_compile_options(cuflash_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
        >
    )
    set_target_properties(cuflash_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    if(ENABLE_RAPIDCHECK)
        target_link_libraries(cuflash_tests PRIVATE rapidcheck_gtest)
        target_compile_definitions(cuflash_tests PRIVATE CUFLASH_ENABLE_RAPIDCHECK=1)
    else()
        target_compile_definitions(cuflash_tests PRIVATE CUFLASH_ENABLE_RAPIDCHECK=0)
    endif()

    add_executable(cuflash_api_smoke tests/integration/test_api_smoke.cpp)
    target_link_libraries(cuflash_api_smoke PRIVATE cuflash)
    set_target_properties(cuflash_api_smoke PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # Sanitize the test drivers too, not just the library: host-side bugs most
    # often surface in the test harness code. SANITIZER_FLAGS is populated by
    # the ENABLE_ASAN/ENABLE_UBSAN block above.
    if(ENABLE_ASAN OR ENABLE_UBSAN)
        foreach(_san_target cuflash_tests cuflash_api_smoke)
            target_compile_options(${_san_target} PRIVATE ${SANITIZER_FLAGS}
                                                           -fno-omit-frame-pointer)
            target_link_options(${_san_target} PRIVATE ${SANITIZER_FLAGS})
        endforeach()
    endif()

    if(BUILD_TESTING)
        add_test(NAME cuflash_api_smoke COMMAND cuflash_api_smoke)

        include(GoogleTest)
        gtest_discover_tests(cuflash_tests DISCOVERY_TIMEOUT 60)
    endif()

    add_custom_target(check
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS cuflash_tests cuflash_api_smoke
    )
endif()

# Example executable
if(BUILD_EXAMPLES)
    add_executable(example examples/basic_usage.cu)
    target_link_libraries(example PRIVATE cuflash)
    set_target_properties(example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if(BUILD_TESTING AND BUILD_EXAMPLES)
    add_test(NAME cuflash_example_build COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target example)
    set_tests_properties(cuflash_example_build PROPERTIES LABELS "smoke")
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        344117638c8ff7e239044fd0fa7085839fc03021  # v1.8.3
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)

    add_executable(cuflash_bench benchmarks/bench_flash_attention.cu)
    target_link_libraries(cuflash_bench PRIVATE cuflash benchmark::benchmark)
    set_target_properties(cuflash_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    # Private include dirs so the benchmark can reuse the internal tiling
    # config (ForwardTilingConfig) to model logical HBM traffic accurately.
    target_include_directories(cuflash_bench PRIVATE
        ${CMAKE_SOURCE_DIR}/src/kernels
        ${CMAKE_SOURCE_DIR}/src
    )
    target_compile_options(cuflash_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
        >
    )
endif()

if(BUILD_TESTING AND BUILD_SHARED_LIBS)
    set(CUFLASH_PACKAGE_SMOKE_BUILD_DIR ${CMAKE_BINARY_DIR}/package-smoke)
    set(CUFLASH_PACKAGE_SMOKE_INSTALL_DIR ${CMAKE_BINARY_DIR}/package-install)

    add_test(
        NAME cuflash_package_smoke
        COMMAND ${CMAKE_COMMAND}
            -Dproject_build_dir=${CMAKE_BINARY_DIR}
            -Dsource_dir=${CMAKE_SOURCE_DIR}/tests/package_smoke
            -Dbinary_dir=${CUFLASH_PACKAGE_SMOKE_BUILD_DIR}
            -Dinstall_dir=${CUFLASH_PACKAGE_SMOKE_INSTALL_DIR}
            -Dgenerator=${CMAKE_GENERATOR}
            -Dbuild_type=${CMAKE_BUILD_TYPE}
            -Dc_compiler=${CMAKE_C_COMPILER}
            -Dcxx_compiler=${CMAKE_CXX_COMPILER}
            -Dcuda_architectures=${CMAKE_CUDA_ARCHITECTURES}
            -Dpackage_smoke_enable_asan=${ENABLE_ASAN}
            -Dpackage_smoke_enable_ubsan=${ENABLE_UBSAN}
            -Dctest_command=${CMAKE_CTEST_COMMAND}
            -P ${CMAKE_SOURCE_DIR}/cmake/run_package_smoke.cmake
    )
    set_tests_properties(cuflash_package_smoke PROPERTIES LABELS "smoke;package")

    # PyTorch comparison test (optional, requires Python and PyTorch)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME cuflash_pytorch_comparison
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/integration/test_pytorch_comparison.py
        )
        set_tests_properties(cuflash_pytorch_comparison PROPERTIES
            LABELS "integration;pytorch"
            SKIP_RETURN_CODE 77
        )
    endif()
endif()

# Installation
install(TARGETS cuflash
    EXPORT cuflashTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)

# Install generated version header
install(FILES ${CMAKE_BINARY_DIR}/include/cuflash/version.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cuflash
)

install(EXPORT cuflashTargets
    FILE cuflashTargets.cmake
    NAMESPACE cuflash::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuflash
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuflashConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cuflashConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuflash
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/cuflashConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/cuflashConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/cuflashConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuflash
)

export(EXPORT cuflashTargets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/cuflashTargets.cmake
    NAMESPACE cuflash::
)

export(PACKAGE cuflash)
