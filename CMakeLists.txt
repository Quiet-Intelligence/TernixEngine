cmake_minimum_required(VERSION 3.20)
project(TernixEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
# Build device code for the GPU actually present (CMake >= 3.24), else Ampere (cp.async needs sm_80+).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 80)   # cp.async needs sm_80+
    endif()
endif()
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)  # device code needs no C++20; older CMake lacks the CUDA20 dialect
    set(TERNIX_HAS_CUDA ON)
    message(STATUS "CUDA compiler found. Enabling GPU support.")
else()
    set(TERNIX_HAS_CUDA OFF)
    message(WARNING "CUDA compiler NOT found. Building CPU-only (AVX2) version.")
endif()

find_package(OpenMP REQUIRED)
if(TERNIX_HAS_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

include(FetchContent)

# x86-64 baseline for GCC/Clang: AVX2 + FMA + F16C + BMI2 + POPCNT; AVX-VNNI intrinsics are compiled in but only executed after a
# runtime cpuid check (src/cpu_features.cpp).
set(TERNIX_GNU_FLAGS $<$<COMPILE_LANGUAGE:CXX>:-mavx2> $<$<COMPILE_LANGUAGE:CXX>:-mfma> $<$<COMPILE_LANGUAGE:CXX>:-mf16c> $<$<COMPILE_LANGUAGE:CXX>:-mbmi2>
    $<$<COMPILE_LANGUAGE:CXX>:-mpopcnt> $<$<COMPILE_LANGUAGE:CXX>:-mavxvnni> $<$<COMPILE_LANGUAGE:CXX>:-O3>)

# Google Benchmark
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# GoogleTest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

set(TERNIX_SOURCES 
    src/cpu_features.cpp
    src/threadpool.cpp
    src/i2k.cpp
    src/mul_mat_ternary.cpp
    src/tensor.cpp
    src/simd_math.cpp
    src/bitpack.cpp
    src/format.cpp
    src/tokenizer.cpp
    src/ops.cpp
    src/head.cpp
    src/sampler.cpp
    src/model.cpp
)

if(TERNIX_HAS_CUDA)
    list(APPEND TERNIX_SOURCES src/cuda_kernels.cu src/cuda_engine.cu)
endif()

add_library(ternix_core STATIC ${TERNIX_SOURCES})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    
endif()
set_target_properties(ternix_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(TERNIX_HAS_CUDA)
    target_compile_definitions(ternix_core PUBLIC TERNIX_HAS_CUDA=1)
endif()
target_include_directories(ternix_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ternix_core PUBLIC OpenMP::OpenMP_CXX)
if(TERNIX_HAS_CUDA)
    target_link_libraries(ternix_core PUBLIC CUDA::cudart)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ternix_core PRIVATE ${TERNIX_GNU_FLAGS})
elseif(MSVC)
    target_compile_options(ternix_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2> $<$<COMPILE_LANGUAGE:CXX>:/O2>)
endif()

add_executable(ternix_cli src/main.cpp)
target_link_libraries(ternix_cli PRIVATE ternix_core)

# Tests
enable_testing()
add_executable(ternix_tests tests/test_bitpack.cpp tests/test_simd_math.cpp tests/test_ternary_matmul.cpp tests/test_i2k.cpp)
target_link_libraries(ternix_tests PRIVATE ternix_core gtest_main)
add_test(NAME CoreTests COMMAND ternix_tests)

if(TERNIX_HAS_CUDA)
    add_executable(ternix_cuda_tests tests/test_cuda_matmul.cu tests/test_cuda_engine.cu)
    target_link_libraries(ternix_cuda_tests PRIVATE ternix_core gtest_main CUDA::cudart)
    add_test(NAME CudaTests COMMAND ternix_cuda_tests)
    add_executable(bench_cuda benchmarks/bench_cuda.cu)
    target_link_libraries(bench_cuda PRIVATE ternix_core CUDA::cudart)
endif()

# Benchmarks
add_executable(bench_simd_math benchmarks/bench_simd_math.cpp)
target_link_libraries(bench_simd_math PRIVATE ternix_core benchmark::benchmark)
add_executable(bench_i2k benchmarks/bench_i2k.cpp)
target_link_libraries(bench_i2k PRIVATE ternix_core)
add_executable(bench_bitmap benchmarks/bench_bitmap.cpp)
target_link_libraries(bench_bitmap PRIVATE ternix_core)
add_executable(bench_membw benchmarks/bench_membw.cpp)
target_link_libraries(bench_membw PRIVATE ternix_core)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(bench_simd_math PRIVATE ${TERNIX_GNU_FLAGS})
    target_compile_options(ternix_cli PRIVATE ${TERNIX_GNU_FLAGS})
elseif(MSVC)
    target_compile_options(bench_simd_math PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2> $<$<COMPILE_LANGUAGE:CXX>:/O2>)
    target_compile_options(ternix_cli PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2> $<$<COMPILE_LANGUAGE:CXX>:/O2>)
endif()

# Python Bindings
pybind11_add_module(ternix_engine bindings/python/pybind11_wrapper.cpp)
target_link_libraries(ternix_engine PRIVATE ternix_core)
if(TERNIX_HAS_CUDA)
    target_link_libraries(ternix_engine PRIVATE CUDA::cudart)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ternix_engine PRIVATE ${TERNIX_GNU_FLAGS})
elseif(MSVC)
    target_compile_options(ternix_engine PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2> $<$<COMPILE_LANGUAGE:CXX>:/O2>)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    foreach(t bench_i2k bench_bitmap bench_membw ternix_tests)
        target_compile_options(${t} PRIVATE ${TERNIX_GNU_FLAGS})
    endforeach()
elseif(MSVC)
    foreach(t bench_i2k bench_bitmap bench_membw ternix_tests)
        target_compile_options(${t} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    endforeach()
endif()
