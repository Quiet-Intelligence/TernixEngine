@echo off
rem TernixEngine dev pipeline: configure -> build -> unit tests -> Python tests -> benchmarks -> charts/README tables.
rem Run from an "x64 Native Tools Command Prompt for VS" (MSVC + Ninja on PATH).
rem   run_all.bat               full pipeline
rem   run_all.bat --build-only  configure + build only
rem Optional: set PYTHON_EXE to the interpreter whose ABI the extension should target.
setlocal

if "%PYTHON_EXE%"=="" set PYTHON_EXE=python

echo === TernixEngine Dev Pipeline ===

echo.
echo [0/6] Python dependencies for plotting and the binding tests...
"%PYTHON_EXE%" -m pip install --quiet numpy matplotlib pytest

echo.
echo [1/6] Configuring CMake (Release)...
if exist build\CMakeCache.txt del /F /Q build\CMakeCache.txt
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE="%PYTHON_EXE%"
if %ERRORLEVEL% neq 0 ( echo CMake configuration failed & exit /b %ERRORLEVEL% )

echo.
echo [2/6] Compiling...
cmake --build build
if %ERRORLEVEL% neq 0 ( echo Compilation failed & exit /b %ERRORLEVEL% )

if "%1"=="--build-only" exit /b 0

echo.
echo [3/6] C++ and CUDA unit tests (CUDA tests are built only when nvcc is found)...
pushd build
ctest --output-on-failure
set CTEST_RC=%ERRORLEVEL%
popd
if %CTEST_RC% neq 0 ( echo Unit tests failed & exit /b %CTEST_RC% )

echo.
echo [4/6] Python binding tests...
set PYTHONPATH=%CD%\build
"%PYTHON_EXE%" -m pytest tests\test_python_binding.py -q
if %ERRORLEVEL% neq 0 ( echo Python tests failed & exit /b %ERRORLEVEL% )

echo.
echo [5/6] Benchmarks (median of 5 repetitions, pinned to logical CPU 2 = a P-core on hybrid parts)...
if not exist benchmarks mkdir benchmarks
if "%TERNIX_BENCH_CPU%"=="" set TERNIX_BENCH_CPU=2
build\bench_simd_math.exe --benchmark_repetitions=5 --benchmark_report_aggregates_only=true --benchmark_format=json --benchmark_out=benchmarks\results.json > nul
if exist build\bench_cuda.exe (
    build\bench_cuda.exe benchmarks\results_cuda.json
    if %ERRORLEVEL% neq 0 ( echo GPU benchmark failed verification & exit /b %ERRORLEVEL% )
)

echo.
echo [6/6] Charts and README tables...
"%PYTHON_EXE%" scripts\plot_results.py --input benchmarks\results.json --cuda-input benchmarks\results_cuda.json --output benchmarks --readme README.md

echo.
echo Pipeline successful. Artifacts: benchmarks\*.png, README.md tables.
