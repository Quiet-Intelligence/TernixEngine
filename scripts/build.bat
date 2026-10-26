@echo off
rem Configure and build from any shell: loads the MSVC environment, then runs CMake + Ninja.
rem   scripts\build.bat [build-dir] [extra cmake configure args...]
setlocal
set "BUILD_DIR=%~1"
if "%BUILD_DIR%"=="" set "BUILD_DIR=build"
shift
set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" ( echo vcvars64.bat not found & exit /b 1 )
call "%VCVARS%" >nul || exit /b 1
set "EXTRA="
:args
if "%~1"=="" goto run
set "EXTRA=%EXTRA% %~1"
shift
goto args
:run
cmake -S . -B %BUILD_DIR% -G Ninja -DCMAKE_BUILD_TYPE=Release %EXTRA% || exit /b 1
cmake --build %BUILD_DIR% || exit /b 1
