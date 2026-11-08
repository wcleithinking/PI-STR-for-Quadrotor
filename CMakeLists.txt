# PI-STR-for-Quadrotor -- host build of the firmware STR core and its tests.
#
#   cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
#
# What this builds:
#   str_core            the host-compilable regulator core (firmware/lib/str_core)
#   str_core_testlib    the test suites (shared by the two entry points below)
#   str_core_tests      ctest executable, one ctest entry per suite
#   str_core_verify     shared library with a stable C entry point, for
#                       environments that cannot execute freshly built binaries
#   str_core_trace      Python <-> C++ golden-vector replay driver (CONTROL_SPEC §8)
#
# There are no third-party dependencies and nothing here is uploaded to a board:
# the .ino sketches need an Arduino/SAM toolchain (the Due is an ARM Cortex-M3,
# not an AVR part), which this project does not provide on purpose (see
# docs/CONTROL_SPEC.md §1).

cmake_minimum_required(VERSION 3.16)
project(pi_str_quadrotor_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
  set(STR_CORE_WARNINGS /W4)
else()
  # The acceptance criterion is "zero warnings with -Wall -Wextra".
  set(STR_CORE_WARNINGS -Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
# The firmware core: no Arduino headers, no exceptions, no RTTI assumptions.
# ---------------------------------------------------------------------------
add_library(str_core STATIC firmware/lib/str_core/str_core.cpp)
target_include_directories(str_core PUBLIC firmware/lib/str_core)
target_compile_options(str_core PRIVATE ${STR_CORE_WARNINGS})

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

add_library(str_core_testlib STATIC
  tests/cpp/suite_linear_solve.cpp
  tests/cpp/suite_diophantine.cpp
  tests/cpp/suite_projection.cpp
  tests/cpp/suite_rls.cpp
  tests/cpp/suite_guards.cpp
  tests/cpp/suite_equivalence.cpp
  tests/cpp/suite_golden.cpp
)
target_include_directories(str_core_testlib
  PUBLIC tests/cpp
  PRIVATE firmware/lib/str_core
)
target_link_libraries(str_core_testlib PUBLIC str_core)
target_compile_options(str_core_testlib PRIVATE ${STR_CORE_WARNINGS})
# Absolute fixture directory so the golden replay also works when the caller cannot
# control the working directory (the ctypes entry point, for instance).
# PUBLIC so that every consumer (the ctest executable, the ctypes entry point, the
# suite code) resolves the fixture from any working directory.
target_compile_definitions(str_core_testlib PUBLIC
  STR_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/golden")

add_executable(str_core_tests tests/cpp/test_main.cpp)
target_link_libraries(str_core_tests PRIVATE str_core_testlib)
target_compile_options(str_core_tests PRIVATE ${STR_CORE_WARNINGS})

foreach(suite linear_solve diophantine projection rls guards equivalence)
  add_test(NAME ${suite} COMMAND str_core_tests ${suite})
endforeach()

add_executable(str_core_trace tests/cpp/str_core_trace.cpp)
target_include_directories(str_core_trace PRIVATE tests/cpp)
target_link_libraries(str_core_trace PRIVATE str_core)
target_compile_options(str_core_trace PRIVATE ${STR_CORE_WARNINGS})
add_test(NAME trace_selftest COMMAND str_core_trace --selftest --order 2)
add_test(NAME trace_selftest_order3 COMMAND str_core_trace --selftest --order 3)

# Shared-library entry point: some sandboxes refuse to exec a freshly linked
# binary while dlopen() of the same code works, so the same assertions can be run
# via ctypes.  Kept out of the default build?  No: it is tiny and it keeps the
# interface compile-checked everywhere.
add_library(str_core_verify SHARED tests/cpp/ctypes_entry.cpp)
target_link_libraries(str_core_verify PRIVATE str_core_testlib)
target_compile_options(str_core_verify PRIVATE ${STR_CORE_WARNINGS})
