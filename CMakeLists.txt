cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0110 NEW)

set(PROJECT_NAME entity)

project(
  ${PROJECT_NAME}
  VERSION 1.2.0
  LANGUAGES CXX C)
add_compile_options("-D ENTITY_VERSION=\"${PROJECT_VERSION}\"")
execute_process(
  COMMAND
    bash -c
    "git diff --quiet src/ && echo $(git rev-parse HEAD) || echo $(git rev-parse HEAD)-mod"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_HASH
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "Git hash: ${GIT_HASH}")
add_compile_options("-D ENTITY_GIT_HASH=\"${GIT_HASH}\"")

set(CMAKE_CXX_EXTENSIONS OFF)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/styling.cmake)

# ----------------------------- Configurations ----------------------------- #
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/defaults.cmake)

# defaults
set(DEBUG
    ${default_debug}
    CACHE BOOL "Debug mode")

set(precision
    ${default_precision}
    CACHE STRING "Precision")
set(pgen
    ${default_pgen}
    CACHE STRING "Problem generator")

set(gui
    ${default_gui}
    CACHE BOOL "Use GUI [nttiny]")
set(output
    ${default_output}
    CACHE BOOL "Enable output")
set(mpi
    ${default_mpi}
    CACHE BOOL "Use MPI")

# -------------------------- Compilation settings -------------------------- #
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(${DEBUG} STREQUAL "OFF")
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "CMake build type")
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -DNDEBUG -Wno-unused-local-typedefs -Wno-unknown-cuda-version"
  )
else()
  set(CMAKE_BUILD_TYPE
      Debug
      CACHE STRING "CMake build type")
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -DDEBUG -Wall -Wextra -Wno-unknown-pragmas")
endif()

# options
set(precisions
    "single" "double"
    CACHE STRING "Precisions")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake)

# ------------------------- Third-Party Tests ------------------------------ #
set(BUILD_TESTING
    OFF
    CACHE BOOL "Build tests")

# ------------------------ Third-party dependencies ------------------------ #
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/kokkosConfig.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/dependencies.cmake)

find_or_fetch_dependency(Kokkos FALSE)
find_or_fetch_dependency(plog TRUE)
set(DEPENDENCIES Kokkos::kokkos)
include_directories(${plog_SRC}/include)

# -------------------------------- Main code ------------------------------- #
set_precision(${precision})

# MPI
if(${mpi})
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/MPIConfig.cmake)
  set(DEPENDENCIES ${DEPENDENCIES} MPI::MPI_CXX)
endif()

# Output
if(${output})
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/adios2Config.cmake)
  find_or_fetch_dependency(adios2 FALSE)
  if(NOT DEFINED ENV{HDF5_ROOT})
    if(DEFINED ENV{CONDA_PREFIX})
      execute_process(COMMAND bash -c "conda list | grep \"hdf5\" -q"
                      RESULT_VARIABLE HDF5_INSTALLED)
      if(HDF5_INSTALLED EQUAL 0)
        set(HDF5_ROOT $ENV{CONDA_PREFIX})
      endif()
    endif()
  endif()
  find_package(HDF5 REQUIRED)

  if(${mpi})
    set(DEPENDENCIES ${DEPENDENCIES} adios2::cxx11_mpi)
  else()
    set(DEPENDENCIES ${DEPENDENCIES} adios2::cxx11)
  endif()
endif()

link_libraries(${DEPENDENCIES})

if(TESTS)
  # ---------------------------------- Tests --------------------------------- #
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/tests.cmake)
elseif(BENCHMARK)
  # ------------------------------ Benchmark --------------------------------- #
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/benchmark.cmake)
else()
  # ----------------------------------- GUI ---------------------------------- #
  if(${gui})
    find_or_fetch_dependency(nttiny FALSE)
  endif()

  # ------------------------------- Main source ------------------------------ #
  set_problem_generator(${pgen})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src src)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/report.cmake)
endif()
