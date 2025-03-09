# ----------------------------- Kokkos settings ---------------------------- #
if(${DEBUG} STREQUAL "OFF")
  set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION
      ON
      CACHE BOOL "Kokkos aggressive vectorization")
  set(Kokkos_ENABLE_COMPILER_WARNINGS
      OFF
      CACHE BOOL "Kokkos compiler warnings")
  set(Kokkos_ENABLE_DEBUG
      OFF
      CACHE BOOL "Kokkos debug")
  set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK
      OFF
      CACHE BOOL "Kokkos debug bounds check")
else()
  set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION
      OFF
      CACHE BOOL "Kokkos aggressive vectorization")
  set(Kokkos_ENABLE_COMPILER_WARNINGS
      ON
      CACHE BOOL "Kokkos compiler warnings")
  set(Kokkos_ENABLE_DEBUG
      ON
      CACHE BOOL "Kokkos debug")
  set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK
      ON
      CACHE BOOL "Kokkos debug bounds check")
endif()

set(Kokkos_ENABLE_HIP
    ${default_KOKKOS_ENABLE_HIP}
    CACHE BOOL "Enable HIP")
set(Kokkos_ENABLE_CUDA
    ${default_KOKKOS_ENABLE_CUDA}
    CACHE BOOL "Enable CUDA")
set(Kokkos_ENABLE_OPENMP
    ${default_KOKKOS_ENABLE_OPENMP}
    CACHE BOOL "Enable OpenMP")

if(${BUILD_TESTING} STREQUAL "OFF")
  set(Kokkos_ENABLE_TESTS
      OFF
      CACHE BOOL "Kokkos tests")
else()
  set(Kokkos_ENABLE_TESTS
      ON
      CACHE BOOL "Kokkos tests")
endif()
