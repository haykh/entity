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

# set memory space
if(${Kokkos_ENABLE_CUDA})
  add_compile_definitions(CUDA_ENABLED)
  set(ACC_MEM_SPACE Kokkos::CudaSpace)
elseif(${Kokkos_ENABLE_HIP})
  add_compile_definitions(HIP_ENABLED)
  set(ACC_MEM_SPACE Kokkos::HIPSpace)
else()
  set(ACC_MEM_SPACE Kokkos::HostSpace)
endif()

set(HOST_MEM_SPACE Kokkos::HostSpace)

# set execution space
if(${Kokkos_ENABLE_CUDA})
  set(ACC_EXE_SPACE Kokkos::Cuda)
elseif(${Kokkos_ENABLE_HIP})
  set(ACC_EXE_SPACE Kokkos::HIP)
elseif(${Kokkos_ENABLE_OPENMP})
  set(ACC_EXE_SPACE Kokkos::OpenMP)
else()
  set(ACC_EXE_SPACE Kokkos::Serial)
endif()

if(${Kokkos_ENABLE_OPENMP})
  set(HOST_EXE_SPACE Kokkos::OpenMP)
else()
  set(HOST_EXE_SPACE Kokkos::Serial)
endif()

add_compile_options("-D AccelExeSpace=${ACC_EXE_SPACE}")
add_compile_options("-D AccelMemSpace=${ACC_MEM_SPACE}")
add_compile_options("-D HostExeSpace=${HOST_EXE_SPACE}")
add_compile_options("-D HostMemSpace=${HOST_MEM_SPACE}")

if(${BUILD_TESTING} STREQUAL "OFF")
  set(Kokkos_ENABLE_TESTS
      OFF
      CACHE BOOL "Kokkos tests")
else()
  set(Kokkos_ENABLE_TESTS
      ON
      CACHE BOOL "Kokkos tests")
endif()
