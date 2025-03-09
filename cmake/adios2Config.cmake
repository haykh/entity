# ----------------------------- Adios2 settings ---------------------------- #
set(ADIOS2_BUILD_EXAMPLES
    OFF
    CACHE BOOL "Build ADIOS2 examples")

# Language support
set(ADIOS2_USE_Python
    OFF
    CACHE BOOL "Use Python for ADIOS2")
set(ADIOS2_USE_Fortran
    OFF
    CACHE BOOL "Use Fortran for ADIOS2")

# Format/compression support
set(ADIOS2_USE_ZeroMQ
    OFF
    CACHE BOOL "Use ZeroMQ for ADIOS2")

set(ADIOS2_USE_MPI
    ${mpi}
    CACHE BOOL "Use MPI for ADIOS2")

set(ADIOS2_USE_CUDA
    OFF
    CACHE BOOL "Use CUDA for ADIOS2")

add_compile_options("-D OUTPUT_ENABLED")
