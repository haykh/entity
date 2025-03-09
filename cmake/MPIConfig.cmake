find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})
add_compile_options("-D MPI_ENABLED")

