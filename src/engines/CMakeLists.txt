# cmake-lint: disable=C0103
# ------------------------------
# @defines: ntt_engines [STATIC/SHARED]
#
# @sources:
#
# @includes:
#
# * ../
#
# @depends:
#
# * ntt_global [required]
# * ntt_framework [required]
# * ntt_metrics [required]
# * ntt_kernels [required]
# * ntt_archetypes [required]
# * ntt_pgen [required]
# * ntt_output [optional]
#
# @uses:
#
# * kokkos [required]
# * plog [required]
# * adios2 [optional]
# * hdf5 [optional]
# * mpi [optional]
# ------------------------------

add_library(ntt_engines INTERFACE)

set(libs ntt_global ntt_framework ntt_metrics ntt_archetypes ntt_kernels
         ntt_pgen)
if(${output})
  list(APPEND libs ntt_output)
endif()
add_dependencies(ntt_engines ${libs})
target_link_libraries(ntt_engines INTERFACE ${libs})
target_compile_definitions(ntt_engines INTERFACE PGEN=\"${PGEN}\")

target_include_directories(
  ntt_engines
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../)
