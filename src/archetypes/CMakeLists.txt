# ------------------------------
# @defines: ntt_archetypes [INTERFACE]
#
# @includes:
#
# * ../
#
# @depends:
#
# * ntt_global [required]
# * ntt_kernels [required]
#
# @uses:
#
# * kokkos [required]
# * mpi [optional]
# ------------------------------

add_library(ntt_archetypes INTERFACE)

set(libs ntt_global ntt_kernels)
add_dependencies(ntt_archetypes ${libs})
target_link_libraries(ntt_archetypes INTERFACE ${libs})

target_include_directories(ntt_archetypes
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../)

