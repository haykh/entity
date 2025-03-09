# ------------------------------
# @defines: ntt_metrics [INTERFACE]
#
# @includes:
#
# * ../
#
# @depends:
#
# * ntt_global [required]
#
# @uses:
#
# * kokkos [required]
# ------------------------------

add_library(ntt_metrics INTERFACE)

set(libs ntt_global)
add_dependencies(ntt_metrics ${libs})
target_link_libraries(ntt_metrics INTERFACE ${libs})

target_include_directories(ntt_metrics
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../)

