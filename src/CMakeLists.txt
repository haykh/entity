# cmake-lint: disable=C0103
# ------------------------------
# @defines: entity [STATIC/SHARED]
#
# @sources:
#
# * entity.cpp
#
# @depends:
#
# * ntt_global [required]
# * ntt_framework [required]
# * ntt_metrics [required]
# * ntt_engine [required]
# * ntt_pgen [required]
#
# @uses:
#
# * kokkos [required]
# * plog [required]
# * ADIOS2 [optional]
# * mpi [optional]
# ------------------------------

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(ENTITY ${PROJECT_NAME}.xc)
set(SOURCES ${SRC_DIR}/entity.cpp)

add_executable(${ENTITY} ${SOURCES})
set(libs ntt_global ntt_framework ntt_metrics ntt_engines ntt_pgen)
add_dependencies(${ENTITY} ${libs})
target_link_libraries(${ENTITY} PUBLIC ${libs})
install(TARGETS ${ENTITY} DESTINATION bin)
