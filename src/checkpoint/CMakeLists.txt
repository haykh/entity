# ------------------------------
# @defines: ntt_checkpoint [STATIC/SHARED]
#
# @sources:
#
# * writer.cpp
# * reader.cpp
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
# * ADIOS2 [required]
# * mpi [optional]
# ------------------------------

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SOURCES ${SRC_DIR}/writer.cpp ${SRC_DIR}/reader.cpp)
add_library(ntt_checkpoint ${SOURCES})

set(libs ntt_global)
add_dependencies(ntt_checkpoint ${libs})
target_link_libraries(ntt_checkpoint PUBLIC ${libs})
target_link_libraries(ntt_checkpoint PRIVATE stdc++fs)

target_include_directories(
  ntt_checkpoint
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../)
