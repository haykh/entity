# ------------------------------
# @defines: ntt_output [STATIC/SHARED]
#
# @sources:
#
# * writer.cpp
# * fields.cpp
# * utils/interpret_prompt.cpp
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
set(SOURCES ${SRC_DIR}/writer.cpp ${SRC_DIR}/fields.cpp
            ${SRC_DIR}/utils/interpret_prompt.cpp)
add_library(ntt_output ${SOURCES})

set(libs ntt_global)
add_dependencies(ntt_output ${libs})
target_link_libraries(ntt_output PUBLIC ${libs})
target_link_libraries(ntt_output PRIVATE stdc++fs)

target_include_directories(
  ntt_output
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../)
