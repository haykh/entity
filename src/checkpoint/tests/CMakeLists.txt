# ------------------------------
# @brief: Generates tests for the `ntt_checkpoint` module
#
# @uses:
#
# * kokkos [required]
# * adios2 [required]
# * mpi [optional]
# ------------------------------

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

function(gen_test title)
  set(exec test-output-${title}.xc)
  set(src ${title}.cpp)
  add_executable(${exec} ${src})

  set(libs ntt_checkpoint ntt_global)
  add_dependencies(${exec} ${libs})
  target_link_libraries(${exec} PRIVATE ${libs} stdc++fs)

  add_test(NAME "CHECKPOINT::${title}" COMMAND "${exec}")
endfunction()

if(NOT ${mpi})
  gen_test(checkpoint-nompi)
else()
  gen_test(checkpoint-mpi)
endif()
