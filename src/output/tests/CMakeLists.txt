# ------------------------------
# @brief: Generates tests for the `ntt_output` module
#
# @uses:
#
# * kokkos [required]
# * mpi [optional]
# * adios2 [optional]
# ------------------------------

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

function(gen_test title is_parallel)
  set(exec test-output-${title}.xc)
  set(src ${title}.cpp)
  add_executable(${exec} ${src})

  set(libs ntt_output ntt_global ntt_metrics ntt_framework)
  add_dependencies(${exec} ${libs})
  target_link_libraries(${exec} PRIVATE ${libs} stdc++fs)

  if(${is_parallel})
    add_test(NAME "OUTPUT::${title}"
             COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "4"
                     "${exec}")
  else()
    add_test(NAME "OUTPUT::${title}" COMMAND "${exec}")
  endif()
endfunction()

if(NOT ${mpi})
  gen_test(fields false)
  gen_test(writer-nompi false)
else()
  gen_test(writer-mpi true)
endif()
