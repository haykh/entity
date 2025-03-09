# ------------------------------
# @brief: Generates tests for the `ntt_framework` module
#
# @uses:
#
# * kokkos [required]
# * plog [required]
# * toml11 [required]
# * mpi [optional]
# * adios2 [optional]
#
# !TODO:
#
# * add tests for mesh separately
# * add test for 3D metadomain
# ------------------------------

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

function(gen_test title is_parallel)
  set(exec test-framework-${title}.xc)
  set(src ${title}.cpp)
  add_executable(${exec} ${src})

  set(libs ntt_framework ntt_global ntt_metrics)
  add_dependencies(${exec} ${libs})
  target_link_libraries(${exec} PRIVATE ${libs})

  if(${is_parallel})
    add_test(NAME "FRAMEWORK::${title}"
             COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "4"
                     "${exec}")
  else()
    add_test(NAME "FRAMEWORK::${title}" COMMAND "${exec}")
  endif()
endfunction()

if(${mpi})
  gen_test(comm_mpi true)
else()
  gen_test(parameters false)
  gen_test(particles false)
  gen_test(fields false)
  gen_test(grid_mesh false)
  if(${DEBUG})
    gen_test(metadomain false)
  endif()
  gen_test(comm_nompi false)
endif()

# this test is only run manually to ensure ...
# ... command line args are working properly ...
# ... and that the logging is done correctly
#
# gen_test(simulation)
