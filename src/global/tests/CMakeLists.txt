# ------------------------------
# @brief: Generates tests for the `ntt_global` module
#
# @uses:
#
# * kokkos [required]
# * plog [required]
# * mpi [optional]
#
# !TODO:
#
# * add optional tests for the `mpi_aliases.h`
# ------------------------------

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

function(gen_test title)
  set(exec test-global-${title}.xc)
  set(src ${title}.cpp)
  add_executable(${exec} ${src})

  set(libs ntt_global)
  add_dependencies(${exec} ${libs})
  target_link_libraries(${exec} PRIVATE ${libs})

  add_test(NAME "GLOBAL::${title}" COMMAND "${exec}")
endfunction()

gen_test(global)
gen_test(enums)
gen_test(kokkos_aliases)
gen_test(directions)
gen_test(comparators)
gen_test(numeric)
gen_test(param_container)
gen_test(sorting)
