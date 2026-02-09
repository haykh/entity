include(CTest)
enable_testing()

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(TEST_DIRECTORIES "")

if(NOT ${mpi})
  list(APPEND TEST_DIRECTORIES global)
  list(APPEND TEST_DIRECTORIES metrics)
  list(APPEND TEST_DIRECTORIES kernels)
  list(APPEND TEST_DIRECTORIES archetypes)
  list(APPEND TEST_DIRECTORIES framework)
elseif(${mpi} AND ${output})
  list(APPEND TEST_DIRECTORIES framework)
endif()

list(APPEND TEST_DIRECTORIES output)

foreach(test_dir IN LISTS TEST_DIRECTORIES)
  add_subdirectory(${SRC_DIR}/${test_dir}/tests
                   ${CMAKE_CURRENT_BINARY_DIR}/${test_dir}/tests)
endforeach()
