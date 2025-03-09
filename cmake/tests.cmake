include(CTest)
enable_testing()

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_subdirectory(${SRC_DIR}/global ${CMAKE_CURRENT_BINARY_DIR}/global)
add_subdirectory(${SRC_DIR}/metrics ${CMAKE_CURRENT_BINARY_DIR}/metrics)
add_subdirectory(${SRC_DIR}/kernels ${CMAKE_CURRENT_BINARY_DIR}/kernels)
add_subdirectory(${SRC_DIR}/archetypes ${CMAKE_CURRENT_BINARY_DIR}/archetypes)
add_subdirectory(${SRC_DIR}/framework ${CMAKE_CURRENT_BINARY_DIR}/framework)
if(${output})
  add_subdirectory(${SRC_DIR}/output ${CMAKE_CURRENT_BINARY_DIR}/output)
  add_subdirectory(${SRC_DIR}/checkpoint ${CMAKE_CURRENT_BINARY_DIR}/checkpoint)
endif()

if(${mpi})
  # tests with mpi
  if(${output})
    add_subdirectory(${SRC_DIR}/output/tests
                     ${CMAKE_CURRENT_BINARY_DIR}/output/tests)
    add_subdirectory(${SRC_DIR}/checkpoint/tests
                     ${CMAKE_CURRENT_BINARY_DIR}/checkpoint/tests)
    add_subdirectory(${SRC_DIR}/framework/tests
                     ${CMAKE_CURRENT_BINARY_DIR}/framework/tests)
  endif()
else()
  # tests without mpi
  add_subdirectory(${SRC_DIR}/global/tests
                   ${CMAKE_CURRENT_BINARY_DIR}/global/tests)
  add_subdirectory(${SRC_DIR}/metrics/tests
                   ${CMAKE_CURRENT_BINARY_DIR}/metrics/tests)
  add_subdirectory(${SRC_DIR}/kernels/tests
                   ${CMAKE_CURRENT_BINARY_DIR}/kernels/tests)
  add_subdirectory(${SRC_DIR}/archetypes/tests
                   ${CMAKE_CURRENT_BINARY_DIR}/archetypes/tests)
  add_subdirectory(${SRC_DIR}/framework/tests
                   ${CMAKE_CURRENT_BINARY_DIR}/framework/tests)
  if(${output})
    add_subdirectory(${SRC_DIR}/output/tests
                     ${CMAKE_CURRENT_BINARY_DIR}/output/tests)
    add_subdirectory(${SRC_DIR}/checkpoint/tests
                     ${CMAKE_CURRENT_BINARY_DIR}/checkpoint/tests)
  endif()
endif()
