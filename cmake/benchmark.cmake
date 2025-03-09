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

set(exec benchmark.xc)
set(src ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/benchmark.cpp)

add_executable(${exec} ${src})

set(libs ntt_global ntt_metrics ntt_kernels ntt_archetypes ntt_framework)
if(${output})
  list(APPEND libs ntt_output ntt_checkpoint)
endif()
add_dependencies(${exec} ${libs})
target_link_libraries(${exec} PRIVATE ${libs} stdc++fs)
