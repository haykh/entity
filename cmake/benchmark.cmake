# cmake-lint: disable=C0103

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(exec benchmark.xc)
set(src ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/benchmark.cpp)

add_executable(${exec} ${src})

set(libs ntt_global ntt_metrics ntt_kernels ntt_archetypes ntt_framework)
if(${output})
  list(APPEND libs ntt_output)
endif()
add_dependencies(${exec} ${libs})
target_link_libraries(${exec} PRIVATE ${libs} stdc++fs)
