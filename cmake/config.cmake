# -------------------------------- Precision ------------------------------- #
function(set_precision precision_name)
  list(FIND precisions ${precision_name} PRECISION_FOUND)

  if(${PRECISION_FOUND} EQUAL -1)
    message(
      FATAL_ERROR
        "Invalid precision: ${precision_name}\nValid options are: ${precisions}"
    )
  endif()

  if(${precision_name} STREQUAL "single")
    add_compile_options("-DSINGLE_PRECISION")
  endif()
endfunction()

# ---------------------------- Problem generator --------------------------- #
function(set_problem_generator pgen_name)
  file(GLOB_RECURSE PGENS "${CMAKE_CURRENT_SOURCE_DIR}/setups/**/pgen.hpp"
       "${CMAKE_CURRENT_SOURCE_DIR}/setups/pgen.hpp")
  foreach(PGEN ${PGENS})
    get_filename_component(PGEN_NAME ${PGEN} DIRECTORY)
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/setups/" "" PGEN_NAME
                   ${PGEN_NAME})
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/setups" "" PGEN_NAME
                   ${PGEN_NAME})
    list(APPEND PGEN_NAMES ${PGEN_NAME})
  endforeach()
  list(FIND PGEN_NAMES ${pgen_name} PGEN_FOUND)
  if(NOT ${pgen_name} STREQUAL "." AND ${PGEN_FOUND} EQUAL -1)
    message(
      FATAL_ERROR
        "Invalid problem generator: ${pgen_name}\nValid options are: ${PGEN_NAMES}"
    )
  endif()
  set(PGEN
      ${pgen_name}
      PARENT_SCOPE)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/setups/${pgen_name})
  set(PGEN_FOUND
      TRUE
      PARENT_SCOPE)
  set(problem_generators
      ${PGEN_NAMES}
      PARENT_SCOPE)
endfunction()
