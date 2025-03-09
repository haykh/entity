set(Kokkos_REPOSITORY
    https://github.com/kokkos/kokkos.git
    CACHE STRING "Kokkos repository")
set(plog_REPOSITORY
    https://github.com/SergiusTheBest/plog.git
    CACHE STRING "plog repository")

# set (adios2_REPOSITORY https://github.com/ornladios/ADIOS2.git CACHE STRING
# "ADIOS2 repository")
function(check_internet_connection)
  if(OFFLINE STREQUAL "ON")
    set(FETCHCONTENT_FULLY_DISCONNECTED
        ON
        CACHE BOOL "Connection status")
    message(STATUS "${Blue}Offline mode.${ColorReset}")
  else()
    execute_process(
      COMMAND ping 8.8.8.8 -c 2
      RESULT_VARIABLE NO_CONNECTION
      OUTPUT_QUIET)

    if(NO_CONNECTION GREATER 0)
      set(FETCHCONTENT_FULLY_DISCONNECTED
          ON
          CACHE BOOL "Connection status")
      message(
        STATUS "${Red}No internet connection. Fetching disabled.${ColorReset}")
    else()
      set(FETCHCONTENT_FULLY_DISCONNECTED
          OFF
          CACHE BOOL "Connection status")
      message(STATUS "${Green}Internet connection established.${ColorReset}")
    endif()
  endif()
endfunction()

function(find_or_fetch_dependency package_name header_only)
  if(NOT header_only)
    find_package(${package_name} QUIET)
  endif()

  if(NOT ${package_name}_FOUND)
    if(DEFINED ${package_name}_REPOSITORY AND NOT
                                              FETCHCONTENT_FULLY_DISCONNECTED)
      # fetching package
      message(
        STATUS
          "${Blue}${package_name} not found. Fetching from ${${package_name}_REPOSITORY}${ColorReset}"
      )
      include(FetchContent)
      if(${package_name} STREQUAL "Kokkos")
        FetchContent_Declare(
          ${package_name}
          GIT_REPOSITORY ${${package_name}_REPOSITORY}
          GIT_TAG 4.3.00)
      else()
        FetchContent_Declare(${package_name}
                             GIT_REPOSITORY ${${package_name}_REPOSITORY})
      endif()
      FetchContent_MakeAvailable(${package_name})

      set(lower_pckg_name ${package_name})
      string(TOLOWER ${lower_pckg_name} lower_pckg_name)

      set(${package_name}_SRC
          ${CMAKE_CURRENT_BINARY_DIR}/_deps/${lower_pckg_name}-src
          CACHE PATH "Path to ${package_name} src")
      set(${package_name}_FETCHED
          TRUE
          CACHE BOOL "Whether ${package_name} was fetched")
      message(STATUS "${Green}${package_name} fetched.${ColorReset}")

    else()
      # get as submodule
      message(
        STATUS
          "${Yellow}${package_name} not found. Using as submodule.${ColorReset}"
      )

      set(${package_name}_FETCHED
          FALSE
          CACHE BOOL "Whether ${package_name} was fetched")

      if(NOT FETCHCONTENT_FULLY_DISCONNECTED)
        message(
          STATUS "${GREEN}Updating ${package_name} submodule.${ColorReset}")
        execute_process(
          COMMAND git submodule update --init --remote
                  ${CMAKE_CURRENT_SOURCE_DIR}/extern/${package_name}
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
      endif()

      add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/${package_name}
                       extern/${package_name})
      set(${package_name}_SRC
          ${CMAKE_CURRENT_SOURCE_DIR}/extern/${package_name}
          CACHE PATH "Path to ${package_name} src")
      set(${package_name}_BUILD_DIR
          ${CMAKE_CURRENT_SOURCE_DIR}/build/extern/${package_name}
          CACHE PATH "Path to ${package_name} build")
    endif()
  else()
    message(STATUS "${Green}${package_name} found.${ColorReset}")
    set(${package_name}_FETCHED
        FALSE
        CACHE BOOL "Whether ${package_name} was fetched")
    set(${package_name}_VERSION
        ${${package_name}_VERSION}
        CACHE INTERNAL "${package_name} version")
  endif()

  if(${package_name} STREQUAL "adios2")
    if(NOT DEFINED adios2_VERSION OR adios2_VERSION STREQUAL "")
      get_directory_property(adios2_VERSION DIRECTORY ${adios2_BUILD_DIR}
                                                      DEFINITION ADIOS2_VERSION)
      set(adios2_VERSION
          ${adios2_VERSION}
          CACHE INTERNAL "ADIOS2 version")
    endif()
  endif()

  if(${package_name} STREQUAL "Kokkos")
    if(NOT DEFINED Kokkos_VERSION OR Kokkos_VERSION STREQUAL "")
      get_directory_property(Kokkos_VERSION DIRECTORY ${Kokkos_SRC} DEFINITION
                                                      Kokkos_VERSION)
      set(Kokkos_VERSION
          ${Kokkos_VERSION}
          CACHE INTERNAL "Kokkos version")
    endif()
  endif()
endfunction()

check_internet_connection()
