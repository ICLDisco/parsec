# - Find COREBLAS include dirs and libraries
# Use this module by invoking find_package with the form:
#  find_package(COREBLAS
#               [REQUIRED]) # Fail with error if coreblas is not found
# This module finds headers and coreblas library.
# Results are reported in variables:
#  COREBLAS_FOUND           - True if headers and requested libraries were found
#  COREBLAS_INCLUDE_DIRS    - coreblas include directories
#  COREBLAS_LIBRARY_DIRS    - Link directories for coreblas libraries
#  COREBLAS_LIBRARIES       - coreblas component libraries to be linked
#
#=============================================================================
# Copyright (c) 2009-2013 The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
#=============================================================================

unset(COREBLAS_C_COMPILE_SUCCESS)
unset(COREBLAS_F_COMPILE_SUCCESS)

# First we try to use pkg-config to find what we're looking for
# in the directory specified by the COREBLAS_DIR or COREBLAS_PKG_DIR
if(COREBLAS_DIR)
  if(NOT COREBLAS_PKG_DIR)
    set(COREBLAS_PKG_DIR "${COREBLAS_DIR}/lib/pkgconfig")
  endif(NOT COREBLAS_PKG_DIR)
endif(COREBLAS_DIR)

if(PLASMA_DIR)
  if(NOT COREBLAS_PKG_DIR)
    set(COREBLAS_PKG_DIR "${PLASMA_DIR}/lib/pkgconfig")
  endif(NOT COREBLAS_PKG_DIR)
endif(PLASMA_DIR)

# Optionally use pkg-config to detect include/library dirs (if pkg-config is available)
# -------------------------------------------------------------------------------------
STRING(REPLACE ":" ";" PATH_PKGCONFIGPATH "${COREBLAS_PKG_DIR}:$ENV{PKG_CONFIG_PATH}")
FIND_FILE(COREBLAS_PKG_FILE
    NAME  coreblas.pc
    PATHS ${PATH_PKGCONFIGPATH})
MARK_AS_ADVANCED(COREBLAS_PKG_FILE)

FIND_PACKAGE(PkgConfig QUIET)
IF(PKG_CONFIG_EXECUTABLE AND COREBLAS_PKG_FILE)
    pkg_search_module(COREBLAS coreblas)
ELSE()
    MESSAGE(STATUS "Looking for COREBLAS - pkgconfig not used")
ENDIF()

if(NOT COREBLAS_FOUND)
  #
  # No pkg-config supported on this system. Hope the user provided
  # all the required variables in order to detect COREBLAS. This include
  # either:
  # - COREBLAS_DIR: root dir to COREBLAS installation
  # - COREBLAS_PKG_DIR: directory where the coreblas.pc is installed
  #
  if(NOT PLASMA_DIR AND NOT COREBLAS_DIR AND NOT COREBLAS_PKG_DIR)
    if(COREBLAS_FIND_REQUIRED)
      message(FATAL_ERROR "pkg-config not available. You need to provide COREBLAS_DIR or COREBLAS_PKG_DIR")
    else(COREBLAS_FIND_REQUIRED)
      message(STATUS "pkg-config not available. You need to provide COREBLAS_DIR or COREBLAS_PKG_DIR")
    endif(COREBLAS_FIND_REQUIRED)

  else(NOT PLASMA_DIR AND NOT COREBLAS_DIR AND NOT COREBLAS_PKG_DIR)

    if (NOT COREBLAS_PKG_FILE)

      if(COREBLAS_FIND_REQUIRED)
        message(FATAL_ERROR "${COREBLAS_PKG_DIR}/coreblas.pc doesn't exist")
      else(COREBLAS_FIND_REQUIRED)
        message(STATUS "${COREBLAS_PKG_DIR}/coreblas.pc doesn't exist")
      endif(COREBLAS_FIND_REQUIRED)

    else(NOT COREBLAS_PKG_FILE)

      file(STRINGS "${COREBLAS_PKG_FILE}" _cflags REGEX "Cflags:")
      file(STRINGS "${COREBLAS_PKG_FILE}" _libs   REGEX "Libs:")

      string(REGEX REPLACE "Cflags:" "" _cflags ${_cflags})
      string(REGEX REPLACE "Libs:"   "" _libs   ${_libs}  )
      string(REGEX REPLACE " +" ";" _cflags ${_cflags})
      string(REGEX REPLACE " +" ";" _libs   ${_libs}  )

      foreach(_cflag ${_cflags})
        string(REGEX REPLACE "^-I(.*)" "\\1" _incdir "${_cflag}")
        if ("${_cflag}" MATCHES "-I.*")
          list(APPEND COREBLAS_INCLUDE_DIRS ${_incdir})
        else ("${_cflag}" MATCHES "-I.*")
          list(APPEND COREBLAS_CFLAGS ${_cflag})
        endif()
      endforeach()

      foreach(_lib ${_libs})
        string(REGEX REPLACE "^-L(.*)" "\\1" _libdir "${_lib}")
	string(REGEX REPLACE "^-Wl,--rpath=(.*)" "\\1" _libdir2 "${_lib}")
	if ("${_lib}" MATCHES "-Wl,--rpath=.*")
	  list(APPEND COREBLAS_LIBRARY_DIRS ${_libdir2})
	else("${_lib}" MATCHES "-Wl,--rpath=.*")
	  if ("${_lib}" MATCHES "-L.*")
            list(APPEND COREBLAS_LIBRARY_DIRS ${_libdir})
          else ("${_lib}" MATCHES "-L.*")
            string(REGEX REPLACE "^-l(.*)" "\\1" _onelib "${_lib}")
            if ("${_lib}" MATCHES "-l.*")
              list(APPEND COREBLAS_LIBRARIES ${_onelib})
            else ("${_lib}" MATCHES "-l.*")
              list(APPEND COREBLAS_LDFLAGS ${_lib})
            endif()
          else()
            list(APPEND COREBLAS_LDFLAGS ${_lib})
          endif()
	endif()
      endforeach()

      if(COREBLAS_INCLUDE_DIRS)
        list(REMOVE_DUPLICATES COREBLAS_INCLUDE_DIRS)
      endif()
      if(COREBLAS_CFLAGS)
        list(REMOVE_DUPLICATES COREBLAS_CFLAGS)
      endif()
      if(COREBLAS_LIBRARY_DIRS)
        list(REMOVE_DUPLICATES COREBLAS_LIBRARY_DIRS)
      endif()
      if(COREBLAS_LIBRARIES)
        list(REMOVE_DUPLICATES COREBLAS_LIBRARIES)
      endif()
      if(COREBLAS_LDFLAGS)
        list(REMOVE_DUPLICATES COREBLAS_LDFLAGS)
      endif()
    endif(NOT COREBLAS_PKG_FILE)
  endif(NOT PLASMA_DIR AND NOT COREBLAS_DIR AND NOT COREBLAS_PKG_DIR)

else(NOT COREBLAS_FOUND)

  set(_cflags ${COREBLAS_CFLAGS})
  set(COREBLAS_CFLAGS "")
  foreach(_cflag ${_cflags})
    string(REGEX REPLACE "^-I(.*)" "\\1" _incdir "${_cflag}")
    if ("${_cflag}" MATCHES "-I.*")
      list(APPEND COREBLAS_INCLUDE_DIRS ${_incdir})
    else ("${_cflag}" MATCHES "-I.*")
      list(APPEND COREBLAS_CFLAGS ${_cflag})
    endif()
  endforeach()

  set(_libs ${COREBLAS_LDFLAGS})
  set(COREBLAS_LDFLAGS "")
  foreach(_lib ${_libs})
    string(REGEX REPLACE "^-L(.*)" "\\1" _libdir "${_lib}")
    if ("${_lib}" MATCHES "-L.*")
      list(APPEND COREBLAS_LIBRARY_DIRS ${_libdir})
    else ("${_lib}" MATCHES "-L.*")
      string(REGEX REPLACE "^-l(.*)" "\\1" _onelib "${_lib}")
      if ("${_lib}" MATCHES "-l.*")
        list(APPEND COREBLAS_LIBRARIES ${_onelib})
      else ("${_lib}" MATCHES "-l.*")
        list(APPEND COREBLAS_LDFLAGS ${_lib})
      endif()
    else()
        list(APPEND COREBLAS_LDFLAGS ${_lib})
    endif()
  endforeach()

  list(REMOVE_DUPLICATES COREBLAS_INCLUDE_DIRS)
  list(REMOVE_DUPLICATES COREBLAS_LIBRARY_DIRS)
  list(REMOVE_DUPLICATES COREBLAS_LIBRARIES)

endif(NOT COREBLAS_FOUND)

# check that COREBLAS has been found
# ----------------------------------
include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(COREBLAS DEFAULT_MSG
                                  COREBLAS_INCLUDE_DIRS
                                  COREBLAS_LIBRARIES
                                  COREBLAS_LIBRARY_DIRS)

if(COREBLAS_FOUND)
  # Validate the include file <core_blas.h>
  include(CheckIncludeFile)
  set(COREBLAS_tmp_includes ${CMAKE_REQUIRED_INCLUDES})
  list(APPEND CMAKE_REQUIRED_INCLUDES ${COREBLAS_INCLUDE_DIRS})

  check_include_file(core_blas.h COREBLAS_COREBLAS_H_FOUND)

  if ( NOT COREBLAS_COREBLAS_H_FOUND )
    if(COREBLAS_FIND_REQUIRED)
      message(FATAL_ERROR "Couln't find the core_blas.h header in ${COREBLAS_INCLUDE_DIRS}")
    endif(COREBLAS_FIND_REQUIRED)
    set(COREBLAS_FOUND FALSE)
    return()
  endif()

  # Validate the library
  include(CheckCSourceCompiles)

  set(COREBLAS_tmp_libraries ${CMAKE_REQUIRED_LIBRARIES})
  set(COREBLAS_tmp_flags     ${CMAKE_REQUIRED_FLAGS})
  list(APPEND CMAKE_REQUIRED_LIBRARIES ${COREBLAS_LIBRARIES})

  # CMAKE_REQUIRED_FLAGS must be a string, not a list
  # if CMAKE_REQUIRED_FLAGS is a list (separated by ;), only the first element of the list is passed to check_c_source_compile
  # Since COREBLAS_LDFLAGS and COREBLAS_CFLAGS hold lists, we convert them by hand to a string
  foreach(arg ${COREBLAS_LDFLAGS})
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${arg}")
  endforeach(arg ${COREBLAS_LDFLAGS})
  foreach(arg ${COREBLAS_CFLAGS})
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${arg}")
  endforeach(arg ${COREBLAS_CFLAGS})
  foreach(arg ${COREBLAS_LIBRARY_DIRS})
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -L${arg}")
  endforeach(arg ${COREBLAS_CFLAGS})

  check_c_source_compiles(
    "int main(int argc, char* argv[]) {
       CORE_dpltmg(); return 0;
     }"
    COREBLAS_C_COMPILE_SUCCESS
    )

  if(NOT COREBLAS_C_COMPILE_SUCCESS)
    get_property(_LANGUAGES_ GLOBAL PROPERTY ENABLED_LANGUAGES)
    if(NOT _LANGUAGES_ MATCHES Fortran)
      if(COREBLAS_FIND_REQUIRED)
        message(FATAL_ERROR "Find COREBLAS requires Fortran support so Fortran must be enabled.")
      else(COREBLAS_FIND_REQUIRED)
        message(STATUS "Looking for COREBLAS... - NOT found (Fortran not enabled)") #
        set(COREBLAS_FOUND FALSE)
        return()
      endif(COREBLAS_FIND_REQUIRED)
    endif(NOT _LANGUAGES_ MATCHES Fortran)
    include(CheckFortranFunctionExists)
    list(APPEND CMAKE_REQUIRED_LIBRARIES ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
    check_c_source_compiles(
      "int main(int argc, char* argv[]) {
        CORE_dpltmg(); return 0;
     }"
      COREBLAS_F_COMPILE_SUCCESS
      )

    if(NOT COREBLAS_F_COMPILE_SUCCESS)
      if(COREBLAS_FIND_REQUIRED)
        message(FATAL_ERROR "Find COREBLAS requires Fortran support so Fortran must be enabled.")
      else(COREBLAS_FIND_REQUIRED)
        message(STATUS "Looking for COREBLAS... - NOT found")
        set(COREBLAS_FOUND FALSE)
        return()
      endif(COREBLAS_FIND_REQUIRED)
    endif(NOT COREBLAS_F_COMPILE_SUCCESS)
  endif(NOT COREBLAS_C_COMPILE_SUCCESS)

  set(${CMAKE_REQUIRED_INCLUDES}  COREBLAS_tmp_includes)
  set(${CMAKE_REQUIRED_LIBRARIES} COREBLAS_tmp_libraries)
  set(${CMAKE_REQUIRED_FLAGS}     COREBLAS_tmp_flags)
  unset(COREBLAS_tmp_libraries)
  unset(COREBLAS_tmp_includes)
  unset(COREBLAS_tmp_flags)
endif(COREBLAS_FOUND)

if(NOT COREBLAS_FIND_QUIETLY)
  set(COREBLAS_status_message
    "
    COREBLAS_INCLUDE_DIRS = [${COREBLAS_INCLUDE_DIRS}]
    COREBLAS_LIBRARY_DIRS = [${COREBLAS_LIBRARY_DIRS}]
    COREBLAS_LIBRARIES    = [${COREBLAS_LIBRARIES}]
    COREBLAS_LDFLAGS      = [${COREBLAS_LDFLAGS}]")

  if(COREBLAS_C_COMPILE_SUCCESS OR COREBLAS_F_COMPILE_SUCCESS)
    if(COREBLAS_F_COMPILE_SUCCESS)
      set(COREBLAS_REQUIRE_FORTRAN_LINKER TRUE)
      mark_as_advanced(COREBLAS_REQUIRE_FORTRAN_LINKER)
      message(STATUS "A Library with COREBLAS API found (using C compiler and Fortran linker).")
    endif(COREBLAS_F_COMPILE_SUCCESS)
    string(REGEX REPLACE ";" " " COREBLAS_LDFLAGS "${COREBLAS_LDFLAGS}")
    set(COREBLAS_FOUND TRUE)
    find_package_message(COREBLAS
      "Found COREBLAS: ${COREBLAS_status_message}"
      "[${COREBLAS_INCLUDE_DIRS}][${COREBLAS_LIBRARY_DIRS}]")
  else(COREBLAS_C_COMPILE_SUCCESS OR COREBLAS_F_COMPILE_SUCCESS)
    if(COREBLAS_FIND_REQUIRED)
      message(FATAL_ERROR
        "A required library with COREBLAS API not found. Please specify library location.${COREBLAS_status_message}")
    else(COREBLAS_FIND_REQUIRED)
      message(STATUS
        "A library with COREBLAS API not found. Please specify library location.${COREBLAS_status_message}")
    endif(COREBLAS_FIND_REQUIRED)
  endif(COREBLAS_C_COMPILE_SUCCESS OR COREBLAS_F_COMPILE_SUCCESS)
endif(NOT COREBLAS_FIND_QUIETLY)

set(COREBLAS_DIR          "${COREBLAS_DIR}"          CACHE PATH   "Location of the COREBLAS library" FORCE)
set(COREBLAS_PKG_DIR      "${COREBLAS_PKG_DIR}"      CACHE PATH   "Location of the COREBLAS pkg-config decription file" FORCE)

unset(COREBLAS_C_COMPILE_SUCCESS)
unset(COREBLAS_F_COMPILE_SUCCESS)
