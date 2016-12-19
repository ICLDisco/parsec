# - Find PLASMA library
# This module finds an installed  library that implements the PLASMA
# linear-algebra interface (see http://icl.cs.utk.edu/plasma/).
# The list of libraries searched for is taken
# from the autoconf macro file, acx_blas.m4 (distributed at
# http://ac-archive.sourceforge.net/ac-archive/acx_blas.html).
#
# This module sets the following variables:
#  PLASMA_FOUND - set to true if a library implementing the PLASMA
#    interface is found
#  PLASMA_PKG_DIR      - Directory where the PLASMA pkg file is stored
#  PLASMA_LIBRARIES    - only the libraries (w/o the '-l')
#  PLASMA_LIBRARY_DIRS - the paths of the libraries (w/o the '-L')
#  PLASMA_LDFLAGS      - all required linker flags
#  PLASMA_INCLUDE_DIRS - the '-I' preprocessor flags (w/o the '-I')
#  PLASMA_CFLAGS       - all required cflags
#
##########

unset(PLASMA_C_COMPILE_SUCCESS)
unset(PLASMA_F_COMPILE_SUCCESS)

# if (PLASMA_FOUND)
#   exit()
# endif()

# First we try to use pkg-config to find what we're looking for
# in the directory specified by the PLASMA_DIR or PLASMA_PKG_DIR
if(PLASMA_DIR)
  if(NOT PLASMA_PKG_DIR)
    set(PLASMA_PKG_DIR "${PLASMA_DIR}/lib/pkgconfig")
  endif(NOT PLASMA_PKG_DIR)
endif(PLASMA_DIR)

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  set(ENV{PKG_CONFIG_PATH} "${PLASMA_PKG_DIR}:$ENV{PKG_CONFIG_PATH}")
  pkg_check_modules(PLASMA plasma)
endif(PKG_CONFIG_FOUND)

if(NOT PLASMA_FOUND)
  #
  # No pkg-config supported on this system. Hope the user provided
  # all the required variables in order to detect PLASMA. This include
  # either:
  # - PLASMA_DIR: root dir to PLASMA installation
  # - PLASMA_PKG_DIR: directory where the plasma.pc is installed
  #
  if(NOT PLASMA_DIR AND NOT PLASMA_PKG_DIR)
    if(PLASMA_FIND_REQUIRED)
      message(FATAL_ERROR "pkg-config not available. You need to provide PLASMA_DIR or PLASMA_PKG_DIR")
    else(PLASMA_FIND_REQUIRED)
      message(STATUS "pkg-config not available. You need to provide PLASMA_DIR or PLASMA_PKG_DIR")
    endif(PLASMA_FIND_REQUIRED)

  else(NOT PLASMA_DIR AND NOT PLASMA_PKG_DIR)

    if (PLASMA_PKG_DIR)
      set(_plasma_pkg_file "${PLASMA_PKG_DIR}/plasma.pc")
    else()
      set(_plasma_pkg_file "${PLASMA_DIR}/lib/pkgconfig/plasma.pc")
    endif()

    if (NOT EXISTS ${_plasma_pkg_file})
      if(PLASMA_FIND_REQUIRED)
        message(FATAL_ERROR "${_plasma_pkg_file} doesn't exist")
      else(PLASMA_FIND_REQUIRED)
        message(STATUS "${_plasma_pkg_file} doesn't exist")
      endif(PLASMA_FIND_REQUIRED)
    else(NOT EXISTS ${_plasma_pkg_file})
      file(STRINGS "${_plasma_pkg_file}" _cflags REGEX "Cflags:")
      file(STRINGS "${_plasma_pkg_file}" _libs   REGEX "Libs:")

      string(REGEX REPLACE "Cflags:" "" _cflags ${_cflags})
      string(REGEX REPLACE "Libs:"   "" _libs   ${_libs}  )
      string(REGEX REPLACE " +" ";" _cflags ${_cflags})
      string(REGEX REPLACE " +" ";" _libs   ${_libs}  )

      foreach(_cflag ${_cflags})
        string(REGEX REPLACE "^-I(.*)" "\\1" _incdir "${_cflag}")
        if ("${_cflag}" MATCHES "-I.*")
          list(APPEND PLASMA_INCLUDE_DIRS ${_incdir})
        else ("${_cflag}" MATCHES "-I.*")
          list(APPEND PLASMA_CFLAGS ${_cflag})
        endif()
      endforeach()

      foreach(_lib ${_libs})
        string(REGEX REPLACE "^-L(.*)" "\\1" _libdir "${_lib}")
        if ("${_lib}" MATCHES "-L.*")
          list(APPEND PLASMA_LIBRARY_DIRS ${_libdir})
        else ("${_lib}" MATCHES "-L.*")
          string(REGEX REPLACE "^-l(.*)" "\\1" _onelib "${_lib}")
          if ("${_lib}" MATCHES "-l.*")
            list(APPEND PLASMA_LIBRARIES ${_onelib})
          else ("${_lib}" MATCHES "-l.*")
            list(APPEND PLASMA_LDFLAGS ${_lib})
          endif()
        endif()
      endforeach()

      if(PLASMA_INCLUDE_DIRS)	
	list(REMOVE_DUPLICATES PLASMA_INCLUDE_DIRS)
      endif()
      if(PLASMA_CFLAGS)	
	list(REMOVE_DUPLICATES PLASMA_CFLAGS)
      endif()
      if(PLASMA_LIBRARY_DIRS)
	list(REMOVE_DUPLICATES PLASMA_LIBRARY_DIRS)
      endif()
      if(PLASMA_LIBRARIES)
	list(REMOVE_DUPLICATES PLASMA_LIBRARIES)
      endif()
      if(PLASMA_LDFLAGS)
	list(REMOVE_DUPLICATES PLASMA_LDFLAGS)
      endif()

    endif(NOT EXISTS ${_plasma_pkg_file})
  endif(NOT PLASMA_DIR AND NOT PLASMA_PKG_DIR)
  set(PLASMA_FOUND TRUE)
else(NOT PLASMA_FOUND)
  set(_cflags ${PLASMA_CFLAGS})
  set(PLASMA_CFLAGS "")
  foreach(_cflag ${_cflags})
    string(REGEX REPLACE "^-I(.*)" "\\1" _incdir "${_cflag}")
    if ("${_cflag}" MATCHES "-I.*")
      list(APPEND PLASMA_INCLUDE_DIRS ${_incdir})
    else ("${_cflag}" MATCHES "-I.*")
      list(APPEND PLASMA_CFLAGS ${_cflag})
    endif()
  endforeach()

  set(_libs ${PLASMA_LDFLAGS})
  set(PLASMA_LDFLAGS "")
  foreach(_lib ${_libs})
    string(REGEX REPLACE "^-L(.*)" "\\1" _libdir "${_lib}")
    if ("${_lib}" MATCHES "-L.*")
      list(APPEND PLASMA_LIBRARY_DIRS ${_libdir})
    else ("${_lib}" MATCHES "-L.*")
      string(REGEX REPLACE "^-l(.*)" "\\1" _onelib "${_lib}")
      if ("${_lib}" MATCHES "-l.*")
        list(APPEND PLASMA_LIBRARIES ${_onelib})
      else ("${_lib}" MATCHES "-l.*")
        list(APPEND PLASMA_LDFLAGS ${_lib})
      endif()
    endif()
  endforeach()

  list(REMOVE_DUPLICATES PLASMA_INCLUDE_DIRS)
  list(REMOVE_DUPLICATES PLASMA_LIBRARY_DIRS)
  list(REMOVE_DUPLICATES PLASMA_LIBRARIES)

endif(NOT PLASMA_FOUND)

if(PLASMA_FOUND)

  #
  # There is a circular dependency in PLASMA between the libplasma and libcoreblas.
  # Unfortunately, this cannot be handled by pkg-config (as it remove the duplicates)
  # so we have to add it by hand.
  # Those parameters are also removed by pkg-config if they are present around mkl libs
  #
  # Check if the linker has group. MacOS doesn't support it
  include(CheckCSourceRuns)
  CMAKE_PUSH_CHECK_STATE()
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -Wl,--start-group -Wl,--end-group")
  check_c_source_runs( "int main() { return 0; }" PARSEC_HAVE_LINKER_GROUP )
  CMAKE_POP_CHECK_STATE()

  if(PARSEC_HAVE_LINKER_GROUP)
    list(INSERT PLASMA_LIBRARIES 0 -Wl,--start-group)
    list(APPEND PLASMA_LIBRARIES -Wl,--end-group)
  else()
    list(APPEND PLASMA_LIBRARIES plasma)
  endif()

  # Validate the include file <plasma.h>
  include(CheckIncludeFile)

  set(PLASMA_tmp_includes ${CMAKE_REQUIRED_INCLUDES})
  list(APPEND CMAKE_REQUIRED_INCLUDES ${PLASMA_INCLUDE_DIRS})

  check_include_file(plasma.h PLASMA_PLASMA_H_FOUND)

  if ( NOT PLASMA_PLASMA_H_FOUND )
    if(PLASMA_FIND_REQUIRED)
      message(FATAL_ERROR "Couln't find the plasma.h header in ${PLASMA_INCLUDE_DIRS}")
    endif(PLASMA_FIND_REQUIRED)
    set(PLASMA_FOUND FALSE)
    return()
  endif()

  # Validate the library
  include(CheckCSourceCompiles)

  set(PLASMA_tmp_libraries ${CMAKE_REQUIRED_LIBRARIES})
  set(PLASMA_tmp_flags     ${CMAKE_REQUIRED_FLAGS})
  list(APPEND CMAKE_REQUIRED_LIBRARIES ${PLASMA_LIBRARIES})

  # CMAKE_REQUIRED_FLAGS must be a string, not a list
  # if CMAKE_REQUIRED_FLAGS is a list (separated by ;), only the first element of the list is passed to check_c_source_compile
  # Since PLASMA_LDFLAGS and PLASMA_CFLAGS hold lists, we convert them by hand to a string
  foreach(arg ${PLASMA_LDFLAGS})
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${arg}")
  endforeach(arg ${PLASMA_LDFLAGS})
  foreach(arg ${PLASMA_CFLAGS})
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${arg}")
  endforeach(arg ${PLASMA_CFLAGS})
  foreach(arg ${PLASMA_LIBRARY_DIRS})
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -L${arg}")
  endforeach(arg ${PLASMA_CFLAGS})

  check_c_source_compiles(
    "int main(int argc, char* argv[]) {
       PLASMA_dgeqrf(); return 0;
     }"
    PLASMA_C_COMPILE_SUCCESS
    )

  if(NOT PLASMA_C_COMPILE_SUCCESS)
    get_property(_LANGUAGES_ GLOBAL PROPERTY ENABLED_LANGUAGES)
    if(NOT _LANGUAGES_ MATCHES Fortran)
      if(PLASMA_FIND_REQUIRED)
        message(FATAL_ERROR "Find PLASMA requires Fortran support so Fortran must be enabled.")
      else(PLASMA_FIND_REQUIRED)
        message(STATUS "Looking for PLASMA... - NOT found (Fortran not enabled)") #
        set(PLASMA_FOUND FALSE)
        return()
      endif(PLASMA_FIND_REQUIRED)
    endif(NOT _LANGUAGES_ MATCHES Fortran)
    include(CheckFortranFunctionExists)
    list(APPEND CMAKE_REQUIRED_LIBRARIES ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
    check_c_source_compiles(
      "int main(int argc, char* argv[]) {
       PLASMA_dgeqrf(); return 0;
     }"
      PLASMA_F_COMPILE_SUCCESS
      )

    if(NOT PLASMA_F_COMPILE_SUCCESS)
      if(PLASMA_FIND_REQUIRED)
        message(FATAL_ERROR "Find PLASMA requires Fortran support so Fortran must be enabled.")
      else(PLASMA_FIND_REQUIRED)
        message(STATUS "Looking for PLASMA... - NOT found")
        set(PLASMA_FOUND FALSE)
        return()
      endif(PLASMA_FIND_REQUIRED)
    endif(NOT PLASMA_F_COMPILE_SUCCESS)
  endif(NOT PLASMA_C_COMPILE_SUCCESS)

  set(${CMAKE_REQUIRED_INCLUDES}  PLASMA_tmp_includes)
  set(${CMAKE_REQUIRED_LIBRARIES} PLASMA_tmp_libraries)
  set(${CMAKE_REQUIRED_FLAGS}     PLASMA_tmp_flags)
  unset(PLASMA_tmp_libraries)
  unset(PLASMA_tmp_includes)
  unset(PLASMA_tmp_flags)
endif(PLASMA_FOUND)

if(NOT PLASMA_FIND_QUIETLY)
  set(PLASMA_status_message
    "
    PLASMA_CFLAGS       = [${PLASMA_CFLAGS}]
    PLASMA_LDFLAGS      = [${PLASMA_LDFLAGS}]
    PLASMA_INCLUDE_DIRS = [${PLASMA_INCLUDE_DIRS}]
    PLASMA_LIBRARY_DIRS = [${PLASMA_LIBRARY_DIRS}]
    PLASMA_LIBRARIES = [${PLASMA_LIBRARIES}]")

  if(PLASMA_C_COMPILE_SUCCESS OR PLASMA_F_COMPILE_SUCCESS)
    if(PLASMA_F_COMPILE_SUCCESS)
      set(PLASMA_REQUIRE_FORTRAN_LINKER TRUE)
      mark_as_advanced(PLASMA_REQUIRE_FORTRAN_LINKER)
      message(STATUS "A Library with PLASMA API found (using C compiler and Fortran linker).")
    endif(PLASMA_F_COMPILE_SUCCESS)
    string(REGEX REPLACE ";" " " PLASMA_LDFLAGS "${PLASMA_LDFLAGS}")
    set(PLASMA_FOUND TRUE)
    find_package_message(PLASMA
      "Found PLASMA: ${PLASMA_status_message}"
      "[${PLASMA_CFLAGS}][${PLASMA_LDFLAGS}][${PLASMA_INCLUDE_DIRS}][${PLASMA_LIBRARY_DIRS}]")
  else(PLASMA_C_COMPILE_SUCCESS OR PLASMA_F_COMPILE_SUCCESS)
    if(PLASMA_FIND_REQUIRED)
      message(FATAL_ERROR
        "A required library with PLASMA API not found. Please specify library location.${PLASMA_status_message}")
    else(PLASMA_FIND_REQUIRED)
      message(STATUS
        "A library with PLASMA API not found. Options depending on PLASMA will be disabled.")
    endif(PLASMA_FIND_REQUIRED)
  endif(PLASMA_C_COMPILE_SUCCESS OR PLASMA_F_COMPILE_SUCCESS)
endif(NOT PLASMA_FIND_QUIETLY)

mark_as_advanced(PLASMA_PKG_DIR PLASMA_LIBRARIES PLASMA_INCLUDE_DIRS PLASMA_LINKER_FLAGS)
set(PLASMA_DIR          "${PLASMA_DIR}"          CACHE PATH   "Location of the PLASMA library" FORCE)
set(PLASMA_PKG_DIR      "${PLASMA_PKG_DIR}"      CACHE PATH   "Location of the PLASMA pkg-config decription file" FORCE)
#set(PLASMA_LIBRARIES "${PLASMA_LIBRARIES}" CACHE STRING "libraries to link with PLASMA" FORCE)

unset(PLASMA_C_COMPILE_SUCCESS)
unset(PLASMA_F_COMPILE_SUCCESS)
