# - Find PLASMA library
# This module finds an installed  library that implements the PLASMA
# linear-algebra interface (see http://icl.cs.utk.edu/plasma/).
# The list of libraries searched for is taken
# from the autoconf macro file, acx_blas.m4 (distributed at
# http://ac-archive.sourceforge.net/ac-archive/acx_blas.html).
#
# This module sets the following variables:
#  PLASMA_FOUND - set to true if a library implementing the PLASMA interface
#    is found
#  PLASMA_PKG_DIR - Directory where the PLASMA pkg file is stored
#  PLASMA_LIBRARIES - uncached list of libraries (using full path name) to
#    link against to use PLASMA
#  PLASMA_INCLUDE_DIRS - Directory where the PLASMA include files are located
#  PLASMA_LINKER_FLAGS - uncached list of required linker flags (excluding -l
#    and -L).
#  PLASMA_STATIC  if set on this determines what kind of linkage we do (static)
#  PLASMA_VENDOR  if set checks only the specified vendor, if not set checks
#     all the possibilities
##########

unset(PLASMA_C_COMPILE_SUCCESS)
unset(PLASMA_F_COMPILE_SUCCESS)

# First we try to use pkg-config to find what we're looking for
# in the directory specified by the PLASMA_DIR or PLASMA_PKG_DIR
include(FindPkgConfig)
if(PLASMA_DIR)
  if(NOT PLASMA_PKG_DIR)
    set(PLASMA_PKG_DIR "${PLASMA_DIR}/lib/pkgconfig")
  endif(NOT PLASMA_PKG_DIR)
endif(PLASMA_DIR)

set(ENV{PKG_CONFIG_PATH} "${PLASMA_PKG_DIR}:$ENV{PKG_CONFIG_PATH}")
pkg_search_module(PLASMA plasma)
if(PKG_CONFIG_FOUND)
  if(PLASMA_FOUND)
    # 
    # There is a circular dependency in PLASMA between the libplasma and libcoreblas.
    # Unfortunately, this cannot be handled by pkg-config (as it remove the duplicates)
    # so we have to add it by hand.
    # 
    list(APPEND PLASMA_LDFLAGS -lplasma)
    string(REGEX REPLACE ";" " " PLASMA_LDFLAGS "${PLASMA_LDFLAGS}")
  endif(PLASMA_FOUND)
endif(PKG_CONFIG_FOUND)

if(NOT PLASMA_FOUND)
  #
  # No pkg-config supported on this system. Hope the user provided
  # all the required variables in order to detect PLASMA. This include
  # either:
  # - PLASMA_INCLUDE_DIRS, PLASMA_LDFLAGS or PLASMA_LIBRARIES
  # - PLASMA_DIR and PLASMA_LIBRARIES
  #
  if(NOT PLASMA_INCLUDE_DIRS)
    if(NOT PLASMA_DIR)
      if(PLASMA_FIND_REQUIRED)
        message(FATAL_ERROR "pkg-config not available. You need to provide PLASMA_DIR and PLASMA_LIBRARIES")
      else(PLASMA_FIND_REQUIRED)
        message(STATUS "pkg-config not available. You need to provide PLASMA_DIR and PLASMA_LIBRARIES")
      endif(PLASMA_FIND_REQUIRED)
    endif(NOT PLASMA_DIR)
    set(PLASMA_INCLUDE_DIRS "${PLASMA_DIR}/include")
  endif(NOT PLASMA_INCLUDE_DIRS)
  if(NOT PLASMA_LDFLAGS)
    if(PLAMA_DIR)
      set(PLASMA_LDFLAGS "${PLASMA_DIR}/lib")
    else(PLAMA_DIR)
      if(PLASMA_FIND_REQUIRED)
        message(FATAL_ERROR "pkg-config not available. You need to provide PLASMA_DIR and PLASMA_LIBRARIES")
      else(PLASMA_FIND_REQUIRED)
        message(STATUS "pkg-config not available. You need to provide PLASMA_DIR and PLASMA_LIBRARIES")
      endif(PLASMA_FIND_REQUIRED)
    endif(PLAMA_DIR)
  endif(NOT PLASMA_LDFLAGS)
endif(NOT PLASMA_FOUND)

if(PLASMA_INCLUDE_DIRS AND (PLASMA_LDFLAGS OR PLASMA_LIBRARIES))
  set(PLASMA_SINCLUDE_DIRS ${PLASMA_INCLUDE_DIRS})
  # Validate the include file <plasma.h>
  find_path(PLASMA_INCLUDE_DIRS
    plasma.h
    PATHS ${PLASMA_SINCLUDE_DIRS}
    )
  if(NOT PLASMA_INCLUDE_DIRS)
    if(PLASMA_FIND_REQUIRED)
      message(FATAL_ERROR "Couln't find the plasma.h header in ${PLASMA_SINCLUDE_DIRS}")
    endif(PLASMA_FIND_REQUIRED)
  endif(NOT PLASMA_INCLUDE_DIRS)

  # Validate the library
  include(CheckCSourceCompiles)

  set(PLASMA_tmp_libraries ${CMAKE_REQUIRED_LIBRARIES})
  set(PLASMA_tmp_flags ${CMAKE_REQUIRED_FLAGS})
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
  endif(NOT PLASMA_C_COMPILE_SUCCESS)

  set(${CMAKE_REQUIRED_LIBRARIES} PLASMA_tmp_libraries)
  set(${CMAKE_REQUIRED_FLAGS} PLASMA_tmp_flags)
  unset(PLASMA_tmp_libraries)
  unset(PLASMA_tmp_includes)
  unset(PLASMA_tmp_flags)
endif(PLASMA_INCLUDE_DIRS AND (PLASMA_LDFLAGS OR PLASMA_LIBRARIES))

if(NOT PLASMA_FIND_QUIETLY)
  if(PLASMA_C_COMPILE_SUCCESS OR PLASMA_F_COMPILE_SUCCESS)
    if(PLASMA_F_COMPILE_SUCCESS)
      set(PLASMA_REQUIRE_FORTRAN_LINKER TRUE)
      mark_as_advanced(PLASMA_REQUIRE_FORTRAN_LINKER)
      message(STATUS "A Library with PLASMA API found (using C compiler and Fortran linker).")
    endif(PLASMA_F_COMPILE_SUCCESS)
    string(REGEX REPLACE ";" " " PLASMA_LDFLAGS "${PLASMA_LDFLAGS}")
    set(PLASMA_FOUND TRUE)
    find_package_message(PLASMA
      "Found PLASMA: ${PLASMA_LIBRARIES}
    PLASMA_CFLAGS       = [${PLASMA_CFLAGS}]
    PLASMA_LDFLAGS      = [${PLASMA_LDFLAGS}]
    PLASMA_INCLUDE_DIRS = [${PLASMA_INCLUDE_DIRS}]
    PLASMA_LIBRARY_DIRS = [${PLASMA_LIBRARY_DIRS}]"
      "[${PLASMA_CFLAGS}][${PLASMA_LDFLAGS}][${PLASMA_INCLUDE_DIRS}][${PLASMA_LIBRARY_DIRS}]")
  else(PLASMA_C_COMPILE_SUCCESS OR PLASMA_F_COMPILE_SUCCESS)
    if(PLASMA_FIND_REQUIRED)
      message(FATAL_ERROR
        "A required library with PLASMA API not found. Please specify library location.
    PLASMA_CFLAGS       = [${PLASMA_CFLAGS}]
    PLASMA_LDFLAGS      = [${PLASMA_LDFLAGS}]
    PLASMA_INCLUDE_DIRS = [${PLASMA_INCLUDE_DIRS}]
    PLASMA_LIBRARY_DIRS = [${PLASMA_LIBRARY_DIRS}]")
    else(PLASMA_FIND_REQUIRED)
      message(STATUS
        "A library with PLASMA API not found. Please specify library location.
    PLASMA_CFLAGS       = [${PLASMA_CFLAGS}]
    PLASMA_LDFLAGS      = [${PLASMA_LDFLAGS}]
    PLASMA_INCLUDE_DIRS = [${PLASMA_INCLUDE_DIRS}]
    PLASMA_LIBRARY_DIRS = [${PLASMA_LIBRARY_DIRS}]")
    endif(PLASMA_FIND_REQUIRED)
  endif(PLASMA_C_COMPILE_SUCCESS OR PLASMA_F_COMPILE_SUCCESS)
endif(NOT PLASMA_FIND_QUIETLY)

mark_as_advanced(PLASMA_PKG_DIR PLASMA_LIBRARIES PLASMA_INCLUDE_DIRS PLASMA_LINKER_FLAGS)
set(PLASMA_DIR "${PLASMA_DIR}" CACHE PATH "Location of the PLASMA library" FORCE)
set(PLASMA_PKG_DIR "${PLASMA_PKG_DIR}" CACHE PATH "Location of the PLASMA pkg-config decription file" FORCE)
set(PLASMA_LINKER_FLAGS "${PLASMA_LINKER_FLAGS}" CACHE STRING "Linker flags to build with PLASMA" FORCE)
set(PLASMA_INCLUDE_DIRS "${PLASMA_INCLUDE_DIRS}" CACHE PATH "PLASMA include directories" FORCE)
set(PLASMA_LIBRARIES "${PLASMA_LIBRARIES}" CACHE STRING "libraries to link with PLASMA" FORCE)

unset(PLASMA_C_COMPILE_SUCCESS)
unset(PLASMA_F_COMPILE_SUCCESS)
