# - Find PLASMA library
# This module finds an installed  lirary that implements the PLASMA
# linear-algebra interface (see http://icl.cs.utk.edu/plasma/).
# The list of libraries searched for is taken
# from the autoconf macro file, acx_blas.m4 (distributed at
# http://ac-archive.sourceforge.net/ac-archive/acx_blas.html).
#
# This module sets the following variables:
#  PLASMA_FOUND - set to true if a library implementing the PLASMA interface
#    is found
#  PLASMA_LINKER_FLAGS - uncached list of required linker flags (excluding -l
#    and -L).
#  PLASMA_LIBRARIES - uncached list of libraries (using full path name) to
#    link against to use PLASMA
#  PLASMA_STATIC  if set on this determines what kind of linkage we do (static)
#  PLASMA_VENDOR  if set checks only the specified vendor, if not set checks
#     all the possibilities
##########

get_property(_LANGUAGES_ GLOBAL PROPERTY ENABLED_LANGUAGES)
if(NOT _LANGUAGES_ MATCHES Fortran)
  if(PLASMA_FIND_REQUIRED)
    message(FATAL_ERROR "FindPLASMA requires Fortran support so Fortran must be enabled.")
  else(PLASMA_FIND_REQUIRED)
    message(STATUS "Looking for PLASMA... - NOT found (Fortran not enabled)") #
    return()
  endif(PLASMA_FIND_REQUIRED)
endif(NOT _LANGUAGES_ MATCHES Fortran)

#FIND_PACKAGE(BLAS REQUIRED)
FIND_PACKAGE(LAPACK REQUIRED)

include(CheckFortranFunctionExists)
include(CheckIncludeFiles)

if(PLASMA_DIR)
  set(PLASMA_INCLUDE_PATH "${PLASMA_DIR}/include;${PLASMA_DIR}/src")
  set(PLASMA_LIBRARY_PATH "${PLASMA_DIR}/lib")
endif(PLASMA_DIR)

set(CMAKE_REQUIRED_INCLUDES ${PLASMA_INCLUDE_PATH})
check_include_files(plasma.h FOUND_PLASMA_INCLUDE)
if(FOUND_PLASMA_INCLUDE)
  check_library_exists("plasma;coreblas" PLASMA_Init ${PLASMA_LIBRARY_PATH} FOUND_PLASMA_LIB)
  if( FOUND_PLASMA_LIB )
    set(PLASMA_LIBRARY "${PLASMA_LIBRARY_PATH}/libplasma.a;${PLASMA_LIBRARY_PATH}/libcoreblas.a")
  endif( FOUND_PLASMA_LIB )
endif(FOUND_PLASMA_INCLUDE)

if(FOUND_PLASMA_INCLUDE AND FOUND_PLASMA_LIB)
  set(PLASMA_FOUND TRUE)
else(FOUND_PLASMA_INCLUDE AND FOUND_PLASMA_LIB)
  set(PLASMA_FOUND FALSE)
endif(FOUND_PLASMA_INCLUDE AND FOUND_PLASMA_LIB)

if(NOT PLASMA_FIND_QUIETLY)
  if(PLASMA_FOUND)
    message(STATUS "A library with PLASMA API found.")
  else(PLASMA_FOUND)
    if(PLASMA_FIND_REQUIRED)
      message(FATAL_ERROR
        "A required library with PLASMA API not found. Please specify library location."
        )
    else(PLASMA_FIND_REQUIRED)
      message(STATUS
        "A library with PLASMA API not found. Please specify library location."
        )
    endif(PLASMA_FIND_REQUIRED)
  endif(PLASMA_FOUND)
endif(NOT PLASMA_FIND_QUIETLY)
