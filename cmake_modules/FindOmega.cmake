#/*
# * Copyright (c) 2012      The University of Tennessee and The University
# *                         of Tennessee Research Foundation.  All rights
# *                         reserved.
# * Authors: Aurelien Bouteiller,
# */

# - Find the Omega library
# This module finds an installed  lirary that implements the 
# Omega polyhydral analysis tool (see http://www.cs.umd.edu/projects/omega/).
#
# This module sets the following variables:
#  OMEGA_FOUND - set to true if a library implementing the interface
#    is found
#  OMEGA_LINKER_FLAGS - uncached list of required linker flags (excluding -l
#    and -L).
#  OMEGA_LIBRARIES - uncached list of libraries (using full path name) to
#    link against to use OMEGA
#  OMEGA_INCLUDE_DIR - directory where the OMEGA header files are
#
##########


if( OMEGA_DIR )
  if( NOT OMEGA_INCLUDE_DIR )
    set(OMEGA_INCLUDE_DIR ${OMEGA_DIR}/omega_lib/include ${OMEGA_DIR}/basic/include)
  endif( NOT OMEGA_INCLUDE_DIR )
  if( NOT OMEGA_LIBRARIES )
    set(OMEGA_LIBRARIES "${OMEGA_DIR}/omega_lib/obj")
  endif( NOT OMEGA_LIBRARIES )
endif( OMEGA_DIR )

if( NOT OMEGA_INCLUDE_DIR )
  set(OMEGA_INCLUDE_DIR)
endif( NOT OMEGA_INCLUDE_DIR )
if( NOT OMEGA_LIBRARIES )
  set(OMEGA_LIBRARIES)
endif( NOT OMEGA_LIBRARIES )
if( NOT OMEGA_LINKER_FLAGS )
  set(OMEGA_LINKER_FLAGS)
endif( NOT OMEGA_LINKER_FLAGS )

include(CheckIncludeFileCXX)
list(APPEND CMAKE_REQUIRED_INCLUDES ${OMEGA_INCLUDE_DIR})
message(STATUS "Looking for omega.h in ${CMAKE_REQUIRED_INCLUDES}")
message(STATUS "OMEGA include files found at ${OMEGA_INCLUDE_DIR}")
find_library(OMEGA_LIB omega
            PATHS ${OMEGA_LIBRARIES}
            DOC "Where the Omega  libraries are")
check_library_exists(${OMEGA_LIB} omega:initializeProblem ${OMEGA_LIBRARIES} FOUND_OMEGA_LIB)
  if( FOUND_OMEGA_LIB )
    set(OMEGA_LIBRARY "${OMEGA_LIBRARIES}/libomega.a")
    set(OMEGA_LIBRARIES "-L${OMEGA_LIBRARIES} -lomega")
  endif( FOUND_OMEGA_LIB )

if(FOUND_OMEGA_INCLUDE AND FOUND_OMEGA_LIB)
  set(OMEGA_FOUND TRUE)
else(FOUND_OMEGA_INCLUDE AND FOUND_OMEGA_LIB)
  set(OMEGA_FOUND FALSE)
endif(FOUND_OMEGA_INCLUDE AND FOUND_OMEGA_LIB)

if(NOT OMEGA_FIND_QUIETLY)
  if(OMEGA_FOUND)
    message(STATUS "A library with OMEGA API found.")
    set(HAVE_OMEGA 1)
  else(OMEGA_FOUND)
    if(OMEGA_FIND_REQUIRED)
      message(FATAL_ERROR
        "A required library with OMEGA API not found. Please specify library location "
        "using OMEGA_DIR or a combination of OMEGA_INCLUDE_DIR and OMEGA_LIBRARIES "
        "or by setting OMEGA_DIR")
    else(OMEGA_FIND_REQUIRED)
      message(STATUS
        "A required library with OMEGA API not found. Please specify library location "
        "using OMEGA_DIR or a combination of OMEGA_INCLUDE_DIR and OMEGA_LIBRARIES "
        "or by setting OMEGA_DIR")
    endif(OMEGA_FIND_REQUIRED)
  endif(OMEGA_FOUND)
endif(NOT OMEGA_FIND_QUIETLY)
