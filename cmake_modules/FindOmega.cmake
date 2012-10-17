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
#  OMEGA_LIBRARIES - uncached list of libraries (using full path name) to
#    link against to use OMEGA
#  OMEGA_INCLUDE_DIRS - uncached directories where the OMEGA header files are
#
##########

mark_as_advanced(OMEGA_INCLUDE_DIR OMEGA_SRC_INCLUDE_DIR OMEGA_DIR OMEGA_LIBRARY)

find_path(OMEGA_INCLUDE_DIR omega.h PATHS "${OMEGA_DIR}" PATH_SUFFIXES include/omega omega_lib/include) 
find_path(OMEGA_SRC_INCLUDE_DIR basic/bool.h PATHS "${OMEGA_DIR}" PATH_SUFFIXES basic/include) 
if(OMEGA_SRC_INCLUDE_DIR) 
  set(OMEGA_INCLUDE_DIRS ${OMEGA_INCLUDE_DIR} ${OMEGA_SRC_INCLUDE_DIR})
else()
  set(OMEGA_INCLUDE_DIRS ${OMEGA_INCLUDE_DIR})
endif()

find_library(OMEGA_LIBRARY omega
             PATHS "${OMEGA_DIR}"
             PATH_SUFFIXES lib omega_lib/obj
             DOC "Where the Omega  libraries are")
set(OMEGA_LIBRARIES ${OMEGA_LIBRARY})

include(CheckCXXSourceCompiles)
list(APPEND CMAKE_REQUIRED_INCLUDES ${OMEGA_INCLUDE_DIRS})
list(APPEND CMAKE_REQUIRED_LIBRARIES ${OMEGA_LIBRARIES})
check_cxx_source_compiles("#include <omega.h>
    int main(void) { Relation R; R.is_set(); return 0;}" OMEGA_FOUND)

if(OMEGA_FOUND)
  if(NOT OMEGA_FIND_QUIETLY)
    message(STATUS "A library with OMEGA API found: [${OMEGA_LIBRARIES}], [${OMEGA_INCLUDE_DIRS}].")
  endif(NOT OMEGA_FIND_QUIETLY)
  set(HAVE_OMEGA 1)
else(OMEGA_FOUND)
  if(OMEGA_FIND_REQUIRED)
    message(FATAL_ERROR
        "An required library with OMEGA API not found.\n"
        "    You need to specify the library location\n"
        "    - using OMEGA_DIR [${OMEGA_DIR}]\n"
        "    - or a combination of OMEGA_INCLUDE_DIR [${OMEGA_INCLUDE_DIR}] and OMEGA_LIBRARY [${OMEGA_LIBRARY}]\n")
  else(OMEGA_FIND_REQUIRED)
    if(NOT OMEGA_FIND_QUIETLY)
      message(STATUS
        "An optional library with OMEGA API not found.\n"
        "    To prevent options depending on Omega from being disabled, please specify the library location\n"
        "    - using OMEGA_DIR [${OMEGA_DIR}]\n"
        "    - or a combination of OMEGA_INCLUDE_DIR [${OMEGA_INCLUDE_DIR}] and OMEGA_LIBRARY [${OMEGA_LIBRARY}]\n")
    endif(NOT OMEGA_FIND_QUIETLY)
  endif(OMEGA_FIND_REQUIRED)
endif(OMEGA_FOUND)

