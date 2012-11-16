# - Find the GTG library
# This module finds an installed  lirary that implements the 
# Generic Trace Generator (GTG) (see https://gforge.inria.fr/projects/gtg/).
#
# This module sets the following variables:
#  GTG_FOUND - set to true if a library implementing the GTG interface
#    is found
#  GTG_LINKER_FLAGS - uncached list of required linker flags (excluding -l
#    and -L).
#  GTG_LIBRARIES - uncached list of libraries (using full path name) to
#    link against to use GTG
#  GTG_INCLUDE_DIR - directory where the GTG header files are
#
##########

include(CheckIncludeFiles)

if( GTG_DIR )
  if( NOT GTG_INCLUDE_DIR )
    set(GTG_INCLUDE_DIR "${GTG_DIR}/include")
  endif( NOT GTG_INCLUDE_DIR )
  if( NOT GTG_LIBRARIES )
    set(GTG_LIBRARIES "${GTG_DIR}/lib")
  endif( NOT GTG_LIBRARIES )
endif( GTG_DIR )

if( NOT GTG_INCLUDE_DIR )
  set(GTG_INCLUDE_DIR)
endif( NOT GTG_INCLUDE_DIR )
if( NOT GTG_LIBRARIES )
  set(GTG_LIBRARIES)
endif( NOT GTG_LIBRARIES )
if( NOT GTG_LINKER_FLAGS )
  set(GTG_LINKER_FLAGS)
endif( NOT GTG_LINKER_FLAGS )

list(APPEND CMAKE_REQUIRED_INCLUDES ${GTG_INCLUDE_DIR})
check_include_file(GTG.h FOUND_GTG_INCLUDE)
if(FOUND_GTG_INCLUDE)
  message(STATUS "GTG include files found at ${GTG_INCLUDE_DIR}")
  check_library_exists("gtg" setTraceType ${GTG_LIBRARIES} FOUND_GTG_LIB)
  if( FOUND_GTG_LIB )
    set(GTG_LIBRARY "${GTG_LIBRARIES}/libgtg.a")
    set(GTG_LIBRARIES "-L${GTG_LIBRARIES} -lgtg")
  endif( FOUND_GTG_LIB )
endif(FOUND_GTG_INCLUDE)

if(FOUND_GTG_INCLUDE AND FOUND_GTG_LIB)
  set(GTG_FOUND TRUE)
else(FOUND_GTG_INCLUDE AND FOUND_GTG_LIB)
  set(GTG_FOUND FALSE)
endif(FOUND_GTG_INCLUDE AND FOUND_GTG_LIB)

if(NOT GTG_FIND_QUIETLY)
  if(GTG_FOUND)
    message(STATUS "A library with GTG API found.")
    set(HAVE_GTG 1)
  else(GTG_FOUND)
    if(GTG_FIND_REQUIRED)
      message(FATAL_ERROR
        "A required library with GTG API not found. Please specify library location "
        "using GTG_DIR or a combination of GTG_INCLUDE_DIR and GTG_LIBRARIES "
        "or by setting GTG_DIR")
    else(GTG_FIND_REQUIRED)
      message(STATUS
        "A required library with GTG API not found. Please specify library location "
        "using GTG_DIR or a combination of GTG_INCLUDE_DIR and GTG_LIBRARIES "
        "or by setting GTG_DIR")
    endif(GTG_FIND_REQUIRED)
  endif(GTG_FOUND)
endif(NOT GTG_FIND_QUIETLY)
