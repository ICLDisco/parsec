# - Find PAPI library
# This module finds an installed  lirary that implements the 
# performance counter interface (PAPI) (see http://icl.cs.utk.edu/papi/).
#
# This module sets the following variables:
#  PAPI_FOUND - set to true if a library implementing the PAPI interface
#    is found
#  PAPI_LINKER_FLAGS - uncached list of required linker flags (excluding -l
#    and -L).
#  PAPI_LIBRARIES - uncached list of libraries (using full path name) to
#    link against to use PAPI
#  PAPI_INCLUDE_DIR - directory where the PAPI header files are
#
##########

include(CheckIncludeFiles)

if(PAPI_DIR)
  set(PAPI_INCLUDE_PATH "${PAPI_DIR}/include")
  set(PAPI_LIBRARY_PATH "${PAPI_DIR}/lib")
endif(PAPI_DIR)

set(CMAKE_REQUIRED_INCLUDES ${PAPI_INCLUDE_PATH}) 
check_include_files(papi.h FOUND_PAPI_INCLUDE)
if(FOUND_PAPI_INCLUDE)
  check_library_exists("papi" PAPI_Init ${PAPI_LIBRARY_PATH} FOUND_PAPI_LIB)
  if( FOUND_PAPI_LIB )
    set(PAPI_LIBRARY "${PAPI_LIBRARY_PATH}/libpapi.a")
  endif( FOUND_PAPI_LIB )
endif(FOUND_PAPI_INCLUDE)

if(FOUND_PAPI_INCLUDE AND FOUND_PAPI_LIB)
  set(PAPI_FOUND TRUE)
else(FOUND_PAPI_INCLUDE AND FOUND_PAPI_LIB)
  set(PAPI_FOUND FALSE)
endif(FOUND_PAPI_INCLUDE AND FOUND_PAPI_LIB)

if(NOT PAPI_FIND_QUIETLY)
  if(PAPI_FOUND)
    message(STATUS "A library with PAPI API found.")
  else(PAPI_FOUND)
    if(PAPI_FIND_REQUIRED)
      message(FATAL_ERROR
        "A required library with PAPI API not found. Please specify library location"
        "using PAPI_DIR or a combination of PAPI_INCLUDE_PATH and PAPI_LIBRARY_PATH")
    else(PAPI_FIND_REQUIRED)
      message(STATUS
        "A required library with PAPI API not found. Please specify library location"
        "using PAPI_DIR or a combination of PAPI_INCLUDE_PATH and PAPI_LIBRARY_PATH")
    endif(PAPI_FIND_REQUIRED)
  endif(PAPI_FOUND)
endif(NOT PAPI_FIND_QUIETLY)
