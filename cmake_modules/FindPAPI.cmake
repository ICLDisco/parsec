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

# If we only have the main PLASMA directory componse the include and
# libraries path based on it.
if( PAPI_DIR )
  if( NOT PAPI_INCLUDE_DIR )
    set(PAPI_INCLUDE_DIR "${PAPI_DIR}/include")
  endif( NOT PAPI_INCLUDE_DIR )
  if( NOT PAPI_LIBRARIES )
    set(PAPI_LIBRARIES "${PAPI_DIR}/lib")
  endif( NOT PAPI_LIBRARIES )
endif( PAPI_DIR )

if( NOT PAPI_INCLUDE_DIR )
  set(PAPI_INCLUDE_DIR)
endif( NOT PAPI_INCLUDE_DIR )
if( NOT PAPI_LIBRARIES )
  set(PAPI_LIBRARIES)
endif( NOT PAPI_LIBRARIES )
if( NOT PAPI_LINKER_FLAGS )
  set(PAPI_LINKER_FLAGS)
endif( NOT PAPI_LINKER_FLAGS )

list(APPEND CMAKE_REQUIRED_INCLUDES ${PAPI_INCLUDE_DIR})
# message(STATUS "Looking for papi.h in ${CMAKE_REQUIRED_INCLUDES}")
check_include_file(papi.h FOUND_PAPI_INCLUDE)
if(FOUND_PAPI_INCLUDE)
  message(STATUS "PAPI include files found at ${PAPI_INCLUDE_DIR}")
  check_library_exists("papi" PAPI_library_init ${PAPI_LIBRARIES} FOUND_PAPI_LIB)
  if( FOUND_PAPI_LIB )
    set(PAPI_LIBRARY "${PAPI_LIBRARIES}/libpapi.a")
    set(PAPI_LIBRARIES "-L${PAPI_LIBRARIES} -lpapi")
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
    set(HAVE_PAPI 1)
  else(PAPI_FOUND)
    if(PAPI_FIND_REQUIRED)
      message(FATAL_ERROR
        "A required library with PAPI API not found. Please specify library location "
        "using PAPI_DIR or a combination of PAPI_INCLUDE_DIR and PAPI_LIBRARIES "
        "or by setting PAPI_DIR")
    else(PAPI_FIND_REQUIRED)
      message(STATUS
        "A required library with PAPI API not found. Please specify library location "
        "using PAPI_DIR or a combination of PAPI_INCLUDE_DIR and PAPI_LIBRARIES "
        "or by setting PAPI_DIR")
    endif(PAPI_FIND_REQUIRED)
  endif(PAPI_FOUND)
endif(NOT PAPI_FIND_QUIETLY)
