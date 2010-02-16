# - Find HWLOC library
# This module finds an installed  library that implements the HWLOC
# linear-algebra interface (see http://www.open-mpi.org/projects/hwloc/).
#
# This module sets the following variables:
#  HWLOC_FOUND - set to true if a library implementing the PLASMA interface
#    is found
#  HWLOC_LINKER_FLAGS - uncached list of required linker flags (excluding -l
#    and -L).
#  HWLOC_LIBRARIES - uncached list of libraries (using full path name) to
#    link against to use PLASMA
#  HWLOC_STATIC  if set on this determines what kind of linkage we do (static)
##########

set(HWLOC_INCLUDE_DIR)
set(HWLOC_LIBRARIES)
set(HWLOC_LINKER_FLAGS)

include(CheckIncludeFile)

set(CMAKE_REQUIRED_INCLUDES "${CMAKE_REQUIRED_INCLUDES};${HWLOC_INCLUDE_DIR}")
#  message(STATUS "Looking for hwloc.h in ${HWLOC_INCLUDE_DIR}")
check_include_file(hwloc.h FOUND_HWLOC_INCLUDE)
if(FOUND_HWLOC_INCLUDE)
  #  message(STATUS "Found hwloc.h at ${HWLOC_INCLUDE_DIR}")
  find_library(HWLOC_LIB hwloc
    PATHS ${HWLOC_LIBRARIES}
    DOC "Where the HWLOC libraries are"
    NO_DEFAULT_PATH)
  if( NOT HWLOC_LIB )
    find_library(HWLOC_LIB hwloc
      PATHS ${HWLOC_LIBRARIES}
      DOC "Where the HWLOC cblas libraries are")
  endif( NOT HWLOC_LIB )
endif(FOUND_HWLOC_INCLUDE)
  
if(FOUND_HWLOC_INCLUDE AND HWLOC_LIB)
  set(HWLOC_FOUND TRUE)
else(FOUND_HWLOC_INCLUDE AND HWLOC_LIB)
  set(HWLOC_FOUND FALSE)
endif(FOUND_HWLOC_INCLUDE AND HWLOC_LIB)

include(FindPackageMessage)
find_package_message(HWLOC "Found HWLOC: ${HWLOC_LIBRARIES}"
  "[${HWLOC_INCLUDE_DIR}][${HWLOC_LIBRARIES}]")

if(NOT HWLOC_FIND_QUIETLY)
  if(HWLOC_FOUND)
    message(STATUS "A library with HWLOC API found.")
    set(HAVE_HWLOC 1)
  else(HWLOC_FOUND)
    if(HWLOC_FIND_REQUIRED)
      message(FATAL_ERROR
        "A required library with HWLOC API not found. Please specify library location."
        )
    else(HWLOC_FIND_REQUIRED)
      message(STATUS
        "A library with HWLOC API not found. Please specify library location."
        )
    endif(HWLOC_FIND_REQUIRED)
  endif(HWLOC_FOUND)
endif(NOT HWLOC_FIND_QUIETLY)
