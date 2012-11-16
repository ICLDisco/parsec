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
#
#  HAVE_HWLOC_PARENT_MEMBER - new API, older versions don't have it
#  HAVE_HWLOC_CACHE_ATTR - new API, older versions don't have it
#  HAVE_HWLOC_OBJ_PU - new API, older versions don't have it
#
##########

# If we only have the main PLASMA directory componse the include and
# libraries path based on it.
if( HWLOC_DIR )
  if( NOT HWLOC_INCLUDE_DIR )
    set(HWLOC_INCLUDE_DIR "${HWLOC_DIR}/include")
  endif( NOT HWLOC_INCLUDE_DIR )
  if( NOT HWLOC_LIBRARY_DIR )
    set(HWLOC_LIBRARY_DIR "${HWLOC_DIR}/lib")
  endif( NOT HWLOC_LIBRARY_DIR )
endif( HWLOC_DIR )

if( NOT HWLOC_INCLUDE_DIR )
  set(HWLOC_INCLUDE_DIR)
endif( NOT HWLOC_INCLUDE_DIR )
if( NOT HWLOC_LIBRARY_DIR )
  set(HWLOC_LIBRARY_DIR)
endif( NOT HWLOC_LIBRARY_DIR )
if( NOT HWLOC_LINKER_FLAGS )
  set(HWLOC_LINKER_FLAGS)
endif( NOT HWLOC_LINKER_FLAGS )

include(CheckIncludeFile)

set(HWLOC_SAVE_CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES})
list(APPEND CMAKE_REQUIRED_INCLUDES ${HWLOC_INCLUDE_DIR})
check_include_file(hwloc.h FOUND_HWLOC_INCLUDE)
if(FOUND_HWLOC_INCLUDE)
  find_library(HWLOC_LIB hwloc
    PATHS ${HWLOC_LIBRARY_DIR}
    DOC "Where the HWLOC libraries are"
    NO_DEFAULT_PATH)
  if( NOT HWLOC_LIB )
    find_library(HWLOC_LIB hwloc
        PATHS ${HWLOC_LIBRARY_DIR}
      DOC "Where the HWLOC  libraries are")
  endif( NOT HWLOC_LIB )
endif(FOUND_HWLOC_INCLUDE)
  
if(FOUND_HWLOC_INCLUDE AND HWLOC_LIB)
  check_struct_has_member( "struct hwloc_obj" parent hwloc.h HAVE_HWLOC_PARENT_MEMBER )             
  check_struct_has_member( "struct hwloc_cache_attr_s" size hwloc.h HAVE_HWLOC_CACHE_ATTR )
  check_c_source_compiles( "#include <hwloc.h>
    int main(void) { hwloc_obj_t o; o->type = HWLOC_OBJ_PU; return 0;}" HAVE_HWLOC_OBJ_PU)
  check_library_exists(${HWLOC_LIB} hwloc_bitmap_free "" HAVE_HWLOC_BITMAP)
  set(HWLOC_FOUND TRUE)

else(FOUND_HWLOC_INCLUDE AND HWLOC_LIB)
  set(HWLOC_FOUND FALSE)
endif(FOUND_HWLOC_INCLUDE AND HWLOC_LIB)

if(NOT HWLOC_FIND_QUIETLY)
  if(HWLOC_FOUND)
    message(STATUS "A library with HWLOC API found.")
    include(FindPackageMessage)
    find_package_message(HWLOC "Found HWLOC: ${HWLOC_LIBRARY_DIR}"
      "[${HWLOC_INCLUDE_DIR}][${HWLOC_LIB}]")
#    include_directories( ${HWLOC_INCLUDE_DIR} )
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

set(CMAKE_REQUIRED_INCLUDES ${HWLOC_SAVE_CMAKE_REQUIRED_INCLUDES})
