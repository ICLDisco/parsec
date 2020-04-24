# - Find HWLOC library
# This module finds an installed library that implements the HWLOC
# interface (see http://www.open-mpi.org/projects/hwloc/).
#
# This module sets the following variables:
#  HWLOC_FOUND - set to true if a library implementing the HWLOC interface
#    is found
#  HWLOC_LINKER_FLAGS - uncached list of required linker flags (excluding -l
#    and -L).
#  HWLOC_LIBRARIES - uncached list of libraries (using full path name) to
#    link against to use HWLOC
#  HWLOC_INCLUDE_DIRS - uncached list of required include directories to
#    access HWLOC headers
#  HWLOC_STATIC  if set on this determines what kind of linkage we do (static)
#
#  PARSEC_HAVE_HWLOC_PARENT_MEMBER - new API, older versions don't have it
#  PARSEC_HAVE_HWLOC_CACHE_ATTR - new API, older versions don't have it
#  PARSEC_HAVE_HWLOC_OBJ_PU - new API, older versions don't have it
#
#  HWLOC::HWLOC interface library target
##########

include(CheckStructHasMember)

mark_as_advanced(FORCE HWLOC_DIR HWLOC_INCLUDE_DIR HWLOC_LIBRARY)

set(HWLOC_DIR "" CACHE PATH "Root directory containing HWLOC")

find_package(PkgConfig QUIET)
if( HWLOC_DIR )
  set(ENV{PKG_CONFIG_PATH} "${HWLOC_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
endif()
pkg_check_modules(PC_HWLOC QUIET hwloc)
set(HWLOC_DEFINITIONS ${PC_HWLOC_CFLAGS_OTHER} )

find_path(HWLOC_INCLUDE_DIR hwloc.h
          HINTS ${HWLOC_DIR} ${PC_HWLOC_INCLUDEDIR} ${PC_HWLOC_INCLUDE_DIRS}
          PATH_SUFFIXES include
          DOC "HWLOC includes" )
set(HWLOC_INCLUDE_DIRS ${HWLOC_INCLUDE_DIR})

find_library(HWLOC_LIBRARY hwloc
             HINTS ${HWLOC_DIR} ${PC_HWLOC_LIBDIR} ${PC_HWLOC_LIBRARY_DIRS}
             PATH_SUFFIXES lib
             DOC "Where the HWLOC libraries are")
set(HWLOC_LIBRARIES ${HWLOC_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HWLOC
    "Could NOT find HWLOC; Options depending on HWLOC will be disabled"
    HWLOC_LIBRARY HWLOC_INCLUDE_DIR )

if(HWLOC_FOUND)
  set(HWLOC_SAVE_CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES})
  list(APPEND CMAKE_REQUIRED_INCLUDES ${HWLOC_INCLUDE_DIR})
  check_struct_has_member( "struct hwloc_obj" parent hwloc.h PARSEC_HAVE_HWLOC_PARENT_MEMBER )
  check_struct_has_member( "struct hwloc_cache_attr_s" size hwloc.h PARSEC_HAVE_HWLOC_CACHE_ATTR )
  check_c_source_compiles( "#include <hwloc.h>
    int main(void) { hwloc_obj_t o; o->type = HWLOC_OBJ_PU; return 0;}" PARSEC_HAVE_HWLOC_OBJ_PU)
  check_library_exists(${HWLOC_LIBRARY} hwloc_bitmap_free "" PARSEC_HAVE_HWLOC_BITMAP)
  set(CMAKE_REQUIRED_INCLUDES ${HWLOC_SAVE_CMAKE_REQUIRED_INCLUDES})
  #===============================================================================
  # Import Target ================================================================
  if(NOT TARGET HWLOC::HWLOC)
    add_library(HWLOC::HWLOC INTERFACE IMPORTED)
  endif(NOT TARGET HWLOC::HWLOC)

  set_property(TARGET HWLOC::HWLOC PROPERTY INTERFACE_LINK_LIBRARIES "${PC_HWLOC_LIB}")
  set_property(TARGET HWLOC::HWLOC APPEND PROPERTY INTERFACE_LINK_LIBRARIES "${HWLOC_LIBRARY}")
  set_property(TARGET HWLOC::HWLOC PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${HWLOC_INCLUDE_DIR}")
  #===============================================================================
else(HWLOC_FOUND)
  unset(PARSEC_HAVE_HWLOC_PARENT_MEMBER CACHE)
  unset(PARSEC_HAVE_HWLOC_CACHE_ATTR CACHE)
  unset(PARSEC_HAVE_HWLOC_OBJ_PU CACHE)
  unset(PARSEC_HAVE_HWLOC_BITMAP CACHE)
endif(HWLOC_FOUND)
