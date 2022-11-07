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
#  HWLOC_DEFINITIONS - uncached list of required compile flags
#
#  PARSEC_HAVE_HWLOC_PARENT_MEMBER - new API, older versions don't have it
#  PARSEC_HAVE_HWLOC_CACHE_ATTR - new API, older versions don't have it
#  PARSEC_HAVE_HWLOC_OBJ_PU - new API, older versions don't have it
#
#  hwloc interface library target
##########

include(CheckStructHasMember)
include(CMakePushCheckState)

mark_as_advanced(FORCE HWLOC_INCLUDE_DIR HWLOC_LIBRARY)

find_package(PkgConfig QUIET)
if( HWLOC_ROOT )
  # We don't use HWLOC_DIR as this is supposed to be used when finding hwloc-config.cmake only
  set(ENV{PKG_CONFIG_PATH} "${HWLOC_ROOT}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
endif()

pkg_check_modules(PC_HWLOC QUIET hwloc)
set(HWLOC_DEFINITIONS ${PC_HWLOC_CFLAGS_OTHER} )

find_path(HWLOC_INCLUDE_DIR hwloc.h
          HINTS ${PC_HWLOC_INCLUDEDIR} ${PC_HWLOC_INCLUDE_DIRS}
          PATH_SUFFIXES include
          DOC "HWLOC includes" )
set(HWLOC_INCLUDE_DIRS ${HWLOC_INCLUDE_DIR})

find_library(HWLOC_LIBRARY hwloc
             HINTS ${PC_HWLOC_LIBDIR} ${PC_HWLOC_LIBRARY_DIRS}
             PATH_SUFFIXES lib
             DOC "Where the HWLOC libraries are")
set(HWLOC_LIBRARIES ${HWLOC_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HWLOC
    "Could NOT find HWLOC; Options depending on HWLOC will be disabled"
    HWLOC_LIBRARY HWLOC_INCLUDE_DIR )

if(HWLOC_FOUND)
  # to try using HWLOC ensure that language C is enabled
  include(CheckLanguage)
  check_language(C)
  if(CMAKE_C_COMPILER)
    enable_language(C)
  else()
    message(FATAL_ERROR "HWLOC found (HWLOC_LIBRARY=${HWLOC_LIBRARY}) but cannot test it due to missing C language support; either enable_language(C) in your project or ensure that C compiler can be discovered")
  endif()

  cmake_push_check_state()
  list(APPEND CMAKE_REQUIRED_INCLUDES ${HWLOC_INCLUDE_DIR})
  check_struct_has_member( "struct hwloc_obj" parent hwloc.h PARSEC_HAVE_HWLOC_PARENT_MEMBER )
  check_struct_has_member( "struct hwloc_cache_attr_s" size hwloc.h PARSEC_HAVE_HWLOC_CACHE_ATTR )
  check_c_source_compiles( "#include <hwloc.h>
    int main(void) { hwloc_obj_t o; o->type = HWLOC_OBJ_PU; return 0;}" PARSEC_HAVE_HWLOC_OBJ_PU)
  check_library_exists(${HWLOC_LIBRARY} hwloc_bitmap_free "" PARSEC_HAVE_HWLOC_BITMAP)
  cmake_pop_check_state()

  #===============================================================================
  # Import Target ================================================================
  if(NOT TARGET hwloc)
    add_library(hwloc INTERFACE IMPORTED)
  endif(NOT TARGET hwloc)

  set_property(TARGET hwloc PROPERTY INTERFACE_LINK_LIBRARIES "${PC_HWLOC_LIB}")
  set_property(TARGET hwloc APPEND PROPERTY INTERFACE_LINK_LIBRARIES "${HWLOC_LIBRARY}")
  set_property(TARGET hwloc PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${HWLOC_INCLUDE_DIR}")
  set_property(TARGET hwloc PROPERTY INTERFACE_COMPILE_OPTIONS "${HWLOC_DEFINITIONS}")
  #===============================================================================

else(HWLOC_FOUND)
  unset(PARSEC_HAVE_HWLOC_PARENT_MEMBER CACHE)
  unset(PARSEC_HAVE_HWLOC_CACHE_ATTR CACHE)
  unset(PARSEC_HAVE_HWLOC_OBJ_PU CACHE)
  unset(PARSEC_HAVE_HWLOC_BITMAP CACHE)
endif(HWLOC_FOUND)
