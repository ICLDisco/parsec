# - Find the PAPI library
# This module finds an installed  lirary that implements the
# performance counter interface (PAPI) (see http://icl.cs.utk.edu/papi/).
#
# This module sets the following variables:
#  PAPI_FOUND - set to true if a library implementing the PAPI interface
#    is found
#  PAPI_LIBRARIES - uncached list of libraries (using full path name) to
#    link against to use PAPI
#  PAPI_INCLUDE_DIRS - directory where the PAPI header files are
#
#  PAPI::PAPI interface library target for linking
##########

find_package(PkgConfig QUIET)

if( PAPI_ROOT )
  set(ENV{PKG_CONFIG_PATH} "${PAPI_ROOT}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
endif()
pkg_check_modules(PC_PAPI QUIET papi)
set(PAPI_DEFINITIONS ${PC_PAPI_CFLAGS_OTHER} )

find_path(PAPI_INCLUDE_DIR papi.h
          PATHS ${PAPI_ROOT}/include ENV PAPI_INCLUDE_DIR
          HINTS ${PC_PAPI_INCLUDEDIR} ${PC_PAPI_INCLUDE_DIRS}
          DOC "Include path for PAPI")

find_library(PAPI_LIBRARY NAMES papi
             HINTS ${PC_PAPI_LIBDIR} ${PC_PAPI_LIBRARY_DIRS}
             DOC "Library path for PAPI")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PAPI DEFAULT_MSG
                                  PAPI_LIBRARY PAPI_INCLUDE_DIR )
if( PAPI_FOUND )
    message(STATUS "PAPI Library found at ${PAPI_INCLUDE_DIR} ${PAPI_LIBRARY}")

  #===============================================================================
  # Importing PAPI as a cmake target
    if(NOT TARGET PAPI::PAPI)
      add_library(PAPI::PAPI INTERFACE IMPORTED)
    endif()

    set_property(TARGET PAPI::PAPI APPEND PROPERTY INTERFACE_LINK_LIBRARIES "${PAPI_LIBRARY}")
    set_property(TARGET PAPI::PAPI PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${PAPI_INCLUDE_DIR}")
  #===============================================================================

  set(PAPI_LIBRARIES ${PAPI_LIBRARY} )
  set(PAPI_INCLUDE_DIRS ${PAPI_INCLUDE_DIR} )
  mark_as_advanced(FORCE PAPI_INCLUDE_DIR PAPI_LIBRARY)

endif( PAPI_FOUND )
