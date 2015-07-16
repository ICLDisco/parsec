# - Find the PAPI library
# This module finds an installed  lirary that implements the 
# performance counter interface (PAPI) (see http://icl.cs.utk.edu/papi/).
#
# This module sets the following variables:
#  PAPI_FOUND - set to true if a library implementing the PAPI interface
#    is found
#  PAPI_LIBRARIES - uncached list of libraries (using full path name) to
#    link against to use PAPI
#  PAPI_INCLUDE_DIR - directory where the PAPI header files are
#
##########

find_package(PkgConfig QUIET)

if( PAPI_DIR )
  set(ENV{PKG_CONFIG_PATH} "${PAPI_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
endif()
pkg_check_modules(PC_PAPI QUIET papi)
set(PAPI_DEFINITIONS ${PC_PAPI_CFLAGS_OTHER} )

find_path(PAPI_INCLUDE_DIR papi.h
          PATHS ${PAPI_DIR}/include ENV PAPI_INCLUDE_DIR
          HINTS ${PC_PAPI_INCLUDEDIR} ${PC_PAPI_INCLUDE_DIRS} ${PAPI_DIR}/include
          DOC "Include path for PAPI")

#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

find_library(PAPI_LIBRARY NAMES papi
             PATHS ${PAPI_DIR}/lib
             HINTS ${PC_PAPI_LIBDIR} ${PC_PAPI_LIBRARY_DIRS} ${PAPI_DIR}/lib
             DOC "Library path for PAPI")

message(STATUS "The PAPI Library is found at ${PAPI_LIBRARY}")

set(PAPI_LIBRARIES ${PAPI_LIBRARY} )
set(PAPI_INCLUDE_DIRS ${PAPI_INCLUDE_DIR} )
mark_as_advanced(FORCE PAPI_DIR PAPI_INCLUDE_DIR PAPI_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PAPI DEFAULT_MSG
                                  PAPI_LIBRARY PAPI_INCLUDE_DIR )

