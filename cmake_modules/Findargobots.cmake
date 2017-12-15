# - This module determines if the Argobots Threading, Tasks,
#   and Synchronization routines library is available, and where
#   to find it.
# The following variables are set
#  LIBARGOBOTS_FOUND - System has libabt
#  LIBARGOBOTS_INCLUDE_DIRS - The libabt include directories
#  LIBARGOBOTS_LIBRARIES - The libraries needed to use libabt
#  LIBARGOBOTS_DEFINITIONS - Compiler switches required for using libabt

set(LIBARGOBOTS_DIR "" CACHE PATH "Root directory containing argobots install")

find_package(PkgConfig)
pkg_check_modules(PC_LIBARGOBOTS QUIET argobots)
set(LIBARGOBOTS_DEFINITIONS ${PC_LIBARGOBOTS_CFLAGS_OTHER})

find_path(LIBARGOBOTS_INCLUDE_DIR abt.h
  HINTS
  ${LIBARGOBOTS_DIR}/include ${PC_LIBARGOBOTS_INCLUDEDIR} ${PC_LIBARGOBOTS_INCLUDE_DIRS} )

find_library(LIBARGOBOTS_LIBRARY NAME abt
  HINTS
  ${LIBARGOBOTS_DIR}/lib ${PC_LIBARGOBOTS_LIBDIR} ${PC_LIBARGOBOTS_LIBRARY_DIRS} )

set(LIBARGOBOTS_LIBRARIES ${LIBARGOBOTS_LIBRARY} )
set(LIBARGOBOTS_INCLUDE_DIRS ${LIBARGOBOTS_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBARGOBOTS_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(LIBARGOBOTS DEFAULT_MSG
                                  LIBARGOBOTS_LIBRARY LIBARGOBOTS_INCLUDE_DIR)

mark_as_advanced(LIBARGOBOTS_INCLUDE_DIR LIBARGOBOTS_LIBRARY )
