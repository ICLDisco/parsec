
# - Find PARSEC library
# This module finds an installed  library that implements the PARSEC
# The list of libraries searched for is taken
# from the autoconf macro file, acx_blas.m4 (distributed at
# http://ac-archive.sourceforge.net/ac-archive/acx_blas.html).
#
# This module is controled by the following variables:
#  PARSEC_DIR - path to look for PaRSEC
#  PARSEC_PKG_DIR - path to look for the parsec.pc pkgconfig file
#  PARSEC_BACKEND - a list of possible network backends to consider
# This module sets the following variables:
#  PARSEC_FOUND - set to true if a library implementing the PLASMA interface
#    is found
#  PARSEC_BACKEND - the actual backend used in the selected PaRSEC library
#  PARSEC_INCLUDE_DIRS - include directories
#  PARSEC_LIBRARIES - uncached list of libraries (using full path name) to
#    link against to use PaRSEC
#  PARSEC_EXTRA_LIBRARIES - uncached list of libraries (using full path name) to
#    link against to use supplementary features of PaRSEC (data distributions, etc)
#  PARSEC_STATIC  if set on this determines what kind of linkage we do (static)
#  PARSEC_VENDOR  if set checks only the specified vendor, if not set checks
#     all the possibilities
##########

# First we try to use pkg-config to find what we're looking for
# in the directory specified by the PARSEC_DIR or PARSEC_PKG_DIR
if( PARSEC_DIR )
  if( PARSEC_PKG_DIR )
    message(STATUS "PARSEC_DIR and PARSEC_PKG_DIR are set at the same time; ${PARSEC_DIR} overrides ${PARSEC_PKG_DIR}.")
  endif()
endif( PARSEC_DIR )
include(FindPkgConfig)
set(ENV{PKG_CONFIG_PATH} "${PARSEC_PKG_DIR}:$ENV{PKG_CONFIG_PATH}")
pkg_search_module(PARSEC parsec)

if( NOT PARSEC_FOUND )
  if( PARSEC_DIR )
    if( NOT PARSEC_INCLUDE_DIRS )
      set(PARSEC_INCLUDE_DIRS "${PARSEC_DIR}/include")
    endif()
    if( NOT PARSEC_TOOLDIR )
      set(PARSEC_TOOLDIR "${PARSEC_DIR}/bin")
    endif()
  else( PARSEC_DIR )
      if( PARSEC_FIND_REQUIRED )
        message(FATAL_ERROR "PaRSEC: NOT FOUND. pkg-config not available. You need to provide PARSEC_DIR.")
      endif()
  endif( PARSEC_DIR )

  include(CheckIncludeFiles)
  list(APPEND CMAKE_REQUIRED_INCLUDES ${PARSEC_INCLUDE_DIRS})
  check_include_files(parsec.h FOUND_PARSEC_INCLUDE)
  include_directories( ${PARSEC_INCLUDE_DIRS} )
  if( NOT FOUND_PARSEC_INCLUDE )
    if( PARSEC_FIND_REQUIRED )
      message(FATAL_ERROR "parsec.h: NOT FOUND in ${PARSEC_INCLUDE_DIRS}.")
    endif()
  endif( NOT FOUND_PARSEC_INCLUDE )

  if( NOT PARSEC_BACKEND )
    set(PARSEC_BACKEND "MPI;")
  endif()
  foreach(backend ${PARSEC_BACKEND})
    find_library( PARSEC_LIB parsec${backend}
      PATHS ${PARSEC_DIR}/lib
      DOC "Which PaRSEC library is used" )
    find_library( PARSEC_EXTRA_LIB parsec_distribution_matrix${backend}
      PATHS ${PARSEC_DIR}/lib
      DOC "Extra libs for PaRSEC" )
    if( PARSEC_FOUND )
      set(PARSEC_BACKEND ${backend})
      set(PARSEC_LIBRARIES ${PARSEC_LIB})
      set(PARSEC_EXTRA_LIBRARIES ${PARSEC_EXTRA_LIB})
    endif()
  endforeach()

  # Add all dependencies from the configuration
  check_include_files(parsec_config.h FOUND_PARSEC_CONFIG)
  if( FOUND_PARSEC_CONFIG )
    CheckSymbolExists(PARSEC_HAVE_HWLOC parsec_config.h CHK_WITH_HWLOC)
    if( CHK_WITH_HWLOC )
      find_package(HWLOC REQUIRED PARSEC_FIND_QUIETLY)
      set(PARSEC_LIBRARIES "${PARSEC_LIBRARIES};${HWLOC_LIBRARIES}")
    endif()
    CheckSymbolExists(PARSEC_HAVE_PAPI parsec_config.h CHK_WITH_PAPI)
    if( CHK_WITH_PAPI )
      find_package(PAPI REQUIRED PARSEC_FIND_QUIETLY)
      set(PARSEC_LIBRARIES "${PARSEC_LIBRARIES};${PAPI_LIBRARIES}")
    endif()
    CheckSymbolExists(PARSEC_HAVE_PTHREADS parsec_config.h CHK_WITH_PTHREADS)
    if( CHK_WITH_PTHREADS )
      find_package(Threads REQUIRED PARSEC_FIND_QUIETLY)
      set(PARSEC_LIBRARIES "${PARSEC_LIBRARIES};${CMAKE_THREAD_LIBS_INIT}")
    endif()
    if( ${PARSEC_BACKEND} STREQUAL "MPI" )
      CheckSymbolExists(PARSEC_HAVE_MPI parsec_config.h CHK_WITH_MPI)
      if( NOT CHK_WITH_MPI )
        message(WARNING "Header parsec_config.h in ${PARSEC_INCLUDE_DIRS} doesn't match the compiled library ${PARSEC_LIB}")
      endif()
      find_package(MPI QUIET)
      if( MPI_C_FOUND )
        set(PARSEC_LIBRARIES "${PARSEC_LIBRARIES};${MPI_LIBRARIES}")
      else(MPI_C_FOUND)
        if( NOT PARSEC_FIND_QUIETLY )
          message(WARNING "MPI version of PaRSEC found in ${PARSEC_LIBRARIES}, but no suitable MPI found.")
        endif()
        set(PARSEC_FOUND FALSE)
      endif()
    endif( ${backend} STREQUAL "MPI" )
  else( FOUND_PARSEC_CONFIG )
    if( NOT PARSEC_FIND_QUIETLY )
      message("parsec_config.h not found; some required libraries may not have been added to the link line...")
    endif()
  endif( FOUND_PARSEC_CONFIG )

  if( PARSEC_FIND_REQUIRED AND NOT PARSEC_FOUND )
    message(FATAL_ERROR "PaRSEC: NOT FOUND in ${PARSEC_DIR}.")
  endif()
endif( NOT PARSEC_FOUND )

mark_as_advanced(PARSEC_DIR PARSEC_PKG_DIR PARSEC_BACKEND PARSEC_LIBRARY PARSEC_LIBRARIES PARSEC_EXTRA_LIBRARIES PARSEC_INCLUDE_DIRS)
set(PARSEC_DIR "${PARSEC_DIR}" CACHE PATH "Location of the PaRSEC library" FORCE)
set(PARSEC_PKG_DIR "${PARSEC_PKG_DIR}" CACHE PATH "Location of the PaRSEC pkg-config description file" FORCE)
set(PARSEC_BACKEND "${PARSEC_BACKEND}" CACHE STRING "Type of distributed memory transport backend used by PaRSEC" FORCE)
set(PARSEC_INCLUDE_DIRS "${PARSEC_INCLUDE_DIRS}" CACHE PATH "PaRSEC include directories" FORCE)
set(PARSEC_LIBRARIES "${PARSEC_LIBRARIES}" CACHE STRING "libraries to link with PaRSEC" FORCE)
set(PARSEC_EXTRA_LIBRARIES "${PARSEC_EXTRA_LIBRARIES}" CACHE STRING "libraries to link with PaRSEC supplements (data distribution, etc.)" FORCE)

find_package_message(PARSEC
    "Found PARSEC: ${PARSEC_LIB}
        PARSEC_BACKEND           = [${PARSEC_BACKEND}]
        PARSEC_INCLUDE_DIRS      = [${PARSEC_INCLUDE_DIRS}]
        PARSEC_LIBRARIES         = [${PARSEC_LIBRARIES}]
        PARSEC_EXTRA_LIBRARIES   = [${PARSEC_EXTRA_LIBRARIES}]"
      "[${PARSEC_BACKEND}][${PARSEC_INCLUDE_DIRS}][${PARSEC_LIBRARIES}][${PARSEC_EXTRA_LIBRARIES}]")

