
# - Find DAGUE library
# This module finds an installed  library that implements the DAGUE
# The list of libraries searched for is taken
# from the autoconf macro file, acx_blas.m4 (distributed at
# http://ac-archive.sourceforge.net/ac-archive/acx_blas.html).
#
# This module is controled by the following variables:
#  DAGUE_DIR - path to look for DAGuE
#  DAGUE_PKG_DIR - path to look for the dague.pc pkgconfig file
#  DAGUE_BACKEND - a list of possible network backends to consider
# This module sets the following variables:
#  DAGUE_FOUND - set to true if a library implementing the PLASMA interface
#    is found
#  DAGUE_BACKEND - the actual backend used in the selected DAGuE library
#  DAGUE_INCLUDE_DIRS - include directories
#  DAGUE_LIBRARIES - uncached list of libraries (using full path name) to
#    link against to use DAGuE
#  DAGUE_EXTRA_LIBRARIES - uncached list of libraries (using full path name) to
#    link against to use supplementary features of DAGuE (data distributions, etc)
#  DAGUE_STATIC  if set on this determines what kind of linkage we do (static)
#  DAGUE_VENDOR  if set checks only the specified vendor, if not set checks
#     all the possibilities
##########

# First we try to use pkg-config to find what we're looking for
# in the directory specified by the DAGUE_DIR or DAGUE_PKG_DIR
if( DAGUE_DIR )
  if( DAGUE_PKG_DIR )
    message(STATUS "DAGUE_DIR and DAGUE_PKG_DIR are set at the same time; ${DAGUE_DIR} overrides ${DAGUE_PKG_DIR}.")
  endif()
endif( DAGUE_DIR )
include(FindPkgConfig)
set(ENV{PKG_CONFIG_PATH} "${DAGUE_PKG_DIR}:$ENV{PKG_CONFIG_PATH}")
pkg_search_module(DAGUE dague)

if( NOT DAGUE_FOUND )
  if( DAGUE_DIR )
    if( NOT DAGUE_INCLUDE_DIRS )
      set(DAGUE_INCLUDE_DIRS "${DAGUE_DIR}/include")
    endif()
    if( NOT DAGUE_TOOLDIR )
      set(DAGUE_TOOLDIR "${DAGUE_DIR}/bin")
    endif()
  else( DAGUE_DIR )
      if( DAGUE_FIND_REQUIRED )
        message(FATAL_ERROR "DAGuE: NOT FOUND. pkg-config not available. You need to provide DAGUE_DIR.")
      endif()
  endif( DAGUE_DIR )

  include(CheckIncludeFiles)
  list(APPEND CMAKE_REQUIRED_INCLUDES ${DAGUE_INCLUDE_DIRS})
  check_include_files(dague.h FOUND_DAGUE_INCLUDE)
  include_directories( ${DAGUE_INCLUDE_DIRS} )
  if( NOT FOUND_DAGUE_INCLUDE )
    if( DAGUE_FIND_REQUIRED )
      message(FATAL_ERROR "dague.h: NOT FOUND in ${DAGUE_INCLUDE_DIRS}.")
    endif()
  endif( NOT FOUND_DAGUE_INCLUDE )

  if( NOT DAGUE_BACKEND )
    set(DAGUE_BACKEND "MPI;")
  endif()
  foreach(backend ${DAGUE_BACKEND})
    find_library( DAGUE_LIB dague${backend}
      PATHS ${DAGUE_DIR}/lib
      DOC "Which DAGuE library is used" )
    find_library( DAGUE_EXTRA_LIB dague_distribution_matrix${backend}
      PATHS ${DAGUE_DIR}/lib
      DOC "Extra libs for DAGuE" )
    if( DAGUE_FOUND )
      set(DAGUE_BACKEND ${backend})
      set(DAGUE_LIBRARIES ${DAGUE_LIB})
      set(DAGUE_EXTRA_LIBRARIES ${DAGUE_EXTRA_LIB})
    endif()
  endforeach()

  # Add all dependencies from the configuration
  check_include_files(dague_config.h FOUND_DAGUE_CONFIG)
  if( FOUND_DAGUE_CONFIG )
    CheckSymbolExists(DAGUE_HAVE_HWLOC dague_config.h CHK_WITH_HWLOC)
    if( CHK_WITH_HWLOC )
      find_package(HWLOC REQUIRED DAGUE_FIND_QUIETLY)
      set(DAGUE_LIBRARIES "${DAGUE_LIBRARIES};${HWLOC_LIB}")
    endif()
    CheckSymbolExists(DAGUE_HAVE_PAPI dague_config.h CHK_WITH_PAPI)
    if( CHK_WITH_PAPI )
      find_package(PAPI REQUIRED DAGUE_FIND_QUIETLY)
      set(DAGUE_LIBRARIES "${DAGUE_LIBRARIES};${PAPI_LIBRARY}")
    endif()
    CheckSymbolExists(DAGUE_HAVE_PTHREADS dague_config.h CHK_WITH_PTHREADS)
    if( CHK_WITH_PTHREADS )
      find_package(Threads REQUIRED DAGUE_FIND_QUIETLY)
      set(DAGUE_LIBRARIES "${DAGUE_LIBRARIES};${CMAKE_THREAD_LIBS_INIT}")
    endif()
    if( ${DAGUE_BACKEND} STREQUAL "MPI" )
      CheckSymbolExists(DAGUE_HAVE_MPI dague_config.h CHK_WITH_MPI)
      if( NOT CHK_WITH_MPI )
        message(WARNING "Header dague_config.h in ${DAGUE_INCLUDE_DIRS} doesn't match the compiled library ${DAGUE_LIB}")
      endif()
      find_package(MPI QUIET)
      if( MPI_C_FOUND )
        set(DAGUE_LIBRARIES "${DAGUE_LIBRARIES};${MPI_LIBRARIES}")
      else(MPI_C_FOUND)
        if( NOT DAGUE_FIND_QUIETLY )
          message(WARNING "MPI version of DAGuE found in ${DAGUE_LIBRARIES}, but no suitable MPI found.")
        endif()
        set(DAGUE_FOUND FALSE)
      endif()
    endif( ${backend} STREQUAL "MPI" )
  else( FOUND_DAGUE_CONFIG )
    if( NOT DAGUE_FIND_QUIETLY )
      message("dague_config.h not found; some required libraries may not have been added to the link line...")
    endif()
  endif( FOUND_DAGUE_CONFIG )

  if( DAGUE_FIND_REQUIRED AND NOT DAGUE_FOUND )
    message(FATAL_ERROR "DAGuE: NOT FOUND in ${DAGUE_DIR}.")
  endif()
endif( NOT DAGUE_FOUND )

mark_as_advanced(DAGUE_DIR DAGUE_PKG_DIR DAGUE_BACKEND DAGUE_LIBRARY DAGUE_LIBRARIES DAGUE_EXTRA_LIBRARIES DAGUE_INCLUDE_DIRS)
set(DAGUE_DIR "${DAGUE_DIR}" CACHE PATH "Location of the DAGuE library" FORCE)
set(DAGUE_PKG_DIR "${DAGUE_PKG_DIR}" CACHE PATH "Location of the DAGuE pkg-config description file" FORCE)
set(DAGUE_BACKEND "${DAGUE_BACKEND}" CACHE STRING "Type of distributed memory transport backend used by DAGuE" FORCE)
set(DAGUE_INCLUDE_DIRS "${DAGUE_INCLUDE_DIRS}" CACHE PATH "DAGuE include directories" FORCE)
set(DAGUE_LIBRARIES "${DAGUE_LIBRARIES}" CACHE STRING "libraries to link with DAGuE" FORCE)
set(DAGUE_EXTRA_LIBRARIES "${DAGUE_EXTRA_LIBRARIES}" CACHE STRING "libraries to link with DAGuE supplements (data distribution, etc.)" FORCE)

find_package_message(DAGUE
    "Found DAGUE: ${DAGUE_LIB}
        DAGUE_BACKEND           = [${DAGUE_BACKEND}]
        DAGUE_INCLUDE_DIRS      = [${DAGUE_INCLUDE_DIRS}]
        DAGUE_LIBRARIES         = [${DAGUE_LIBRARIES}]
        DAGUE_EXTRA_LIBRARIES   = [${DAGUE_EXTRA_LIBRARIES}]"
      "[${DAGUE_BACKEND}][${DAGUE_INCLUDE_DIRS}][${DAGUE_LIBRARIES}][${DAGUE_EXTRA_LIBRARIES}]")

