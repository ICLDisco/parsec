
# - Find DAGUE library
# This module finds an installed  library that implements the DAGUE
# The list of libraries searched for is taken
# from the autoconf macro file, acx_blas.m4 (distributed at
# http://ac-archive.sourceforge.net/ac-archive/acx_blas.html).
#
# This module sets the following variables:
#  DAGUE_FOUND - set to true if a library implementing the PLASMA interface
#    is found
#  DAGUE_LINKER_FLAGS - uncached list of required linker flags (excluding -l
#    and -L).
#  DAGUE_LIBRARIES - uncached list of libraries (using full path name) to
#    link against to use PLASMA
#  DAGUE_STATIC  if set on this determines what kind of linkage we do (static)
#  DAGUE_VENDOR  if set checks only the specified vendor, if not set checks
#     all the possibilities
##########

# If we only have the main PLASMA directory componse the include and
# libraries path based on it.
if( DAGUE_DIR )
  if( NOT DAGUE_INCLUDE_DIR )
    set(DAGUE_INCLUDE_DIR "${DAGUE_DIR}")
  endif( NOT DAGUE_INCLUDE_DIR )
  if( NOT DAGUE_LIBRARIES )
    set(DAGUE_LIBRARIES "${DAGUE_DIR}")
  endif( NOT DAGUE_LIBRARIES )
  if ( NOT DAGUE_TOOLDIR )
    set(DAGUE_TOOLDIR "${DAGUE_DIR}/tools/dague-compiler")
  endif ( NOT DAGUE_TOOLDIR )
else( DAGUE_DIR )
    message( "/!\\ DAGUE_DIR not defined !!!" )
endif( DAGUE_DIR )

list(APPEND CMAKE_REQUIRED_INCLUDES ${DAGUE_INCLUDE_DIR})

include_directories( ${DAGUE_INCLUDE_DIR} )
#include(CheckIncludeFile)
CHECK_INCLUDE_FILES(FOUND_DAGUE_INCLUDE "dague.h")

if ( NOT FOUND_DAGUE_INCLUDE )
    #message( FATAL_ERROR "DAGUE header not found" )
endif ( NOT FOUND_DAGUE_INCLUDE )

find_library( DAGUE_LIB dague
        PATHS ${DAGUE_LIBRARIES}
        DOC "Where the DAGUE libraries are" )

if ( DAGUE_LIB )
    message( "-- DAGUE lib found" )
endif( DAGUE_LIB )

find_library( DAGUEMPI_LIB dague-mpi
        PATHS ${DAGUE_LIBRARIES}
        DOC "Where the DAGUE libraries compiled with MPI support are" )

if ( DAGUEMPI_LIB )
    message( "-- DAGUE Compiled with MPI FOUND" )
endif( DAGUEMPI_LIB )

