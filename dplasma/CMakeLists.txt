cmake_minimum_required (VERSION 2.8)

option(BUILD_DPLASMA
       "Compile the DPLASMA layer" ON)
mark_as_advanced(DPLASMA_DEBUG_QR_PIVGEN)
option(DPLASMA_DEBUG_QR_PIVGEN
  "Enable the QR pivgen testings" OFF)

# DPLASMA specific option.
option(DPLASMA_GPU_WITH_MAGMA
  "Enable GPU support using MAGMA kernels" OFF)
if(DPLASMA_GPU_WITH_MAGMA)
  message(WARNING "MAGMA is not supported yet, ignored.")
endif()

option(DPLASMA_WITH_RECURSIVE
  "Enable recursive kernels to be called when available" OFF)
if(DPLASMA_WITH_RECURSIVE)
  # TODO: This will need to be added to the futur dplasma_config.h.in
  add_definitions(-DDAGUE_HAVE_RECURSIVE)
endif()

if(NOT DPLASMA_PRECISIONS)
  set(DPLASMA_PRECISIONS "s;d;c;z" CACHE STRING "The precisions to compile in dplasma (accepts a colon separated list of s;d;c;z)" FORCE)
else()
  set(DPLASMA_PRECISIONS "${DPLASMA_PRECISIONS}" CACHE STRING "The precisions to compile in dplasma (accepts a colon separated list of s;d;c;z)" FORCE)
endif()

IF( NOT BUILD_DPLASMA )
  RETURN()
ENDIF( NOT BUILD_DPLASMA )

if( BUILD_DPLASMA AND NOT BUILD_DAGUE )
  message(FATAL_ERROR "Building the DPLASMA layer requires the DAGuE framework")
endif( BUILD_DPLASMA AND NOT BUILD_DAGUE )

#
# Find packages to decide if we compile DPLASMA support or not
#
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/")

find_package(COREBLAS)
if( PLASMA_SRC )
   list(APPEND COREBLAS_INCLUDE_DIRS ${PLASMA_SRC})
endif( PLASMA_SRC )

mark_as_advanced( DAGUE_GPU_WITH_MAGMA )
if( DPLASMA_GPU_WITH_MAGMA )
  find_package(MAGMA)
  if( MAGMA_FOUND )
  else( MAGMA_FOUND )
  endif( MAGMA_FOUND )
endif( DPLASMA_GPU_WITH_MAGMA )

include_directories(${COREBLAS_INCLUDE_DIRS})
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if( ${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR} )
else( ${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR} )
  include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR})
  include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR}/include)
endif( ${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR} )

if(COREBLAS_FOUND)
  Add_Subdirectory(include)
  Add_Subdirectory(cores)
  Add_Subdirectory(lib)
  IF( BUILD_TESTING )
    Add_Subdirectory(testing)
  ENDIF( BUILD_TESTING )
endif(COREBLAS_FOUND)

configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/include/dplasma.pc.in"
  "${PROJECT_INCLUDE_DIR}/dplasma.pc" @ONLY)
install(FILES "${PROJECT_INCLUDE_DIR}/dplasma.pc" DESTINATION lib/pkgconfig)
