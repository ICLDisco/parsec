# DPLASMA specific option.
option(DPLASMA_GPU_WITH_MAGMA
  "Enable GPU support using MAGMA kernels" OFF)
if(DPLASMA_GPU_WITH_MAGMA)
  message(WARNING "MAGMA is not supported yet, ignored.")
endif()

if(NOT DPLASMA_PRECISIONS)
  set(DPLASMA_PRECISIONS "s;d" CACHE STRING "The precisions to compile in dplasma (accepts a colon separated list of s;d;c;z)" FORCE)
else()
  set(DPLASMA_PRECISIONS "${DPLASMA_PRECISIONS}" CACHE STRING "The precisions to compile in dplasma (accepts a colon separated list of s;d;c;z)" FORCE)
endif()

#
# Find required packages
#
find_package(PLASMA)
if( PLASMA_SRC )
   list(APPEND PLASMA_INCLUDE_DIR ${PLASMA_SRC})
endif( PLASMA_SRC )

if( DPLASMA_GPU_WITH_MAGMA )
  find_package(MAGMA)
  if( MAGMA_FOUND )
  else()
  endif()
endif( DPLASMA_GPU_WITH_MAGMA )

include_directories(${PLASMA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
if( ${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR} )
else( ${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR} )
  include_directories(${CMAKE_CURRENT_BINARY_DIR})
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
endif( ${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR} )

if(PLASMA_FOUND)
  Add_Subdirectory(include)
  Add_Subdirectory(cores)
  Add_Subdirectory(lib)
  Add_Subdirectory(testing)
endif(PLASMA_FOUND)
