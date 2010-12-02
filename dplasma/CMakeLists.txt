# DPLASMA specific option.
option(DPLASMA_CALL_TRACE
  "Enable the output of the BLAS kernels call trace during execution" OFF)
option(DPLASMA_GPU_WITH_MAGMA
  "Enable GPU support using MAGMA kernels" OFF)
if(DPLASMA_GPU_WITH_MAGMA)
  message(WARNING "MAGMA is not supported yet, ignored.")
endif()

if(NOT DPLASMA_PRECISIONS)
  set(DPLASMA_PRECISIONS "s;d" CACHE STRING "The precisions to compile in dplasma (accepts a colon separated list of s;d;c;z)" FORCE)
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

include_directories(lib)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/lib/)
include_directories(${PLASMA_INCLUDE_DIRS})
if(FOUND_PLASMA)
  Add_Subdirectory(lib)
  Add_Subdirectory(testing)
endif(FOUND_PLASMA)
