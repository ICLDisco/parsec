IF( NOT BUILD_DPLASMA )
  message(FATAL "The CMakeLists.txt in the dplasma directory is not supposed to be included if support for DPLASMA has been disabled")
ENDIF( NOT BUILD_DPLASMA )

mark_as_advanced(DPLASMA_DEBUG_QR_PIVGEN)
option(DPLASMA_DEBUG_QR_PIVGEN
  "Enable the QR pivgen testings" OFF)

# DPLASMA specific option.
option(DPLASMA_GPU_WITH_MAGMA
  "Enable GPU support using MAGMA kernels" OFF)
if(DPLASMA_GPU_WITH_MAGMA)
  message(WARNING "MAGMA is not supported yet, ignored.")
endif()

if(NOT DPLASMA_PRECISIONS)
  set(DPLASMA_PRECISIONS "s;d;c;z" CACHE STRING "The precisions to compile in dplasma (accepts a colon separated list of s;d;c;z)" FORCE)
else()
  set(DPLASMA_PRECISIONS "${DPLASMA_PRECISIONS}" CACHE STRING "The precisions to compile in dplasma (accepts a colon separated list of s;d;c;z)" FORCE)
endif()

#
# Find packages to decide if we compile DPLASMA support or not
#
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
