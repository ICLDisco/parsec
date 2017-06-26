cmake_minimum_required (VERSION 2.8)

option(BUILD_DPLASMA
       "Compile the DPLASMA layer" ON)
IF( NOT BUILD_DPLASMA )
  RETURN()
ENDIF( NOT BUILD_DPLASMA )

# DPLASMA specific option.
option(DPLASMA_WITH_RECURSIVE
  "Enable recursive kernels to be called when available" OFF)
if(DPLASMA_WITH_RECURSIVE)
  # TODO: This will need to be added to the futur dplasma_config.h.in
  add_definitions(-DPARSEC_HAVE_RECURSIVE)
endif()

mark_as_advanced( PARSEC_GPU_WITH_MAGMA )
option(DPLASMA_GPU_WITH_MAGMA
  "Enable GPU support using MAGMA kernels" OFF)
if(DPLASMA_GPU_WITH_MAGMA)
  message(WARNING "MAGMA is not supported yet, ignored.")
endif()

mark_as_advanced(DPLASMA_DEBUG_QR_PIVGEN)
option(DPLASMA_DEBUG_QR_PIVGEN
  "Enable the QR pivgen testings" OFF)

if( BUILD_DPLASMA AND NOT BUILD_PARSEC )
  message(FATAL_ERROR "Building the DPLASMA layer requires the PaRSEC framework")
endif( BUILD_DPLASMA AND NOT BUILD_PARSEC )

#
# Find packages to decide if we compile DPLASMA support or not
#
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/")

include(FortranCInterface)

find_package(COREBLAS)
if( PLASMA_SRC )
   list(APPEND COREBLAS_INCLUDE_DIRS ${PLASMA_SRC})
endif( PLASMA_SRC )

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

if(NOT DPLASMA_PRECISIONS)
  set(DPLASMA_PRECISIONS "s;d;c;z" CACHE STRING "The precisions to compile in dplasma (accepts a colon separated list of s;d;c;z)" FORCE)
else()
  set(DPLASMA_PRECISIONS "${DPLASMA_PRECISIONS}" CACHE STRING "The precisions to compile in dplasma (accepts a colon separated list of s;d;c;z)" FORCE)
endif()
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tools/PrecisionGenerator/")

# Veclib on OSX diverges from the standard BLAS interface. As a result, single
# precision dplasma tests (real and complex) deliver wrong results. 
# Until we have a fix, make sure to disable these 2 presicisions with vecLib.
if( PARSEC_OSX )
  check_function_exists(appleblas_sgeadd PARSEC_OSX_VECLIB)
  if( PARSEC_OSX_VECLIB )
    set(tmp_fixed_prec "")
    foreach(prec ${DPLASMA_PRECISIONS})
      STRING(COMPARE EQUAL "${prec}" "s" is_sprec)
      STRING(COMPARE EQUAL "${prec}" "c" is_cprec)
      if ( NOT (is_sprec OR is_cprec) )
        set(tmp_fixed_prec "${tmp_fixed_prec} ${prec}")
      endif ( NOT (is_sprec OR is_cprec) )
    endforeach()
    STRING(STRIP "${tmp_fixed_prec}" tmp_fixed_prec)
    STRING(REPLACE " " ";" NEW_PREC "${tmp_fixed_prec}")
    STRING(COMPARE NOTEQUAL "${NEW_PREC}" "${DPLASMA_PRECISIONS}" prec_change)
    if( prec_change )
        MESSAGE(WARNING "Detected Apple vecLIB! the Accelerate Framework has a non-standard BLAS interface that causes Dplasma to produce wrong results in single precision. Restricting DPLASMA to ${NEW_PREC} (from ${DPLASMA_PRECISIONS})")
      set(DPLASMA_PRECISIONS ${NEW_PREC})
    endif( prec_change )
  endif( PARSEC_OSX_VECLIB )
endif( PARSEC_OSX )

if(COREBLAS_FOUND)
  Add_Subdirectory(include)
  Add_Subdirectory(cores)
  Add_Subdirectory(lib)
  IF( BUILD_TESTING )
    Add_Subdirectory(testing)
  ENDIF( BUILD_TESTING )
  Add_Subdirectory(examples)
endif(COREBLAS_FOUND)

configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/include/dplasma.pc.in"
  "${PROJECT_INCLUDE_DIR}/dplasma.pc" @ONLY)
install(FILES "${PROJECT_INCLUDE_DIR}/dplasma.pc"
        DESTINATION dplasma/lib/pkgconfig)
