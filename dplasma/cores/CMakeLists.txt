include(RulesPrecisions)
# reset variables
set(generated_files "")
set(generated_headers "")

include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

if( CUDA_FOUND )
    set(CORE_CUDA_HEADERS
        cuda_zgemm.h
        cuda_ztsmqr.h
    )
    set(CORE_CUDA_SOURCES
        cuda_zgemm.c
        cuda_ztsmqr.c
        dplasma_cuda_ztsmqr.c
    )
else()
    set(CORE_CUDA_HEADERS "")
    set(CORE_CUDA_SOURCES "")
endif()

### generate the dplasma_cores headers for all possible precisions
set(HEADERS
    dplasma_zcores.h
    ${CORE_CUDA_HEADERS}
)
precisions_rules_py(generated_headers
                 ${HEADERS}
                 PRECISIONS "s;d;c;z")
add_custom_target(dplasma_cores_includes ALL SOURCES
    ${generated_headers} )

foreach(generated_header ${generated_headers})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${generated_header} DESTINATION include/cores)
endforeach()


### Generate the dplasma wrappers for all required precisions
set(SOURCES
  core_ztrdv.c
  core_zhetrf2_nopiv.c
  core_zgemdm.c
  core_zhedrk.c
  core_ztrmdm.c
  core_zhetrf_nopiv.c
  core_zhebut.c
  core_zamax.c
  core_zgeadd.c
  core_ztradd.c
)
precisions_rules_py(generated_files
                 "${SOURCES}"
                 PRECISIONS "${DPLASMA_PRECISIONS}")

### Generate the dplasma_cores lib
link_directories(${COREBLAS_LIBRARY_DIRS})
add_library(dplasma_cores
  ${generated_files}
)
target_link_libraries(dplasma_cores
  ${COREBLAS_LIBRARIES}
  ${EXTRA_LIBS}
)
add_dependencies(dplasma_cores
         dplasma_includes
         dplasma_cores_includes)

install(TARGETS dplasma_cores
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

if( CUDA_FOUND )
  # generate the cores_cuda library (hooks to cublas)
  precisions_rules_py(generated_cuda_files
                 ${CORE_CUDA_SOURCES}
                 PRECISIONS "${DPLASMA_PRECISIONS}")

  if( NOT "${generated_cuda_files}" STREQUAL "" )
    add_library(dplasma_cores_cuda
      ${generated_cuda_files}
      )
    add_dependencies(dplasma_cores_cuda
      dplasma_includes
      dplasma_cores_includes)
    target_link_libraries(dplasma_cores_cuda
      ${COREBLAS_LIBRARIES}
      ${EXTRA_LIBS}
      )
    cuda_add_cublas_to_target( dplasma_cores_cuda )
    target_link_libraries(dplasma_cores dplasma_cores_cuda)
    install(TARGETS dplasma_cores_cuda
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib)
  endif()

  ### Generate the CUDA .cu kernels if necessary
  # By default drop support for older devices (add 11 and 13 for support */
  if( ${CUDA_VERSION_STRING} VERSION_GREATER "4.1")
    set(CUDA_SM_TARGETS 20 35 CACHE STRING "Specify GPU architectures to build binaries for, BIN(PTX) format is supported")
  else()
    set(CUDA_SM_TARGETS 11 13 20 CACHE STRING "Specify GPU architectures to build binaries for, BIN(PTX) format is supported")
  endif()

# For Tesla
  set(CUDA_NVCC_FLAGS_11 -maxrregcount 32 -arch sm_11 -DCUDA_SM_VERSION=11)
  set(CUDA_NVCC_FLAGS_13 -maxrregcount 32 -arch sm_13 -DCUDA_SM_VERSION=13)
# For Fermi
  set(CUDA_NVCC_FLAGS_20                  -arch sm_20 -DCUDA_SM_VERSION=20)
# For Kepler
  set(CUDA_NVCC_FLAGS_30                  -arch sm_30 -DCUDA_SM_VERSION=30)
  set(CUDA_NVCC_FLAGS_35                  -arch sm_35 -DCUDA_SM_VERSION=35)
  set(CUDA_NVCC_FLAGS_37                  -arch sm_37 -DCUDA_SM_VERSION=37)
# For Maxwell
  set(CUDA_NVCC_FLAGS_50                  -arch sm_50 -DCUDA_SM_VERSION=50)
  set(CUDA_NVCC_FLAGS_52                  -arch sm_52 -DCUDA_SM_VERSION=52)
  set(CUDA_NVCC_FLAGS_53                  -arch sm_53 -DCUDA_SM_VERSION=53)

  ### CUDA .cu sources
  set( CUDA11_SOURCES
#       zgemm_11_12_13.cu
  )
  set( CUDA13_SOURCES ${CUDA11_SOURCES} )
  set( CUDA20_SOURCES
#        zgemm_20_30.cu
  )
  set( CUDA30_SOURCES ${CUDA20_SOURCES} )
  set( CUDA35_SOURCES ${CUDA20_SOURCES} )
  set( CUDA37_SOURCES ${CUDA20_SOURCES} )

  foreach( _smtarget ${CUDA_SM_TARGETS} )
    if( NOT "${CUDA${_smtarget}_SOURCES}" STREQUAL "" )
      set(cuda_generated_files "")
      precisions_rules_py(cuda_generated_files
                          "${CUDA${_smtarget}_SOURCES}"
                          PRECISIONS "${DPLASMA_PRECISIONS}")
      foreach( _cudafile ${cuda_generated_files} )
          get_source_file_property(_IsInBinaryDir ${_cudafile} IS_IN_BINARY_DIR )
          if( _IsInBinaryDir )
              string(REGEX REPLACE
                      ${_cudafile}
                      ${CMAKE_CURRENT_BINARY_DIR}/${_cudafile}
                      cuda_generated_files "${cuda_generated_files}")
          endif( _IsInBinaryDir )
      endforeach()

      set(CUDA_NVCC_FLAGS_BACKUP ${CUDA_NVCC_FLAGS})
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_${_smtarget}} ${CUDA_NVCC_FLAGS})
      cuda_add_library(dplasma_cucores_sm${_smtarget}
                       ${cuda_generated_files})
      cuda_add_cublas_to_target( dplasma_cucores_sm${_smtarget} )
      install(TARGETS dplasma_cucores_sm${_smtarget} LIBRARY DESTINATION lib ARCHIVE DESTINATION lib )
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_BACKUP})
      if( NOT CMAKE_BUILD_SHARED_LIBS )
          target_link_libraries( dplasma_cores_cuda dplasma_cucores_sm${_smtarget} )
      endif()
    endif()
  endforeach()

  cuda_build_clean_target()
endif()
