include(RulesPrecisions)
# reset variables
set(generated_files "")
set(generated_headers "")

include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

### Generate the CUDA kernels if necessary
if(CUDA_FOUND)
    set(CUDA_SM_TARGETS 11 13 20 35)
    set(CUDA_NVCC_FLAGS_11 -maxrregcount 32 -arch sm_11 -DCUDA_SM_VERSION=11)
    set(CUDA_NVCC_FLAGS_13 -maxrregcount 32 -arch sm_13 -DCUDA_SM_VERSION=13)
    set(CUDA_NVCC_FLAGS_20                  -arch sm_20 -DCUDA_SM_VERSION=20)
    set(CUDA_NVCC_FLAGS_35                  -arch sm_35 -DCUDA_SM_VERSION=35)

### CUDA .cu sources
    set( CUDA11_SOURCES
#         zgemm_11_12_13.cu
    )
    set( CUDA13_SOURCES ${CUDA11_SOURCES} )
    set( CUDA20_SOURCES
#         zgemm_20_30.cu
    )
    set( CUDA35_SOURCES ${CUDA11_SOURCES} )

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
                         ${cuda_generated_files} SHARED)
        cuda_add_cublas_to_target( dplasma_cucores_sm${_smtarget} )
        install(TARGETS dplasma_cucores_sm${_smtarget} LIBRARY DESTINATION lib ARCHIVE DESTINATION lib )
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_BACKUP})
        if( NOT CMAKE_BUILD_SHARED_LIBS )
            list(APPEND GPU_KERNEL_LIBS dplasma_cucores_sm${_smtarget})
        endif()
      endif()
    endforeach()

    cuda_build_clean_target()

### Now the gpu support core files
    set(CORE_CUDA_HEADERS
        cuda_zgemm.h
#        cuda_ztsmqr.h
    )
    set(CORE_CUDA_SOURCES
        cuda_zgemm.c
#        cuda_ztsmqr.c
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
endif()

foreach(generated_header ${generated_headers})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${generated_header} DESTINATION include/cores)
endforeach()
