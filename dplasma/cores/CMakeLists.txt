include(RulesPrecisions)
# reset variables
set(generated_files "")
set(generated_headers "")

include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

### Generate the CUDA kernels if necessary
if(CUDA_FOUND)

    # CUDA sources
    set( CUDA_SOURCES
         zgemm_11_12_13.cu
         zgemm_20_30.cu
    )
    precisions_rules_py(cuda_generated_files
                        "${CUDA_SOURCES}"
                        PRECISIONS "${DPLASMA_PRECISIONS}")

    set( CUDA_SOURCES
         ${cuda_generated_files}
#         cuda_sgemm_kernel.cu
#         cuda_sgemm_kernelf.cu
    )

    set(CUDA_NVCC_FLAGS_11 -maxrregcount 32 -arch sm_11 -DCUDA_SM_VERSION=11 ${CUDA_NVCC_FLAGS_BACKUP})
    set(CUDA_NVCC_FLAGS_13 -maxrregcount 32 -arch sm_13 -DCUDA_SM_VERSION=13 ${CUDA_NVCC_FLAGS_BACKUP})
    set(CUDA_NVCC_FLAGS_20                  -arch sm_20 -DCUDA_SM_VERSION=20 ${CUDA_NVCC_FLAGS_BACKUP})

    set(CUDA_BUILD_CUBIN ON)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_NVCC_FLAGS_BACKUP ${CUDA_NVCC_FLAGS})

    set( cudalibs_deps
         ${CUDA_cublas_LIBRARY}
         ${CUDA_CUDART_LIBRARY}
         ${CUDA_CUDA_LIBRARY}
         )

    foreach( _cudafile ${CUDA_SOURCES} )
        string(REGEX REPLACE "\\.cu" "" _cudafilename ${_cudafile})

        get_source_file_property(_IsInBinaryDir ${_cudafile} IS_IN_BINARY_DIR )

        if( _IsInBinaryDir )
            set(SM11_FILES ${SM11_FILES} ${CMAKE_CURRENT_BINARY_DIR}/${_cudafile})
            set(SM13_FILES ${SM13_FILES} ${CMAKE_CURRENT_BINARY_DIR}/${_cudafile})
            set(SM20_FILES ${SM20_FILES} ${CMAKE_CURRENT_BINARY_DIR}/${_cudafile})
        else( _IsInBinaryDir )
            set(SM11_FILES ${SM11_FILES} ${_cudafile})
            set(SM13_FILES ${SM13_FILES} ${_cudafile})
            set(SM20_FILES ${SM20_FILES} ${_cudafile})
        endif( _IsInBinaryDir )
    endforeach()

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_11})
    cuda_add_library(dplasma_sm11 ${SM11_FILES} SHARED )
    target_link_libraries( dplasma_sm11 ${cudalibs_deps} )
    install(TARGETS dplasma_sm11 LIBRARY DESTINATION lib)

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_13})
    cuda_add_library(dplasma_sm13 ${SM13_FILES} SHARED )
    target_link_libraries( dplasma_sm13 ${cudalibs_deps} )
    install(TARGETS dplasma_sm13 LIBRARY DESTINATION lib)

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_20})
    cuda_add_library(dplasma_sm20 ${SM20_FILES} SHARED )
    target_link_libraries( dplasma_sm20 ${cudalibs_deps} )
    install(TARGETS dplasma_sm20 LIBRARY DESTINATION lib)

    # Install cubin files
#     foreach( _cudafile ${CUDA_SOURCES} )
#         string(REGEX REPLACE "\\.cu" "" _cudafilename ${_cudafile})

#         install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/dplasma_sm11.dir/dplasma_sm11_generated_${_cudafile}.o.cubin.txt
#                 DESTINATION lib
#                 RENAME dplasma_sm11_${_cudafilename}.cubin )

#         install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/dplasma_sm13.dir/dplasma_sm13_generated_${_cudafile}.o.cubin.txt
#                 DESTINATION lib
#                 RENAME dplasma_sm13_${_cudafilename}.cubin )

#         install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/dplasma_sm20.dir/dplasma_sm20_generated_${_cudafile}.o.cubin.txt
#                 DESTINATION lib
#                 RENAME dplasma_sm20_${_cudafilename}.cubin )
#     endforeach()

    cuda_build_clean_target()

    set(GPU_KERNEL_SOURCES
        cuda_zgemm.c
#        cuda_stsmqr.c
    )

    set(GPU_KERNEL_HEADERS
        cuda_zgemm.h
#        cuda_stsmqr.h
    )

else()
    set(GPU_KERNEL_SOURCES "")
    set(GPU_KERNEL_HEADERS "")
endif()

set(HEADERS
    dplasma_zcores.h
    ${GPU_KERNEL_HEADERS}
)

### generate the dplasma_cores headers for all possible precisions
precisions_rules_py(generated_headers
                    "${HEADERS}"
                    PRECISIONS "s;d;c;z")

### Generate the dplasma wrappers for all required precisions 
set(SOURCES
  core_ztrdv.c
  core_zhetrf2_nopiv.c
  core_zgemdm.c
  core_zhedrk.c
  core_ztrmdm.c
  core_zhetrf_nopiv.c
  core_zhebut.c
  core_zplssq.c
  ${GPU_KERNEL_SOURCES}
)

precisions_rules_py(generated_files
                 "${SOURCES}"
                 PRECISIONS "${DPLASMA_PRECISIONS}")

### Generate the lib
add_library(dplasma_cores
  ${generated_headers}
  ${generated_files}
)
install(TARGETS dplasma_cores ARCHIVE DESTINATION lib)

foreach(generated_header ${generated_headers})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${generated_header} DESTINATION include/cores)
endforeach()
