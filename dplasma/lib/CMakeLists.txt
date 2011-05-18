include(RulesPrecisions)
include(RulesJDF)
# reset variables
set(generated_files "")
set(generated_jdf "")


if(CUDA_FOUND AND "${DPLASMA_PRECISIONS}" MATCHES "s")
    set(CUDA_BUILD_CUBIN ON)
    set(CUDA_NVCC_FLAGS -maxrregcount 32 -arch sm_11)
    cuda_add_library( sgemm-sm_11 cuda_sgemm_kernel.cu)
    install(TARGETS sgemm-sm_11 ARCHIVE DESTINATION lib)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sgemm-sm_11_generated_cuda_sgemm_kernel.cu.o.cubin.txt
      DESTINATION lib
      RENAME sgemm-sm_11.cubin)

    set(CUDA_NVCC_FLAGS -maxrregcount 32 -arch sm_13)
    cuda_add_library( sgemm-sm_13 cuda_sgemm_kernel.cu)
    install(TARGETS sgemm-sm_13 ARCHIVE DESTINATION lib)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sgemm-sm_13_generated_cuda_sgemm_kernel.cu.o.cubin.txt
      DESTINATION lib
      RENAME sgemm-sm_13.cubin)

    set(CUDA_NVCC_FLAGS -arch sm_20)
    cuda_add_library( sgemm-sm_20 cuda_sgemm_kernelf.cu)
    install(TARGETS sgemm-sm_20 ARCHIVE DESTINATION lib)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sgemm-sm_20_generated_cuda_sgemm_kernelf.cu.o.cubin.txt
      DESTINATION lib
      RENAME sgemm-sm_20.cubin)

    set(GPU_KERNEL_SOURCES cuda_sgemm.c cuda_stsmqr.c)
else()
  set(GPU_KERNEL_SOURCES "")
endif()
set(EXTRA_SOURCES ${GPU_KERNEL_SOURCES} memory_pool.c dplasmatypes.c dplasmaaux.c)

### generate the dplasma headers for all possible precisions
set(HEADERS
  dplasma_z.h
)
precisions_rules(generated_headers "+;z;c;d;s" "${HEADERS}") 

### Generate .c files from .jdf
set(OLDJDF 
  ${CMAKE_CURRENT_SOURCE_DIR}/TSQR.jdf
)
jdf_rules(generated_files "${OLDJDF}")

### Generate .c files from .jdf for all required precisions
set(JDF
  zgemm_NN.jdf zgemm_TN.jdf zgemm_NT.jdf zgemm_TT.jdf
  ztrsm_LLN.jdf ztrsm_LLT.jdf ztrsm_LUN.jdf ztrsm_LUT.jdf ztrsm_RLN.jdf ztrsm_RLT.jdf ztrsm_RUN.jdf ztrsm_RUT.jdf
  ztrmm_LLN.jdf ztrmm_LLT.jdf ztrmm_LUN.jdf ztrmm_LUT.jdf ztrmm_RLN.jdf ztrmm_RLT.jdf ztrmm_RUN.jdf ztrmm_RUT.jdf
  zpotrf_rl.jdf zpotrf_ll.jdf 
  zpotrf_Url.jdf zpotrf_Lrl.jdf 
  zgetrf.jdf zgetrf_sd.jdf ztrsmpl.jdf
  zgeqrf.jdf
  zgelqf.jdf
  zherbt_L.jdf
)
precisions_rules(generated_jdf "+;${DPLASMA_PRECISIONS}" "${JDF}")
jdf_rules(generated_files "${generated_jdf}")

### Generate the dplasma wrappers for all required precisions 
set(SOURCES
  # Level 3 Blas
  zgemm_wrapper.c 
  ztrsm_wrapper.c 
  ztrmm_wrapper.c 
  ztrsmpl_wrapper.c   
  # Lapack
  zpotrf_wrapper.c 
  zpotrs_wrapper.c 
  zposv_wrapper.c 
  zgetrf_wrapper.c 
  zgetrs_wrapper.c 
  zgesv_wrapper.c 
  zgeqrf_wrapper.c
  zgelqf_wrapper.c
  # ???
  zherbt_wrapper.c
)
precisions_rules(generated_files "+;${DPLASMA_PRECISIONS}" "${SOURCES}") 

### Generate the lib 
if (MPI_FOUND)
  add_library(dplasma-mpi
    ${generated_headers}
    ${generated_files} 
    ${EXTRA_SOURCES})
  set_target_properties(dplasma-mpi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  install(TARGETS dplasma-mpi ARCHIVE DESTINATION lib)
else (MPI_FOUND)
  add_library(dplasma 
    ${generated_headers}
    ${generated_files}
    ${EXTRA_SOURCES})
# set_target_properties(dplasma PROPERTIES COMPILE_FLAGS "-DADD_")
  install(TARGETS dplasma ARCHIVE DESTINATION lib)
endif (MPI_FOUND)
install(FILES dplasma.h DESTINATION include)
foreach(generated_header ${generated_headers})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${generated_header} DESTINATION include)
endforeach()
