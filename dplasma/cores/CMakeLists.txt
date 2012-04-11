include(RulesPrecisions)
# reset variables
set(generated_files "")
set(generated_headers "")

### generate the dplasma_cores headers for all possible precisions
precisions_rules_py(generated_headers
                 "dplasma_zcores.h"
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
)
precisions_rules_py(generated_files
                 "${SOURCES}"
                 PRECISIONS "${DPLASMA_PRECISIONS}")

### Generate the CUDA kernels if necessary
if(CUDA_FOUND)
    set(CUDA_BUILD_CUBIN ON)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_NVCC_FLAGS -maxrregcount 32 -arch sm_11 -DCUDA_SM_VERSION=11)
    cuda_add_library(sgemm-sm_11 cuda_sgemm_kernel.cu)
    install(TARGETS sgemm-sm_11 ARCHIVE DESTINATION lib)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sgemm-sm_11_generated_cuda_sgemm_kernel.cu.o.cubin.txt
      DESTINATION lib
      RENAME sgemm-sm_11.cubin)

    set(CUDA_NVCC_FLAGS -maxrregcount 32 -arch sm_13 -DCUDA_SM_VERSION=13)
    cuda_add_library(sgemm-sm_13 cuda_sgemm_kernel.cu)
    install(TARGETS sgemm-sm_13 ARCHIVE DESTINATION lib)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sgemm-sm_13_generated_cuda_sgemm_kernel.cu.o.cubin.txt
      DESTINATION lib
      RENAME sgemm-sm_13.cubin)

    set(CUDA_NVCC_FLAGS -arch sm_20 -DCUDA_SM_VERSION=20)
    cuda_add_library(sgemm-sm_20 cuda_sgemm_kernelf.cu)
    install(TARGETS sgemm-sm_20 ARCHIVE DESTINATION lib)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sgemm-sm_20_generated_cuda_sgemm_kernelf.cu.o.cubin.txt
      DESTINATION lib
      RENAME sgemm-sm_20.cubin)

    cuda_build_clean_target()
    set(GPU_KERNEL_SOURCES cuda_sgemm.c cuda_stsmqr.c)
else()
  set(GPU_KERNEL_SOURCES "")
endif()

### Generate the lib
add_library(dplasma_cores
  ${generated_headers}
  ${generated_files}
  ${GPU_KERNEL_SOURCES})
install(TARGETS dplasma_cores ARCHIVE DESTINATION lib)

foreach(generated_header ${generated_headers})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${generated_header} DESTINATION include/cores)
endforeach()
