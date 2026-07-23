#
# Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
#
set(PINGPONG_GPU_KERNEL_SOURCES)

if(PARSEC_HAVE_CUDA)
  include(ParsecCompilePTG)
  if( NOT TARGET CUDA::cublas )
    message(WARNING "CUBLAS library not automatically detected when CUDA was found -- not enabling nvlink test")
  else( NOT TARGET CUDA::cublas )

    parsec_addtest_executable(C nvlink SOURCES nvlink_main.c nvlink_wrapper.c)
    target_link_libraries(nvlink PRIVATE tests_runtime_common)
    target_include_directories(nvlink PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
    target_ptg_sources(nvlink PRIVATE "nvlink.jdf")
    target_link_libraries(nvlink PRIVATE CUDA::cublas)

    parsec_addtest_executable(C stress SOURCES stress_main.c stress_wrapper.c)
    target_link_libraries(stress PRIVATE tests_runtime_common)
    target_include_directories(stress PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
    target_ptg_sources(stress PRIVATE "stress.jdf")
    target_link_libraries(stress PRIVATE CUDA::cublas)

    parsec_addtest_executable(C stage SOURCES stage_main.c)
    target_link_libraries(stage PRIVATE tests_runtime_common)
    target_include_directories(stage PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
    target_ptg_sources(stage PRIVATE "stage_custom.jdf")
    target_link_libraries(stage PRIVATE CUDA::cublas)

  endif( NOT TARGET CUDA::cublas )

  # Testing for getting best device
  parsec_addtest_executable(C testing_get_best_device SOURCES "testing_get_best_device.c")
  target_link_libraries(testing_get_best_device PRIVATE tests_runtime_common)
  target_include_directories(testing_get_best_device PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
  target_ptg_sources(testing_get_best_device PRIVATE "get_best_device_check.jdf")

  if(CMAKE_CUDA_COMPILER)
    set_source_files_properties(ping_kernel.cu PROPERTIES LANGUAGE CUDA)
    list(APPEND PINGPONG_GPU_KERNEL_SOURCES ping_kernel.cu)
  endif(CMAKE_CUDA_COMPILER)
  # Testing for communications to and from GPU
  parsec_addtest_executable(C cuda_rtt SOURCES rtt_main.c)
  target_include_directories(cuda_rtt PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
  target_link_libraries(cuda_rtt PRIVATE m)
  target_ptg_sources(cuda_rtt PRIVATE "${PROJECT_SOURCE_DIR}/tests/apps/pingpong/rtt.jdf")
endif(PARSEC_HAVE_CUDA)

if(PARSEC_HAVE_HIP)
  check_language(HIP)
  if(CMAKE_HIP_COMPILER)
    enable_language(HIP)
    include(ParsecCompilePTG)
    set_source_files_properties(ping_kernel.hip.c PROPERTIES LANGUAGE HIP)
    list(APPEND PINGPONG_GPU_KERNEL_SOURCES ping_kernel.hip.c)
  endif(CMAKE_HIP_COMPILER)
endif(PARSEC_HAVE_HIP)

# Build a single pingpong pair and add the kernels for every enabled GPU
# backend, so CUDA+HIP builds do not define the same test targets twice.
if(PINGPONG_GPU_KERNEL_SOURCES)
  parsec_addtest_executable(C dtd_pingpong SOURCES dtd_pingpong.c ${PINGPONG_GPU_KERNEL_SOURCES})
  target_include_directories(dtd_pingpong PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)

  parsec_addtest_executable(C ptg_pingpong SOURCES ${PINGPONG_GPU_KERNEL_SOURCES})
  target_include_directories(ptg_pingpong PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
  target_ptg_sources(ptg_pingpong PRIVATE "ptg_pingpong.jdf")
endif()

parsec_addtest_executable(C zonemalloc SOURCES zonemalloc.c)
