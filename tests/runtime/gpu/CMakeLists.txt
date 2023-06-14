if(PARSEC_HAVE_CUDA)
  include(ParsecCompilePTG)
  if( NOT TARGET CUDA::cublas )
    message(WARNING "CUBLAS library not automatically detected when CUDA was found -- not enabling nvlink test")
  else( NOT TARGET CUDA::cublas )

    parsec_addtest_executable(C nvlink SOURCES nvlink_main.c nvlink_wrapper.c)
    target_include_directories(nvlink PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
    target_ptg_sources(nvlink PRIVATE "nvlink.jdf")
    target_link_libraries(nvlink PRIVATE CUDA::cublas)

    parsec_addtest_executable(C stress SOURCES stress_main.c stress_wrapper.c)
    target_include_directories(stress PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
    target_ptg_sources(stress PRIVATE "stress.jdf")
    target_link_libraries(stress PRIVATE CUDA::cublas)

    parsec_addtest_executable(C stage SOURCES stage_main.c)
    target_include_directories(stage PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
    target_ptg_sources(stage PRIVATE "stage_custom.jdf")
    target_link_libraries(stage PRIVATE CUDA::cublas)

  endif( NOT TARGET CUDA::cublas )

  # Testing for geting best device
  parsec_addtest_executable(C testing_get_best_device SOURCES "testing_get_best_device.c")
  target_include_directories(testing_get_best_device PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
  target_ptg_sources(testing_get_best_device PRIVATE "get_best_device_check.jdf")

  if(CMAKE_CUDA_COMPILER)
    set_source_files_properties(ping_kernel.cu PROPERTIES LANGUAGE CUDA)
    parsec_addtest_executable(C dtd_pingpong SOURCES dtd_pingpong.c)
    target_sources(dtd_pingpong PRIVATE ping_kernel.cu)

    parsec_addtest_executable(C ptg_pingpong SOURCES ping_kernel.cu)
    target_ptg_sources(ptg_pingpong PRIVATE "ptg_pingpong.jdf")
  endif(CMAKE_CUDA_COMPILER)
endif(PARSEC_HAVE_CUDA)

if(PARSEC_HAVE_HIP)
  if(CMAKE_HIP_COMPILER)
    include(ParsecCompilePTG)
    set_source_files_properties(ping_kernel.hip.c PROPERTIES LANGUAGE HIP)
    parsec_addtest_executable(C dtd_pingpong SOURCES dtd_pingpong.c ping_kernel.hip.c)

    parsec_addtest_executable(C ptg_pingpong SOURCES ping_kernel.hip.c)
    target_ptg_sources(ptg_pingpong PRIVATE "ptg_pingpong.jdf")
  endif(CMAKE_HIP_COMPILER)
endif(PARSEC_HAVE_HIP)
