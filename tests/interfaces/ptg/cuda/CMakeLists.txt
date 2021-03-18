if(PARSEC_HAVE_CUDA)
  if( NOT TARGET CUDA::cublas )
    message(WARNING "CUBLAS library not automatically detected when CUDA was found -- not enabling nvlink test")
  else( NOT TARGET CUDA::cublas )
    include(ParsecCompilePTG)

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
endif(PARSEC_HAVE_CUDA)
