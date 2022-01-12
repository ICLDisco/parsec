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

  # Testing for geting best device
  parsec_addtest_executable(C testing_get_best_device SOURCES "testing_get_best_device.c")
  target_include_directories(testing_get_best_device PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
  target_ptg_sources(testing_get_best_device PRIVATE "get_best_device_check.jdf")
  parsec_addtest_cmd(get_best_device_shm ${SHM_TEST_CMD_LIST} ./testing_get_best_device -N 400 -t 20 -g 3) 

endif(PARSEC_HAVE_CUDA)
