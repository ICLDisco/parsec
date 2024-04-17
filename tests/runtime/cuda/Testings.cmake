if(PARSEC_HAVE_CUDA)
  parsec_addtest_cmd(runtime/cuda/get_best_device:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/testing_get_best_device -N 400 -t 20 -g 4 -- --mca device_show_statistics 1)
  if(TARGET nvlink)
    parsec_addtest_cmd(runtime/cuda/nvlink:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/nvlink --mca device_cuda_enabled 2 --mca device_show_statistics 1)
  endif()
  if(TARGET stress)
    parsec_addtest_cmd(runtime/cuda/stress:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/stress --mca device_cuda_enabled 2 --mca device_show_statistics 1)
  endif()
  if(TARGET stage)
    parsec_addtest_cmd(runtime/cuda/stage:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/stage --mca device_cuda_enabled 2 --mca device_show_statistics 1)
  endif()
endif()