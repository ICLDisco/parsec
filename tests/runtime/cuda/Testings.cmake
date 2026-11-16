if(PARSEC_HAVE_CUDA)
  # Every CUDA test below assumes the kernels this build produced can run on
  # the device it finds. When they cannot, none of them says so: the launches
  # fail, the kernels write nothing, and the tests report wrong results or
  # deadlock instead. Ask the question once, up front, and let the tests that
  # depend on the answer not run at all when it is no.
  if(TARGET cuda_arch_probe)
    parsec_addtest_cmd(runtime/cuda/arch_probe:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/cuda_arch_probe)
    set_tests_properties(runtime/cuda/arch_probe:gpu PROPERTIES FIXTURES_SETUP parsec_cuda_kernels_run)
  endif()
  parsec_addtest_cmd(runtime/cuda/get_best_device:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/testing_get_best_device -N 400 -t 20 -g 4 -- --mca device_show_statistics 1)
  if(TARGET nvlink)
    parsec_addtest_cmd(runtime/cuda/nvlink:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/nvlink --mca device_cuda_enabled 2 --mca device_show_statistics 1)
  endif()
  if(TARGET stress)
    parsec_addtest_cmd(runtime/cuda/stress:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/stress -- --mca device_cuda_enabled 2 --mca device_show_statistics 1)
    # Far less memory than the tasks need, so the device evicts and reuses
    # copies constantly. Nothing else in the suite makes it reclaim memory,
    # where a copy left with readers leaks and teardown reports it.
    parsec_addtest_cmd(runtime/cuda/stress:gpu:capped ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/stress -n 512 -d 60 -- --mca device_cuda_enabled 1 --mca device_cuda_memory_number_of_blocks 32 --mca device_show_statistics 1)
    if(PARSEC_DEBUG OR PARSEC_DEBUG_NOISIER)
      # device_inject_disable is debug-only. It sends a task back out through
      # parsec_device_kernel_cleanout(), the exit that skips the pop stage.
      parsec_addtest_cmd(runtime/cuda/stress:gpu:capped:disable ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/stress -n 512 -d 60 -- --mca device_cuda_enabled 1 --mca device_cuda_memory_number_of_blocks 32 --mca device_inject_disable 7 --mca device_show_statistics 1)
    endif()
  endif()
  if(TARGET stage)
    parsec_addtest_cmd(runtime/cuda/stage:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/stage --mca device_cuda_enabled 2 --mca device_show_statistics 1)
  endif()
  if(TARGET cuda_rtt)
    parsec_addtest_cmd(runtime/cuda/rtt:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/cuda_rtt -g 1 -l 10 -m 4096 -- --mca device_cuda_enabled 1 --mca device_show_statistics 1)
  endif()
  if(TARGET ptg_pingpong)
    parsec_addtest_cmd(runtime/cuda/ptg_pingpong:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/ptg_pingpong --mca device_cuda_enabled 2 --mca device_show_statistics 1)
    set_tests_properties(runtime/cuda/ptg_pingpong:gpu PROPERTIES FIXTURES_REQUIRED parsec_cuda_kernels_run)
  endif()
  if(TARGET dtd_pingpong)
    parsec_addtest_cmd(runtime/cuda/dtd_pingpong:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/dtd_pingpong --mca device_cuda_enabled 2 --mca device_show_statistics 1)
    set_tests_properties(runtime/cuda/dtd_pingpong:gpu PROPERTIES FIXTURES_REQUIRED parsec_cuda_kernels_run)
  endif()
  if(MPI_C_FOUND AND TARGET ptg_pingpong)
    # Eviction with remote dependencies: the comm thread reads copies the
    # device may be reclaiming, and a forwarded output is read by both the
    # receive and the send. Neither happens in a single-process run.
    parsec_addtest_cmd(runtime/cuda/ptg_pingpong:mp:gpu:capped ${MPI_TEST_CMD_LIST} 2 ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/ptg_pingpong --mca device_cuda_enabled 1 --mca device_cuda_memory_number_of_blocks 32 --mca device_show_statistics 1)
    set_tests_properties(runtime/cuda/ptg_pingpong:mp:gpu:capped PROPERTIES FIXTURES_REQUIRED parsec_cuda_kernels_run)
  endif()
endif()

if(PARSEC_HAVE_HIP)
  if(TARGET ptg_pingpong)
    parsec_addtest_cmd(runtime/cuda/ptg_pingpong:hip ${SHM_TEST_CMD_LIST} ${CTEST_HIP_LAUNCHER_OPTIONS} runtime/cuda/ptg_pingpong --mca device_hip_enabled 2 --mca device_show_statistics 1)
  endif()
  if(TARGET dtd_pingpong)
    parsec_addtest_cmd(runtime/cuda/dtd_pingpong:hip ${SHM_TEST_CMD_LIST} ${CTEST_HIP_LAUNCHER_OPTIONS} runtime/cuda/dtd_pingpong --mca device_hip_enabled 2 --mca device_show_statistics 1)
  endif()
endif()
