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

  # Each task handles one 32x32 double tile (8192 B). RW inputs and forced
  # descriptor outputs must both be required and transferred.
  set(device_stats_cuda_3 "[|]  Dev [ ]*[0-9]+ [|][ ]*3 [|][ ]*[0-9]+[.][0-9]+ [|][ ]*24[.]00KB [|][ ]*24[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[.]00 B[(][ ]*0[.]00[)][ ]*[|][ ]*24[.]00KB [|][ ]*24[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[ ]*[|] cuda")
  set(device_stats_cuda_5 "[|]  Dev [ ]*[0-9]+ [|][ ]*5 [|][ ]*[0-9]+[.][0-9]+ [|][ ]*40[.]00KB [|][ ]*40[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[.]00 B[(][ ]*0[.]00[)][ ]*[|][ ]*40[.]00KB [|][ ]*40[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[ ]*[|] cuda")
  set(device_stats_cuda_8 "[|]  Dev [ ]*[0-9]+ [|][ ]*8 [|][ ]*[0-9]+[.][0-9]+ [|][ ]*64[.]00KB [|][ ]*64[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[.]00 B[(][ ]*0[.]00[)][ ]*[|][ ]*64[.]00KB [|][ ]*64[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[ ]*[|] cuda")
  set(device_stats_cuda_10 "[|]  Dev [ ]*[0-9]+ [|][ ]*10 [|][ ]*[0-9]+[.][0-9]+ [|][ ]*80[.]00KB [|][ ]*80[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[.]00 B[(][ ]*0[.]00[)][ ]*[|][ ]*80[.]00KB [|][ ]*80[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[ ]*[|] cuda")

  parsec_addtest_cmd(runtime/cuda/device_show_statistics:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/device_show_statistics --scenario explicit)
  set_tests_properties(runtime/cuda/device_show_statistics:gpu PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_cuda_3}"
    FAIL_REGULAR_EXPRESSION "Full transfer matrix")

  parsec_addtest_cmd(runtime/cuda/device_show_statistics/default:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/device_show_statistics --scenario default)
  set_tests_properties(runtime/cuda/device_show_statistics/default:gpu PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_cuda_10}")

  parsec_addtest_cmd(runtime/cuda/device_show_statistics/default_matrix:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/device_show_statistics --scenario default)
  set_tests_properties(runtime/cuda/device_show_statistics/default_matrix:gpu PROPERTIES
    PASS_REGULAR_EXPRESSION "Full transfer matrix")

  parsec_addtest_cmd(runtime/cuda/device_show_statistics/unfinished:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/device_show_statistics --scenario unfinished)
  set_tests_properties(runtime/cuda/device_show_statistics/unfinished:gpu PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_cuda_8}"
    FAIL_REGULAR_EXPRESSION "Full transfer matrix")

  parsec_addtest_cmd(runtime/cuda/device_show_statistics/overlap:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/device_show_statistics --scenario overlap)
  set_tests_properties(runtime/cuda/device_show_statistics/overlap:gpu PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_cuda_8}"
    FAIL_REGULAR_EXPRESSION "Full transfer matrix")

  parsec_addtest_cmd(runtime/cuda/device_show_statistics/overlap_inner:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/device_show_statistics --scenario overlap)
  set_tests_properties(runtime/cuda/device_show_statistics/overlap_inner:gpu PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_cuda_5}"
    FAIL_REGULAR_EXPRESSION "Full transfer matrix")

  parsec_addtest_cmd(runtime/cuda/device_show_statistics/reset:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/device_show_statistics --scenario reset)
  set_tests_properties(runtime/cuda/device_show_statistics/reset:gpu PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_cuda_5}"
    FAIL_REGULAR_EXPRESSION "Full transfer matrix")

  parsec_addtest_cmd(runtime/cuda/device_show_statistics/end_without_start:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/device_show_statistics --scenario end-without-start)
  set_tests_properties(runtime/cuda/device_show_statistics/end_without_start:gpu PROPERTIES
    PASS_REGULAR_EXPRESSION "parsec_device_show_statistics_end called with an invalid interval")

  parsec_addtest_cmd(runtime/cuda/device_show_statistics/end_without_start_default:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/device_show_statistics --scenario end-without-start)
  set_tests_properties(runtime/cuda/device_show_statistics/end_without_start_default:gpu PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_cuda_3}")

  parsec_addtest_cmd(runtime/cuda/device_show_statistics/disabled:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/device_show_statistics --scenario disabled)
  set_tests_properties(runtime/cuda/device_show_statistics/disabled:gpu PROPERTIES
    FAIL_REGULAR_EXPRESSION "#[ ]KERNEL;Full transfer matrix")

  if(TARGET nvlink)
    parsec_addtest_cmd(runtime/cuda/nvlink:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/nvlink --mca device_cuda_enabled 2 --mca device_show_statistics 1)
  endif()
  if(TARGET stress)
    parsec_addtest_cmd(runtime/cuda/stress:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} runtime/cuda/stress --mca device_cuda_enabled 2 --mca device_show_statistics 1)
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
endif()

if(PARSEC_HAVE_HIP)
  # Each task handles one 32x32 double tile (8192 B). RW inputs and forced
  # descriptor outputs must both be required and transferred.
  set(device_stats_hip_3 "[|]  Dev [ ]*[0-9]+ [|][ ]*3 [|][ ]*[0-9]+[.][0-9]+ [|][ ]*24[.]00KB [|][ ]*24[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[.]00 B[(][ ]*0[.]00[)][ ]*[|][ ]*24[.]00KB [|][ ]*24[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[ ]*[|] hip")
  set(device_stats_hip_5 "[|]  Dev [ ]*[0-9]+ [|][ ]*5 [|][ ]*[0-9]+[.][0-9]+ [|][ ]*40[.]00KB [|][ ]*40[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[.]00 B[(][ ]*0[.]00[)][ ]*[|][ ]*40[.]00KB [|][ ]*40[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[ ]*[|] hip")
  set(device_stats_hip_8 "[|]  Dev [ ]*[0-9]+ [|][ ]*8 [|][ ]*[0-9]+[.][0-9]+ [|][ ]*64[.]00KB [|][ ]*64[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[.]00 B[(][ ]*0[.]00[)][ ]*[|][ ]*64[.]00KB [|][ ]*64[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[ ]*[|] hip")
  set(device_stats_hip_10 "[|]  Dev [ ]*[0-9]+ [|][ ]*10 [|][ ]*[0-9]+[.][0-9]+ [|][ ]*80[.]00KB [|][ ]*80[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[.]00 B[(][ ]*0[.]00[)][ ]*[|][ ]*80[.]00KB [|][ ]*80[.]00KB[(]100[.]00[)][ ]*[|][ ]*0[ ]*[|] hip")

  parsec_addtest_cmd(runtime/cuda/device_show_statistics:hip ${SHM_TEST_CMD_LIST} ${CTEST_HIP_LAUNCHER_OPTIONS} runtime/cuda/device_show_statistics --scenario explicit)
  set_tests_properties(runtime/cuda/device_show_statistics:hip PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_hip_3}"
    FAIL_REGULAR_EXPRESSION "Full transfer matrix")

  parsec_addtest_cmd(runtime/cuda/device_show_statistics/default:hip ${SHM_TEST_CMD_LIST} ${CTEST_HIP_LAUNCHER_OPTIONS} runtime/cuda/device_show_statistics --scenario default)
  set_tests_properties(runtime/cuda/device_show_statistics/default:hip PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_hip_10}")

  parsec_addtest_cmd(runtime/cuda/device_show_statistics/default_matrix:hip ${SHM_TEST_CMD_LIST} ${CTEST_HIP_LAUNCHER_OPTIONS} runtime/cuda/device_show_statistics --scenario default)
  set_tests_properties(runtime/cuda/device_show_statistics/default_matrix:hip PROPERTIES
    PASS_REGULAR_EXPRESSION "Full transfer matrix")

  parsec_addtest_cmd(runtime/cuda/device_show_statistics/unfinished:hip ${SHM_TEST_CMD_LIST} ${CTEST_HIP_LAUNCHER_OPTIONS} runtime/cuda/device_show_statistics --scenario unfinished)
  set_tests_properties(runtime/cuda/device_show_statistics/unfinished:hip PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_hip_8}"
    FAIL_REGULAR_EXPRESSION "Full transfer matrix")

  parsec_addtest_cmd(runtime/cuda/device_show_statistics/overlap:hip ${SHM_TEST_CMD_LIST} ${CTEST_HIP_LAUNCHER_OPTIONS} runtime/cuda/device_show_statistics --scenario overlap)
  set_tests_properties(runtime/cuda/device_show_statistics/overlap:hip PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_hip_8}"
    FAIL_REGULAR_EXPRESSION "Full transfer matrix")

  parsec_addtest_cmd(runtime/cuda/device_show_statistics/overlap_inner:hip ${SHM_TEST_CMD_LIST} ${CTEST_HIP_LAUNCHER_OPTIONS} runtime/cuda/device_show_statistics --scenario overlap)
  set_tests_properties(runtime/cuda/device_show_statistics/overlap_inner:hip PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_hip_5}"
    FAIL_REGULAR_EXPRESSION "Full transfer matrix")

  parsec_addtest_cmd(runtime/cuda/device_show_statistics/reset:hip ${SHM_TEST_CMD_LIST} ${CTEST_HIP_LAUNCHER_OPTIONS} runtime/cuda/device_show_statistics --scenario reset)
  set_tests_properties(runtime/cuda/device_show_statistics/reset:hip PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_hip_5}"
    FAIL_REGULAR_EXPRESSION "Full transfer matrix")

  parsec_addtest_cmd(runtime/cuda/device_show_statistics/end_without_start:hip ${SHM_TEST_CMD_LIST} ${CTEST_HIP_LAUNCHER_OPTIONS} runtime/cuda/device_show_statistics --scenario end-without-start)
  set_tests_properties(runtime/cuda/device_show_statistics/end_without_start:hip PROPERTIES
    PASS_REGULAR_EXPRESSION "parsec_device_show_statistics_end called with an invalid interval")

  parsec_addtest_cmd(runtime/cuda/device_show_statistics/end_without_start_default:hip ${SHM_TEST_CMD_LIST} ${CTEST_HIP_LAUNCHER_OPTIONS} runtime/cuda/device_show_statistics --scenario end-without-start)
  set_tests_properties(runtime/cuda/device_show_statistics/end_without_start_default:hip PROPERTIES
    PASS_REGULAR_EXPRESSION "${device_stats_hip_3}")

  parsec_addtest_cmd(runtime/cuda/device_show_statistics/disabled:hip ${SHM_TEST_CMD_LIST} ${CTEST_HIP_LAUNCHER_OPTIONS} runtime/cuda/device_show_statistics --scenario disabled)
  set_tests_properties(runtime/cuda/device_show_statistics/disabled:hip PROPERTIES
    FAIL_REGULAR_EXPRESSION "#[ ]KERNEL;Full transfer matrix")

  if(TARGET ptg_pingpong)
    parsec_addtest_cmd(runtime/cuda/ptg_pingpong:hip ${SHM_TEST_CMD_LIST} ${CTEST_HIP_LAUNCHER_OPTIONS} runtime/cuda/ptg_pingpong --mca device_hip_enabled 2 --mca device_show_statistics 1)
  endif()
  if(TARGET dtd_pingpong)
    parsec_addtest_cmd(runtime/cuda/dtd_pingpong:hip ${SHM_TEST_CMD_LIST} ${CTEST_HIP_LAUNCHER_OPTIONS} runtime/cuda/dtd_pingpong --mca device_hip_enabled 2 --mca device_show_statistics 1)
  endif()
endif()
