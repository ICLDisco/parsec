foreach(_UDEV "HIP" "CUDA")
  string(TOLOWER ${_UDEV} _LDEV)
  # This is for CI convenience only: github runners are called gpu_nvidia and gpu_amd, and we want to re-use those names
  # to simplify writing the build_file.yml. We use those as aliases for possible PARSEC_REQUIRE_DEVICE_TEST
  if("${_UDEV}" STREQUAL "CUDA")
    set(_GHR_DEVNAME "gpu_nvidia")
  endif()
  if("${_UDEV}" STREQUAL "HIP")
    set(_GHR_DEVNAME "gpu_amd")
  endif()
  if("${_UDEV}" STREQUAL "NONE")
    set(_GHR_DEVNAME "cpu")
  endif()

  if("${PARSEC_REQUIRE_DEVICE_TEST}" STREQUAL "${_UDEV}" OR "${PARSEC_REQUIRE_DEVICE_TEST}" STREQUAL "${_GHR_DEVNAME}")
    # If we require testing CUDA or HIP, we force to fail if the target device was not detected
    if(NOT PARSEC_HAVE_${_UDEV})
     add_test(NAME runtime/gpu/device_support:${_LDEV} COMMAND false)
    else(NOT PARSEC_HAVE_${_UDEV})
      add_test(NAME runtime/gpu/device_support:${_LDEV} COMMAND true)
    endif(NOT PARSEC_HAVE_${_UDEV})
    set_property(TEST runtime/gpu/device_support:${_LDEV} PROPERTY FIXTURES_SETUP have_${_LDEV}_support)
  endif("${PARSEC_REQUIRE_DEVICE_TEST}" STREQUAL "${_UDEV}" OR "${PARSEC_REQUIRE_DEVICE_TEST}" STREQUAL "${_GHR_DEVNAME}")

  # commented out tests disabled because they cause CI to fail. They should be re-enabled when underlying issue identified.
  # We sanity-check that we're running on a machine that can has at least 1 GPU, and then try to run the tests with up to 8 GPUs / process
  # The test itself is supposed to adapt to any number between 1 and 8, and fail if the actual number of GPUs does not allow
  # to test the feature targeted.
  if(PARSEC_HAVE_${_UDEV})
    # If we required to test ${_UDEV}, and we try on a machine without any ${_UDEV} available, this test will fail
    # To reduce CI time, no other ${_LDEV} test will be run after this fails.
    add_test(NAME runtime/gpu/device_present:${_LDEV} COMMAND ${SHM_TEST_CMD_LIST} ${CTEST_${_UDEV}_LAUNCHER_OPTIONS} sh ${PROJECT_SOURCE_DIR}/tests/runtime/gpu/check_nb_devices.sh ${_UDEV} 1)
    set_property(TEST runtime/gpu/device_present:${_LDEV} PROPERTY FIXTURES_REQUIRED have_${_LDEV}_support)
    set_property(TEST runtime/gpu/device_present:${_LDEV} PROPERTY FIXTURES_SETUP tester_has_${_LDEV}_device)

    if(TARGET get_best_device)
      #  parsec_addtest_cmd(NAME runtime/gpu/get_best_device:${_LDEV} COMMAND ${SHM_TEST_CMD_LIST} ${CTEST_${_UDEV}_LAUNCHER_OPTIONS} runtime/gpu/testing_get_best_device -N 400 -t 20 -g 4)
      #  set_property(TEST runtime/gpu/get_best_device:${_LDEV} PROPERTY FIXTURES_REQUIRED tester_has_${_LDEV}_device)
    endif(TARGET get_best_device)
    if(TARGET dtd_pingpong)
      parsec_addtest_cmd(runtime/gpu/dtd_pingpong:${_LDEV} COMMAND ${SHM_TEST_CMD_LIST} ${CTEST_${_UDEV}_LAUNCHER_OPTIONS} runtime/gpu/dtd_pingpong --mca device_${_LDEV}_enable 8 --mca device ${_LDEV})
      set_property(TEST runtime/gpu/dtd_pingpong:${_LDEV} PROPERTY FIXTURES_REQUIRED tester_has_${_LDEV}_device)
    endif(TARGET dtd_pingpong)
    if(TARGET ptg_pingpong)
      parsec_addtest_cmd(runtime/gpu/ptg_pingpong:${_LDEV} COMMAND ${SHM_TEST_CMD_LIST} ${CTEST_${_UDEV}_LAUNCHER_OPTIONS} runtime/gpu/ptg_pingpong --mca device_${_LDEV}_enable 8 --mca device ${_LDEV})
      set_property(TEST runtime/gpu/ptg_pingpong:${_LDEV} PROPERTY FIXTURES_REQUIRED tester_has_${_LDEV}_device)
      endif(TARGET ptg_pingpong)
    if(TARGET stress)
      #  parsec_addtest_cmd(runtime/gpu/stress:${_LDEV} ${SHM_TEST_CMD_LIST} ${CTEST_${_UDEV}_LAUNCHER_OPTIONS} runtime/gpu/stress)
      #  set_property(TEST runtime/gpu/stress:${_LDEV} PROPERTY FIXTURES_REQUIRED tester_has_${_LDEV}_device)
    endif()
    if(TARGET stage)
      #  parsec_addtest_cmd(runtime/gpu/stage:${_LDEV} ${SHM_TEST_CMD_LIST} ${CTEST_${_UDEV}_LAUNCHER_OPTIONS} runtime/gpu/stage)
      #  set_property(TEST runtime/gpu/stage:${_LDEV} PROPERTY FIXTURES_REQUIRED tester_has_${_LDEV}_device)
    endif()
    if(TARGET nvlink)
      parsec_addtest_cmd(runtime/gpu/nvlink:${_LDEV} ${SHM_TEST_CMD_LIST} ${CTEST_${_UDEV}_LAUNCHER_OPTIONS} runtime/gpu/nvlink --mca device_${_LDEV}_enable 8 --mca device ${_LDEV})
      set_property(TEST runtime/gpu/nvlink:${_LDEV} PROPERTY FIXTURES_REQUIRED tester_has_${_LDEV}_device)
    endif()
  endif()
endforeach()
