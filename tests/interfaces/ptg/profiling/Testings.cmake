find_package (Python COMPONENTS Interpreter)

if(PARSEC_PROF_TRACE AND Python_FOUND)
  execute_process(COMMAND ${Python_EXECUTABLE} -c
                          "import sysconfig; import sys; print('{}-{}.{}'.format(sysconfig.get_platform(),sys.version_info[0],sys.version_info[1]), end='')"
                  OUTPUT_VARIABLE SYSCONF)
  add_test(generate_async_profile ${SHM_TEST_CMD_LIST} ./async 1000 -- --mca profile_filename async  --mca mca_pins task_profiler)

  add_test(generate_hdf5_file ${SHM_TEST_CMD_LIST}
              ${Python_EXECUTABLE}
              ${CMAKE_BINARY_DIR}/tools/profiling/python/examples/profile2h5.py async-0.prof --output=async.h5)
  set_property(TEST generate_hdf5_file APPEND PROPERTY DEPENDS generate_async_profile)
  set_property(TEST generate_hdf5_file APPEND PROPERTY ENVIRONMENT
               LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/tools/profiling/python/build/temp.${SYSCONF}:$ENV{LD_LIBRARY_PATH})
  set_property(TEST generate_hdf5_file APPEND PROPERTY ENVIRONMENT
               PYTHONPATH=${CMAKE_BINARY_DIR}/tools/profiling/python/build/lib.${SYSCONF}/:$ENV{PYTHONPATH})

  add_test(check_hdf5_file ${SHM_TEST_CMD_LIST} ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check-profile.py async.h5)
  set_property(TEST check_hdf5_file APPEND PROPERTY DEPENDS generate_hdf5_file)

  add_test(cleanup_profile_files ${SHM_TEST_CMD_LIST} rm -f async-0.prof async.h5)
endif(PARSEC_PROF_TRACE AND Python_FOUND)
