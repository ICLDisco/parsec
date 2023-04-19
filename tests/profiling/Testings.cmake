find_package (Python COMPONENTS Interpreter)
if(Python_FOUND AND PARSEC_PYTHON_TOOLS AND PARSEC_PROF_TRACE AND MPI_C_FOUND)
  # BW test
  parsec_addtest_cmd(profiling/bw_generate_prof:mp ${MPI_TEST_CMD_LIST} 2 apps/pingpong/bw_test -n 10 -f 10 -l 2097152 -- --mca profile_filename bw  --mca mca_pins task_profiler)
  set_property(TEST profiling/bw_generate_prof:mp PROPERTY FIXTURES_SETUP bw_prof_files)

  set(TMPPYTHONPATH "${PROJECT_BINARY_DIR}/tools/profiling/python/python.test/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages")
  parsec_addtest_cmd(profiling/bw_generate_hdf5 ${SHM_TEST_CMD_LIST}
    ${Python_EXECUTABLE}
    ${PROJECT_BINARY_DIR}/tools/profiling/python/profile2h5.py --output=bw.h5 bw-0.prof bw-1.prof)
  set_property(TEST profiling/bw_generate_hdf5 APPEND PROPERTY ENVIRONMENT
    PYTHONPATH=${TMPPYTHONPATH}/:$ENV{PYTHONPATH})
  set_property(TEST profiling/bw_generate_hdf5 PROPERTY FIXTURES_REQUIRED bw_prof_files)
  set_property(TEST profiling/bw_generate_hdf5 PROPERTY FIXTURES_SETUP bw_h5_files)

  parsec_addtest_cmd(profiling/bw_check_hdf5 ${SHM_TEST_CMD_LIST} ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/profiling/check-comms.py)
  set_property(TEST profiling/bw_check_hdf5 PROPERTY FIXTURES_REQUIRED bw_h5_files)

  parsec_addtest_cmd(profiling/bw_cleanup_files ${SHM_TEST_CMD_LIST} rm -f bw-0.prof bw-1.prof bw.h5)
  set_property(TEST profiling/bw_cleanup_files PROPERTY FIXTURES_CLEANUP bw_prof_files;bw_h5_files)

  # ASYNC test
  parsec_addtest_cmd(profiling/async_generate_prof ${SHM_TEST_CMD_LIST} profiling/async 100 -- --mca profile_filename async  --mca mca_pins task_profiler)
  set_property(TEST profiling/async_generate_prof PROPERTY FIXTURES_SETUP async_prof_files)

  parsec_addtest_cmd(profiling/async_generate_hdf5 ${SHM_TEST_CMD_LIST}
                     ${Python_EXECUTABLE}
                     ${PROJECT_BINARY_DIR}/tools/profiling/python/profile2h5.py --output=async.h5 async-0.prof)
  set_property(TEST profiling/async_generate_hdf5 APPEND PROPERTY ENVIRONMENT
    PYTHONPATH=${TMPPYTHONPATH}/:$ENV{PYTHONPATH})
  set_property(TEST profiling/async_generate_hdf5 PROPERTY FIXTURES_REQUIRED async_prof_files)
  set_property(TEST profiling/async_generate_hdf5 PROPERTY FIXTURES_SETUP async_h5_files)

  parsec_addtest_cmd(profiling/async_check_hdf5 ${SHM_TEST_CMD_LIST} ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/profiling/check-async.py async.h5)
  set_property(TEST profiling/async_check_hdf5 PROPERTY FIXTURES_REQUIRED async_h5_files)

  parsec_addtest_cmd(profiling/async_cleanup_files ${SHM_TEST_CMD_LIST} rm -f async-0.prof async.h5)
  set_property(TEST profiling/async_cleanup_files PROPERTY FIXTURES_CLEANUP async_prof_files;async_h5_files)

  if(PARSEC_PROF_GRAPHER)
    # BW with DOT test; do not use the same file names as BW tests to avoid requiring serialization of tests (RESOURCE_LOCK)
    parsec_addtest_cmd(profiling/bwd_generate_prof_and_dot:mp ${MPI_TEST_CMD_LIST} 2 apps/pingpong/bw_test -n 3 -f 2 -l 2097152 -- --mca profile_filename bwd  --mca mca_pins task_profiler --mca profile_dot bwd)
    set_property(TEST profiling/bwd_generate_prof_and_dot:mp PROPERTY FIXTURES_SETUP bwd_prof_and_dot_files)

    parsec_addtest_cmd(profiling/bwd_generate_hdf5 ${SHM_TEST_CMD_LIST}
                       ${Python_EXECUTABLE}
                       ${PROJECT_BINARY_DIR}/tools/profiling/python/profile2h5.py --output=bwd.h5 bwd-0.prof bwd-1.prof)
    set_property(TEST profiling/bwd_generate_hdf5 APPEND PROPERTY ENVIRONMENT
                 PYTHONPATH=${TMPPYTHONPATH}/:$ENV{PYTHONPATH})
    set_property(TEST profiling/bwd_generate_hdf5 PROPERTY FIXTURES_REQUIRED bwd_prof_and_dot_files)
    set_property(TEST profiling/bwd_generate_hdf5 PROPERTY FIXTURES_SETUP bwd_prof_and_dot_h5_files)

    parsec_addtest_cmd(profiling/bwd_check_hdf5_and_dot ${SHM_TEST_CMD_LIST} ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/profiling/python/examples/example-DAG-and-Trace.py --dot bwd-0.dot --dot bwd-1.dot --h5 bwd.h5)
    set_property(TEST profiling/bwd_check_hdf5_and_dot PROPERTY FIXTURES_REQUIRED bwd_prof_and_dot_h5_files)

    parsec_addtest_cmd(profiling/bwd_cleanup_files ${SHM_TEST_CMD_LIST} rm -f bwd-0.prof bwd-1.prof bwd.h5 bwd-0.dot bwd-1.dot)
    set_property(TEST profiling/bwd_cleanup_files PROPERTY FIXTURES_CLEANUP bwd_prof_and_dot_files;bwd_prof_and_dot_h5_files)
  endif(PARSEC_PROF_GRAPHER)
endif(Python_FOUND AND PARSEC_PYTHON_TOOLS AND PARSEC_PROF_TRACE AND MPI_C_FOUND)
