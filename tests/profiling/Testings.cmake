find_package (Python COMPONENTS Interpreter)
if(Python_FOUND AND PARSEC_PYTHON_TOOLS AND PARSEC_PROF_TRACE AND MPI_C_FOUND)
  parsec_addtest_cmd(profiling/generate_profile_bw:mp ${MPI_TEST_CMD_LIST} 2 apps/pingpong/bw_test -n 10 -f 10 -l 2097152 -- --parsec profile_filename bw  --parsec mca_pins task_profiler)

  set_property(TEST profiling/generate_profile_bw:mp PROPERTY RESOURCE_LOCK bw_file)
  set_property(TEST profiling/generate_profile_bw:mp PROPERTY FIXTURES_SETUP bw_file_base)

  set(TMPPYTHONPATH "${PROJECT_BINARY_DIR}/tools/profiling/python/python.test/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages")
  parsec_addtest_cmd(profiling/generate_hdf5 ${SHM_TEST_CMD_LIST}
    ${Python_EXECUTABLE}
    ${PROJECT_BINARY_DIR}/tools/profiling/python/profile2h5.py --output=bw.h5 bw-0.prof bw-1.prof)
  set_property(TEST profiling/generate_hdf5 APPEND PROPERTY DEPENDS profiling/generate_profile_bw:mp)
  set_property(TEST profiling/generate_hdf5 APPEND PROPERTY ENVIRONMENT
    PYTHONPATH=${TMPPYTHONPATH}/:$ENV{PYTHONPATH})
  set_property(TEST profiling/generate_hdf5 PROPERTY RESOURCE_LOCK bw_file)
  set_property(TEST profiling/generate_hdf5 PROPERTY FIXTURES_REQUIRED bw_file_base)
  set_property(TEST profiling/generate_hdf5 PROPERTY FIXTURES_SETUP bw_file_base)

  parsec_addtest_cmd(profiling/check_hdf5 ${SHM_TEST_CMD_LIST} ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/profiling/check-comms.py)
  set_property(TEST profiling/check_hdf5 APPEND PROPERTY DEPENDS profiling/generate_hdf5)
  set_property(TEST profiling/generate_hdf5 PROPERTY RESOURCE_LOCK bw_file)
  set_property(TEST profiling/generate_hdf5 PROPERTY FIXTURES_REQUIRED bw_file)

  parsec_addtest_cmd(profiling/generate_profile_async ${SHM_TEST_CMD_LIST} profiling/async 100 -- --parsec profile_filename async  --parsec mca_pins task_profiler)
  set_property(TEST profiling/generate_profile_async PROPERTY RESOURCE_LOCK async_file)
  set_property(TEST profiling/generate_profile_async PROPERTY FIXTURES_SETUP async_file_base)

  parsec_addtest_cmd(profiling/generate_hdf5_async ${SHM_TEST_CMD_LIST}
                     ${Python_EXECUTABLE}
                     ${PROJECT_BINARY_DIR}/tools/profiling/python/profile2h5.py --output=async.h5 async-0.prof)
  set_property(TEST profiling/generate_hdf5_async APPEND PROPERTY DEPENDS profiling/generate_profile_async)
  set_property(TEST profiling/generate_hdf5_async APPEND PROPERTY ENVIRONMENT
    PYTHONPATH=${TMPPYTHONPATH}/:$ENV{PYTHONPATH})
  set_property(TEST profiling/generate_hdf5_async PROPERTY RESOURCE_LOCK async_file)
  set_property(TEST profiling/generate_hdf5_async PROPERTY FIXTURES_REQUIRED async_file_base)
  set_property(TEST profiling/generate_hdf5_async PROPERTY FIXTURES_SETUP async_file)

  parsec_addtest_cmd(profiling/check_hdf5_async ${SHM_TEST_CMD_LIST} ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/profiling/check-async.py async.h5)
    set_property(TEST profiling/check_hdf5_async APPEND PROPERTY DEPENDS profiling/generate_hdf5_async)
  set_property(TEST profiling/check_hdf5_async PROPERTY RESOURCE_LOCK async_file)
  set_property(TEST profiling/check_hdf5_async PROPERTY FIXTURES_REQUIRED async_file)

  #  parsec_addtest_cmd(profiling/cleanup_profile_files ${SHM_TEST_CMD_LIST} rm -f bw-0.prof bw-1.prof bw.h5 async-0.prof async.h5)

  if(PARSEC_PROF_GRAPHER)
    parsec_addtest_cmd(profiling/generate_profile_and_dot_bw:mp ${MPI_TEST_CMD_LIST} 2 apps/pingpong/bw_test -n 3 -f 2 -l 2097152 -- --parsec profile_filename bw  --parsec mca_pins task_profiler --parsec profile_dot bw)
    set_property(TEST profiling/generate_profile_and_dot_bw:mp PROPERTY RESOURCE_LOCK bw_file_and_dot)
    set_property(TEST profiling/generate_profile_and_dot_bw:mp PROPERTY FIXTURES_SETUP bw_file_base_and_dot)

    parsec_addtest_cmd(profiling/generate_hdf5_for_dag_and_dot ${SHM_TEST_CMD_LIST}
            ${Python_EXECUTABLE}
            ${PROJECT_BINARY_DIR}/tools/profiling/python/profile2h5.py --output=bw.h5 bw-0.prof bw-1.prof)
    set_property(TEST profiling/generate_hdf5_for_dag_and_dot APPEND PROPERTY DEPENDS profiling/generate_profile_and_dot_bw:mp)
    set_property(TEST profiling/generate_hdf5_for_dag_and_dot APPEND PROPERTY ENVIRONMENT
                 PYTHONPATH=${TMPPYTHONPATH}/:$ENV{PYTHONPATH})
    set_property(TEST profiling/generate_hdf5_for_dag_and_dot PROPERTY RESOURCE_LOCK bw_file_and_dot)
    set_property(TEST profiling/generate_hdf5_for_dag_and_dot PROPERTY FIXTURES_REQUIRED bw_file_base_and_dot)
    set_property(TEST profiling/generate_hdf5_for_dag_and_dot PROPERTY FIXTURES_SETUP bw_file)

    parsec_addtest_cmd(profiling/check_DAG_and_Trace ${SHM_TEST_CMD_LIST} ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/profiling/python/examples/example-DAG-and-Trace.py --parsec profile_dot bw-0.dot --parsec profile_dot bw-1.dot --h5 bw.h5)
    set_property(TEST profiling/check_DAG_and_Trace APPEND PROPERTY DEPENDS profiling/generate_hdf5_for_dag_and_dot)
    set_property(TEST profiling/check_DAG_and_Trace PROPERTY RESOURCE_LOCK bw_file_and_dot)
    set_property(TEST profiling/check_DAG_and_Trace PROPERTY FIXTURES_REQUIRED bw_file)

    parsec_addtest_cmd(profiling/cleanup_dot_files ${SHM_TEST_CMD_LIST} rm -f bw-0.prof bw-1.prof bw.h5 bw-0.dot bw-1.dot)
  endif(PARSEC_PROF_GRAPHER)
endif(Python_FOUND AND PARSEC_PYTHON_TOOLS AND PARSEC_PROF_TRACE AND MPI_C_FOUND)
