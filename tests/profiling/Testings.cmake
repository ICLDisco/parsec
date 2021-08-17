find_package (Python COMPONENTS Interpreter)
if(Python_FOUND AND PARSEC_PYTHON_TOOLS AND PARSEC_PROF_TRACE AND MPI_C_FOUND)
  execute_process(COMMAND ${Python_EXECUTABLE} -c
    "from __future__ import print_function; import sysconfig; import sys; print('{}-{}.{}'.format(sysconfig.get_platform(),sys.version_info[0],sys.version_info[1]), end='')"
    OUTPUT_VARIABLE SYSCONF)
  parsec_addtest_cmd(profiling/generate_profile_bw:mp ${MPI_TEST_CMD_LIST} 2 apps/pingpong/bw_test -n 10 -f 10 -l 2097152 -- --mca profile_filename bw  --mca mca_pins task_profiler)

  parsec_addtest_cmd(profiling/generate_hdf5 ${SHM_TEST_CMD_LIST}
    ${Python_EXECUTABLE}
    ${PROJECT_BINARY_DIR}/tools/profiling/python/profile2h5.py --output=bw.h5 bw-0.prof bw-1.prof)
  set_property(TEST profiling/generate_hdf5 APPEND PROPERTY DEPENDS profiling/generate_profile_bw:mp)
  set_property(TEST profiling/generate_hdf5 APPEND PROPERTY ENVIRONMENT
    LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/tools/profiling/python/build/temp.${SYSCONF}:$ENV{LD_LIBRARY_PATH})
  set_property(TEST profiling/generate_hdf5 APPEND PROPERTY ENVIRONMENT
    PYTHONPATH=${CMAKE_BINARY_DIR}/tools/profiling/python/build/lib.${SYSCONF}/:$ENV{PYTHONPATH})

  parsec_addtest_cmd(profiling/check_hdf5 ${SHM_TEST_CMD_LIST} ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/profiling/check-comms.py)
  set_property(TEST profiling/check_hdf5 APPEND PROPERTY DEPENDS profiling/generate_hdf5)

  parsec_addtest_cmd(profiling/generate_profile_async ${SHM_TEST_CMD_LIST} profiling/async 100 -- --mca profile_filename async  --mca mca_pins task_profiler)

  parsec_addtest_cmd(profiling/generate_hdf5_async ${SHM_TEST_CMD_LIST}
                     ${Python_EXECUTABLE}
                     ${PROJECT_BINARY_DIR}/tools/profiling/python/profile2h5.py --output=async.h5 async-0.prof)
  set_property(TEST profiling/generate_hdf5_async APPEND PROPERTY DEPENDS profiling/generate_profile_async)
  set_property(TEST profiling/generate_hdf5_async APPEND PROPERTY ENVIRONMENT
               LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/tools/profiling/python/build/temp.${SYSCONF}:$ENV{LD_LIBRARY_PATH})
  set_property(TEST profiling/generate_hdf5_async APPEND PROPERTY ENVIRONMENT
               PYTHONPATH=${CMAKE_BINARY_DIR}/tools/profiling/python/build/lib.${SYSCONF}/:$ENV{PYTHONPATH})

  parsec_addtest_cmd(profiling/check_hdf5_async ${SHM_TEST_CMD_LIST} ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/profiling/check-async.py async.h5)
    set_property(TEST profiling/check_hdf5_async APPEND PROPERTY DEPENDS profiling/generate_hdf5_async)

  parsec_addtest_cmd(profiling/cleanup_profile_files ${SHM_TEST_CMD_LIST} rm -f bw-0.prof bw-1.prof bw.h5 async-0.prof async.h5)

  if(PARSEC_PROF_GRAPHER)
    parsec_addtest_cmd(profiling/generate_profile_and_dot_bw:mp ${MPI_TEST_CMD_LIST} 2 apps/pingpong/bw_test -n 3 -f 2 -l 2097152 -- --mca profile_filename bw  --mca mca_pins task_profiler --dot bw)

    parsec_addtest_cmd(profiling/generate_hdf5_for_dag_and_dot ${SHM_TEST_CMD_LIST}
            ${Python_EXECUTABLE}
            ${PROJECT_BINARY_DIR}/tools/profiling/python/profile2h5.py --output=bw.h5 bw-0.prof bw-1.prof)
    set_property(TEST profiling/generate_hdf5_for_dag_and_dot APPEND PROPERTY DEPENDS profiling/generate_profile_and_dot_bw:mp)
    set_property(TEST profiling/generate_hdf5_for_dag_and_dot APPEND PROPERTY ENVIRONMENT
            LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/tools/profiling/python/build/temp.${SYSCONF}:$ENV{LD_LIBRARY_PATH})
    set_property(TEST profiling/generate_hdf5_for_dag_and_dot APPEND PROPERTY ENVIRONMENT
            PYTHONPATH=${CMAKE_BINARY_DIR}/tools/profiling/python/build/lib.${SYSCONF}/:$ENV{PYTHONPATH})

    parsec_addtest_cmd(profiling/check_DAG_and_Trace ${SHM_TEST_CMD_LIST} ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/profiling/python/examples/example-DAG-and-Trace.py --dot bw-0.dot --dot bw-1.dot --h5 bw.h5)
    set_property(TEST profiling/check_DAG_and_Trace APPEND PROPERTY DEPENDS profiling/generate_hdf5_for_dag_and_dot)

    parsec_addtest_cmd(profiling/cleanup_dot_files ${SHM_TEST_CMD_LIST} rm -f bw-0.prof bw-1.prof bw.h5 bw-0.dot bw-1.dot)
  endif(PARSEC_PROF_GRAPHER)
endif(Python_FOUND AND PARSEC_PYTHON_TOOLS AND PARSEC_PROF_TRACE AND MPI_C_FOUND)
