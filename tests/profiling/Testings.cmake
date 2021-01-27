find_package (Python COMPONENTS Interpreter)

if(PARSEC_PROF_TRACE AND MPI_C_FOUND AND Python_FOUND)
    execute_process(COMMAND ${Python_EXECUTABLE} -c
            "import sysconfig; import sys; print('{}-{}.{}'.format(sysconfig.get_platform(),sys.version_info[0],sys.version_info[1]), end='')"
            OUTPUT_VARIABLE SYSCONF)
    parsec_addtest_cmd(generate_bw_profile ${MPI_TEST_CMD_LIST} 2 ../pingpong/bw_test -n 10 -f 10 -l 2097152 -- --mca profile_filename bw  --mca mca_pins task_profiler)

    parsec_addtest_cmd(generate_hdf5_bw_file ${SHM_TEST_CMD_LIST}
            ${Python_EXECUTABLE}
            ${CMAKE_BINARY_DIR}/tools/profiling/python/examples/profile2h5.py --output=bw.h5 bw-0.prof bw-1.prof)
    set_property(TEST generate_hdf5_bw_file APPEND PROPERTY DEPENDS generate_bw_profile)
    set_property(TEST generate_hdf5_bw_file APPEND PROPERTY ENVIRONMENT
            LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/tools/profiling/python/build/temp.${SYSCONF}:$ENV{LD_LIBRARY_PATH})
    set_property(TEST generate_hdf5_bw_file APPEND PROPERTY ENVIRONMENT
            PYTHONPATH=${CMAKE_BINARY_DIR}/tools/profiling/python/build/lib.${SYSCONF}/:$ENV{PYTHONPATH})

    parsec_addtest_cmd(check_hdf5_bw_file ${SHM_TEST_CMD_LIST} ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_comms.py)
    set_property(TEST check_hdf5_bw_file APPEND PROPERTY DEPENDS generate_hdf5_bw_file)

    parsec_addtest_cmd(cleanup_profile_files ${SHM_TEST_CMD_LIST} rm -f bw-0.prof bw-1.prof bw.h5)
endif(PARSEC_PROF_TRACE AND MPI_C_FOUND AND Python_FOUND)
