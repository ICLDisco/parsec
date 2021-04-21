find_package (Python COMPONENTS Interpreter)
if(Python_FOUND AND PARSEC_PYTHON_TOOLS AND PARSEC_PROF_TRACE)
    execute_process(COMMAND ${Python_EXECUTABLE} -c
                            "from __future__ import print_function; import sysconfig; import sys; print('{}-{}.{}'.format(sysconfig.get_platform(),sys.version_info[0],sys.version_info[1]), end='')"
                    OUTPUT_VARIABLE SYSCONF)
    parsec_addtest_cmd(generate_async_profile ${SHM_TEST_CMD_LIST} ./async 100 -- --mca profile_filename async  --mca mca_pins task_profiler)

    parsec_addtest_cmd(generate_hdf5_async_file ${SHM_TEST_CMD_LIST}
                       ${Python_EXECUTABLE}
                       ${PROJECT_BINARY_DIR}/tools/profiling/python/profile2h5.py --output=async.h5 async-0.prof)
    set_property(TEST generate_hdf5_async_file APPEND PROPERTY DEPENDS generate_async_profile)
    set_property(TEST generate_hdf5_async_file APPEND PROPERTY ENVIRONMENT
                 LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/tools/profiling/python/build/temp.${SYSCONF}:$ENV{LD_LIBRARY_PATH})
    set_property(TEST generate_hdf5_async_file APPEND PROPERTY ENVIRONMENT
                 PYTHONPATH=${CMAKE_BINARY_DIR}/tools/profiling/python/build/lib.${SYSCONF}/:$ENV{PYTHONPATH})

    parsec_addtest_cmd(check_hdf5_async_file ${SHM_TEST_CMD_LIST} ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check-profile.py async.h5)
    set_property(TEST check_hdf5_async_file APPEND PROPERTY DEPENDS generate_hdf5_async_file)

    parsec_addtest_cmd(cleanup_profile_async_files ${SHM_TEST_CMD_LIST} rm -f async-0.prof async.h5)
endif(Python_FOUND AND PARSEC_PYTHON_TOOLS AND PARSEC_PROF_TRACE)
