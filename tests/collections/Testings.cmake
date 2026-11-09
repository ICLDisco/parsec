#
# Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
#

parsec_addtest_cmd(collections/reduce ${SHM_TEST_CMD_LIST} collections/reduce)

if( MPI_C_FOUND )
    parsec_addtest_cmd(collections/redistribute:mp ${MPI_TEST_CMD_LIST} 8 collections/redistribute/testing_redistribute -M 2400 -N 2400 -a 2400 -A 2400 -t 300 -T 300 -b 200 -B 200 -m 2000 -n 2000 -I 30 -J 40 -i 100 -j 121 -v -z -x -P 2 -Q 4 -p 4 -q 2)

    set(PARSEC_REDISTRIBUTE_SMALL_ARGS
        -M 12 -N 12 -a 12 -A 12
        -t 2 -T 2 -b 2 -B 2
        -m 4 -n 4
        -x -P 3 -Q 1 -p 3 -q 1)
    # Triangular descriptors pick the triangle that holds the region: the
    # first region lies in the lower triangle, its mirror in the upper one.
    set(PARSEC_REDISTRIBUTE_LOWER_REGION -I 6 -J 0 -i 6 -j 0)
    set(PARSEC_REDISTRIBUTE_UPPER_REGION -I 0 -J 6 -i 0 -j 6)
    set(PARSEC_REDISTRIBUTE_DISTRIBUTIONS 2dbc sym-2dbc sdb)
    set(PARSEC_REDISTRIBUTE_MEMORY_LOCATIONS cpu)
    if(PARSEC_HAVE_CUDA)
        list(APPEND PARSEC_REDISTRIBUTE_MEMORY_LOCATIONS managed cuda)
    endif()

    foreach(_source_distribution ${PARSEC_REDISTRIBUTE_DISTRIBUTIONS})
      foreach(_target_distribution ${PARSEC_REDISTRIBUTE_DISTRIBUTIONS})
        foreach(_source_memory ${PARSEC_REDISTRIBUTE_MEMORY_LOCATIONS})
          foreach(_target_memory ${PARSEC_REDISTRIBUTE_MEMORY_LOCATIONS})
            set(_redistribute_cuda_launcher)
            set(_redistribute_cuda_options)
            if((NOT "${_source_memory}" STREQUAL "cpu") OR
               (NOT "${_target_memory}" STREQUAL "cpu"))
              set(_redistribute_cuda_launcher ${CTEST_CUDA_LAUNCHER_OPTIONS})
              set(_redistribute_cuda_options -- --mca device_cuda_enabled 1 --mca device cuda)
            endif()

            parsec_addtest_cmd(collections/redistribute:mp:${_source_distribution}_to_${_target_distribution}:${_source_memory}_to_${_target_memory}
                ${MPI_TEST_CMD_LIST} 3
                ${_redistribute_cuda_launcher}
                collections/redistribute/testing_redistribute
                ${PARSEC_REDISTRIBUTE_SMALL_ARGS}
                ${PARSEC_REDISTRIBUTE_LOWER_REGION}
                -g ${_source_distribution} -G ${_target_distribution}
                -l ${_source_memory} -L ${_target_memory}
                ${_redistribute_cuda_options})
          endforeach()
        endforeach()
      endforeach()

      foreach(_target_distribution ${PARSEC_REDISTRIBUTE_DISTRIBUTIONS})
        parsec_addtest_cmd(collections/redistribute:mp:${_source_distribution}_to_${_target_distribution}:cpu_to_cpu:upper
            ${MPI_TEST_CMD_LIST} 3
            collections/redistribute/testing_redistribute
            ${PARSEC_REDISTRIBUTE_SMALL_ARGS}
            ${PARSEC_REDISTRIBUTE_UPPER_REGION}
            -g ${_source_distribution} -G ${_target_distribution})
      endforeach()
    endforeach()

    # Redistribute through sub-matrix views that start a few tiles in, so the
    # descriptors carry a nonzero (i, j): whether a tile is stored must be
    # decided from its global position. The lower case keeps equal tile sizes
    # (reshuffle path), the upper case uses 3x3 target tiles (general path).
    foreach(_distribution sym-2dbc sdb)
      parsec_addtest_cmd(collections/redistribute:mp:${_distribution}_to_${_distribution}:cpu_to_cpu:lower_view
          ${MPI_TEST_CMD_LIST} 3
          collections/redistribute/testing_redistribute
          ${PARSEC_REDISTRIBUTE_SMALL_ARGS}
          ${PARSEC_REDISTRIBUTE_LOWER_REGION}
          -g ${_distribution} -G ${_distribution}
          -o 3,0 -O 3,0)
      parsec_addtest_cmd(collections/redistribute:mp:${_distribution}_to_${_distribution}:cpu_to_cpu:upper_view
          ${MPI_TEST_CMD_LIST} 3
          collections/redistribute/testing_redistribute
          ${PARSEC_REDISTRIBUTE_SMALL_ARGS} -b 3 -B 3
          ${PARSEC_REDISTRIBUTE_UPPER_REGION}
          -g ${_distribution} -G ${_distribution}
          -o 0,3 -O 0,2)
    endforeach()

    parsec_addtest_cmd(collections/redistribute_random:mp ${MPI_TEST_CMD_LIST} 8 collections/redistribute/testing_redistribute_random -M 2400 -N 2400 -a 2400 -A 2400 -t 300 -T 300 -b 200 -B 200 -m 2000 -n 2000 -I 30 -J 40 -i 100 -j 121 -v -z -x -P 2 -Q 4 -p 4 -q 2)
endif( MPI_C_FOUND )

# Reshaping describes a copy as two different layouts, which a build without a
# communication backend cannot represent: every derived type collapses to
# PARSEC_DATATYPE_NULL. Even the single process flavors need a distributed build.
if( MPI_C_FOUND )
  parsec_addtest_cmd(collections/reshape ${SHM_TEST_CMD_LIST} collections/reshape/reshape -N 120 -t 9 -c 10)
  parsec_addtest_cmd(collections/reshape:mt ${SHM_TEST_CMD_LIST} collections/reshape/reshape -N 120 -t 9 -c 10 -m 1)
  parsec_addtest_cmd(collections/reshape:mp ${MPI_TEST_CMD_LIST} 4 collections/reshape/reshape -N 120 -t 9 -c 10)
  parsec_addtest_cmd(collections/reshape:mp:mt ${MPI_TEST_CMD_LIST} 4 collections/reshape/reshape -N 120 -t 9 -c 10 -m 1 )

  parsec_addtest_cmd(collections/reshape/input_single_copy ${SHM_TEST_CMD_LIST} collections/reshape/input_dep_reshape_single_copy -N 12 -t 2 -c 2)
  parsec_addtest_cmd(collections/reshape/input_single_copy:mt ${SHM_TEST_CMD_LIST} collections/reshape/input_dep_reshape_single_copy -N 12 -t 2 -c 2 -m 1)
  parsec_addtest_cmd(collections/reshape/input_single_copy:mp ${MPI_TEST_CMD_LIST} 4 collections/reshape/input_dep_reshape_single_copy -N 12 -t 2 -c 2)
  parsec_addtest_cmd(collections/reshape/input_single_copy:mp:mt ${MPI_TEST_CMD_LIST} 4 collections/reshape/input_dep_reshape_single_copy -N 12 -t 2 -c 2 -m 1 )

  #These tests will fail with runtime_comm_short_limit != 0. Explanation on testing_remote_multiple_outs_same_pred_flow.c
  parsec_addtest_cmd(collections/reshape/remote_multiple_outs_same_pred_flow ${SHM_TEST_CMD_LIST} collections/reshape/remote_multiple_outs_same_pred_flow -N 320 -t 9 -c 10 -- --mca runtime_comm_short_limit 0)
  parsec_addtest_cmd(collections/reshape/remote_multiple_outs_same_pred_flow:mt ${SHM_TEST_CMD_LIST} collections/reshape/remote_multiple_outs_same_pred_flow -N 320 -t 9 -c 10 -m 1 -- --mca runtime_comm_short_limit 0)
  parsec_addtest_cmd(collections/reshape/remote_multiple_outs_same_pred_flow:mp ${MPI_TEST_CMD_LIST} 4 collections/reshape/remote_multiple_outs_same_pred_flow -N 320 -t 9 -c 10  -- --mca runtime_comm_short_limit 0)
  parsec_addtest_cmd(collections/reshape/remote_multiple_outs_same_pred_flow:mp:mt ${MPI_TEST_CMD_LIST} 4 collections/reshape/remote_multiple_outs_same_pred_flow -N 320 -t 9 -c 10 -m 1  -- --mca runtime_comm_short_limit 0)
endif( MPI_C_FOUND)

parsec_addtest_cmd(collections/reshape/avoidable ${SHM_TEST_CMD_LIST} collections/reshape/avoidable_reshape -N 100 -t 2 -c 10)

if( MPI_C_FOUND )
  parsec_addtest_cmd(collections/matrix/band ${MPI_TEST_CMD_LIST} 8 collections/two_dim_band/testing_band -N 3200 -T 160 -P 4 -s 5 -S 10 -p 2 -f 2 -F 10 -b 2)
else( MPI_C_FOUND )
  parsec_addtest_cmd(collections/matrix/band ${SHM_TEST_CMD_LIST} collections/two_dim_band/testing_band -N 3200 -T 160 -b 2)
endif( MPI_C_FOUND )
