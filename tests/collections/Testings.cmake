
parsec_addtest_cmd(collections/reduce ${SHM_TEST_CMD_LIST} collections/reduce)

if( MPI_C_FOUND )
    parsec_addtest_cmd(collections/redistribute:mp ${MPI_TEST_CMD_LIST} 8 collections/redistribute/testing_redistribute -M 2400 -N 2400 -a 2400 -A 2400 -t 300 -T 300 -b 200 -B 200 -m 2000 -n 2000 -I 30 -J 40 -i 100 -j 121 -v -z -x -P 2 -Q 4 -p 4 -q 2)
    parsec_addtest_cmd(collections/redistribute_random:mp ${MPI_TEST_CMD_LIST} 8 collections/redistribute/testing_redistribute_random -M 2400 -N 2400 -a 2400 -A 2400 -t 300 -T 300 -b 200 -B 200 -m 2000 -n 2000 -I 30 -J 40 -i 100 -j 121 -v -z -x -P 2 -Q 4 -p 4 -q 2)
else( MPI_C_FOUND )
    parsec_addtest_cmd(collections/redistribute ${MPI_TEST_CMD_LIST} collections/redistribute/testing_redistribute -M 2400 -N 2400 -a 2400 -A 2400 -t 300 -T 300 -b 200 -B 200 -m 2000 -n 2000 -I 30 -J 40 -i 100 -j 121 -v -z -x)
    parsec_addtest_cmd(collections/redistribute_random ${MPI_TEST_CMD_LIST} collections/redistribute/testing_redistribute_random -M 2400 -N 2400 -a 2400 -A 2400 -t 300 -T 300 -b 200 -B 200 -m 2000 -n 2000 -I 30 -J 40 -i 100 -j 121 -v -z -x)
endif( MPI_C_FOUND )

parsec_addtest_cmd(collections/reshape ${SHM_TEST_CMD_LIST} collections/reshape/reshape -N 120 -t 9 -c 10)
parsec_addtest_cmd(collections/reshape:mt ${SHM_TEST_CMD_LIST} collections/reshape/reshape -N 120 -t 9 -c 10 -m 1)
if( MPI_C_FOUND )
  parsec_addtest_cmd(collections/reshape:mp ${MPI_TEST_CMD_LIST} 4 collections/reshape/reshape -N 120 -t 9 -c 10)
  parsec_addtest_cmd(collections/reshape:mp:mt ${MPI_TEST_CMD_LIST} 4 collections/reshape/reshape -N 120 -t 9 -c 10 -m 1 )
endif( MPI_C_FOUND)

parsec_addtest_cmd(collections/reshape/input_single_copy ${SHM_TEST_CMD_LIST} collections/reshape/input_dep_reshape_single_copy -N 12 -t 2 -c 2)
parsec_addtest_cmd(collections/reshape/input_single_copy:mt ${SHM_TEST_CMD_LIST} collections/reshape/input_dep_reshape_single_copy -N 12 -t 2 -c 2 -m 1)
if( MPI_C_FOUND )
  parsec_addtest_cmd(collections/reshape/input_single_copy:mp ${MPI_TEST_CMD_LIST} 4 collections/reshape/input_dep_reshape_single_copy -N 12 -t 2 -c 2)
  parsec_addtest_cmd(collections/reshape/input_single_copy:mp:mt ${MPI_TEST_CMD_LIST} 4 collections/reshape/input_dep_reshape_single_copy -N 12 -t 2 -c 2 -m 1 )
endif( MPI_C_FOUND)

#These tests will fail with runtime_comm_short_limit != 0. Explanation on testing_remote_multiple_outs_same_pred_flow.c
parsec_addtest_cmd(collections/reshape/remote_multiple_outs_same_pred_flow ${SHM_TEST_CMD_LIST} collections/reshape/remote_multiple_outs_same_pred_flow -N 320 -t 9 -c 10 -- --mca runtime_comm_short_limit 0)
parsec_addtest_cmd(collections/reshape/remote_multiple_outs_same_pred_flow:mt ${SHM_TEST_CMD_LIST} collections/reshape/remote_multiple_outs_same_pred_flow -N 320 -t 9 -c 10 -m 1 -- --mca runtime_comm_short_limit 0)
if( MPI_C_FOUND )
  parsec_addtest_cmd(collections/reshape/remote_multiple_outs_same_pred_flow:mp ${MPI_TEST_CMD_LIST} 4 collections/reshape/remote_multiple_outs_same_pred_flow -N 320 -t 9 -c 10  -- --mca runtime_comm_short_limit 0)
  parsec_addtest_cmd(collections/reshape/remote_multiple_outs_same_pred_flow:mp:mt ${MPI_TEST_CMD_LIST} 4 collections/reshape/remote_multiple_outs_same_pred_flow -N 320 -t 9 -c 10 -m 1  -- --mca runtime_comm_short_limit 0)
endif( MPI_C_FOUND)

parsec_addtest_cmd(collections/reshape/avoidable ${SHM_TEST_CMD_LIST} collections/reshape/avoidable_reshape -N 100 -t 2 -c 10)

if( MPI_C_FOUND )
  parsec_addtest_cmd(collections/matrix/band ${MPI_TEST_CMD_LIST} 8 collections/two_dim_band/testing_band -N 3200 -T 160 -P 4 -s 5 -S 10 -p 2 -f 2 -F 10 -b 2)
else( MPI_C_FOUND )
  parsec_addtest_cmd(collections/matrix/band ${SHM_TEST_CMD_LIST} collections/two_dim_band/testing_band -N 3200 -T 160 -b 2)
endif( MPI_C_FOUND )
