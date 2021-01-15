parsec_addtest_cmd(reshape_shm ${SHM_TEST_CMD_LIST} ./reshape -N 120 -t 9 -c 10)
parsec_addtest_cmd(reshape_shm_multi ${SHM_TEST_CMD_LIST} ./reshape -N 120 -t 9 -c 10 -m 1)
if( MPI_C_FOUND )
  parsec_addtest_cmd(reshape_mpi ${MPI_TEST_CMD_LIST} 4 ./reshape -N 120 -t 9 -c 10)
  parsec_addtest_cmd(reshape_mpi_multi ${MPI_TEST_CMD_LIST} 4 ./reshape -N 120 -t 9 -c 10 -m 1 )
endif( MPI_C_FOUND)

parsec_addtest_cmd(input_dep_reshape_single_copy_shm ${SHM_TEST_CMD_LIST} ./input_dep_reshape_single_copy -N 12 -t 2 -c 2)
parsec_addtest_cmd(input_dep_reshape_single_copy_shm_multi ${SHM_TEST_CMD_LIST} ./input_dep_reshape_single_copy -N 12 -t 2 -c 2 -m 1)
if( MPI_C_FOUND )
  parsec_addtest_cmd(input_dep_reshape_single_copy_mpi ${MPI_TEST_CMD_LIST} 4 ./input_dep_reshape_single_copy -N 12 -t 2 -c 2)
  parsec_addtest_cmd(input_dep_reshape_single_copy_mpi_multi ${MPI_TEST_CMD_LIST} 4 ./input_dep_reshape_single_copy -N 12 -t 2 -c 2 -m 1 )
endif( MPI_C_FOUND)

#These tests will fail with runtime_comm_short_limit != 0. Explanation on testing_remote_multiple_outs_same_pred_flow.c
parsec_addtest_cmd(reshape_remote_multiple_outs_same_pred_flow_shm ${SHM_TEST_CMD_LIST} ./remote_multiple_outs_same_pred_flow -N 320 -t 9 -c 10 -- --mca runtime_comm_short_limit 0)
parsec_addtest_cmd(reshape_remote_multiple_outs_same_pred_flow_shm_multi ${SHM_TEST_CMD_LIST} ./remote_multiple_outs_same_pred_flow -N 320 -t 9 -c 10 -m 1 -- --mca runtime_comm_short_limit 0)
if( MPI_C_FOUND )
  parsec_addtest_cmd(reshape_remote_multiple_outs_same_pred_flow_mpi ${MPI_TEST_CMD_LIST} 4 ./remote_multiple_outs_same_pred_flow -N 320 -t 9 -c 10  -- --mca runtime_comm_short_limit 0)
  parsec_addtest_cmd(reshape_remote_multiple_outs_same_pred_flow_mpi_multi ${MPI_TEST_CMD_LIST} 4 ./remote_multiple_outs_same_pred_flow -N 320 -t 9 -c 10 -m 1  -- --mca runtime_comm_short_limit 0)
endif( MPI_C_FOUND)

parsec_addtest_cmd(avoidable_reshape_shm ${SHM_TEST_CMD_LIST} ./avoidable_reshape -N 100 -t 2 -c 10)