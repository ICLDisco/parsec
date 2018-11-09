parsec_addtest_cmd(unit_check_multisize_bcast_shm ${SHM_TEST_CMD_LIST} ./check_multisize_bcast)
if( MPI_C_FOUND )
  parsec_addtest_cmd(unit_check_multisize_bcast_mpi ${MPI_TEST_CMD_LIST} 4 ./check_multisize_bcast)
endif( MPI_C_FOUND)
