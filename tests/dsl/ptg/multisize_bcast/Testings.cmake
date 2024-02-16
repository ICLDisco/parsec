parsec_addtest_cmd(dsl/ptg/multisize_bcast ${SHM_TEST_CMD_LIST} dsl/ptg/multisize_bcast/check_multisize_bcast)
if( MPI_C_FOUND )
  parsec_addtest_cmd(dsl/ptg/multisize_bcast:mp ${MPI_TEST_CMD_LIST} 4 dsl/ptg/multisize_bcast/check_multisize_bcast)
endif( MPI_C_FOUND)
