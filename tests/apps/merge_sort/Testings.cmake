parsec_addtest_cmd(apps/merge_sort ${SHM_TEST_CMD_LIST} apps/merge_sort/merge_sort)
if( MPI_C_FOUND )
  parsec_addtest_cmd(apps/merge_sort:mp ${MPI_TEST_CMD_LIST} 4 apps/merge_sort/merge_sort)
endif( MPI_C_FOUND)

