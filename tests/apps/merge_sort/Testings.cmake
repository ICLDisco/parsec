add_test(parsec/apps/merge_sort ${SHM_TEST_CMD_LIST} apps/merge_sort/merge_sort)
if( MPI_C_FOUND )
  add_test(parsec/apps/merge_sort:mp ${MPI_TEST_CMD_LIST} 4 apps/merge_sort/merge_sort)
endif( MPI_C_FOUND)

