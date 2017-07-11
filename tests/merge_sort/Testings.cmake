add_test(unit_merge_sort_shm ${SHM_TEST_CMD_LIST} ./merge_sort)
if( MPI_C_FOUND )
  add_test(unit_merge_sort_mpi ${MPI_TEST_CMD_LIST} 4 ./merge_sort)
endif( MPI_C_FOUND)

