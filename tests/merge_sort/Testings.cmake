add_test(shm_merge_sort ${SHM_TEST_CMD_LIST} ./merge_sort)
if( MPI_C_FOUND )
  add_test(mpi_merge_sort ${MPI_TEST_CMD_LIST} -np 4 ./merge_sort)
endif( MPI_C_FOUND)

