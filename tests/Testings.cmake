#
# Shared Memory Testings
#

add_test(startup ${SHM_TEST_CMD_LIST} ./startup -i=10 -j=10 -k=10 -v=5)
add_test(startup ${SHM_TEST_CMD_LIST} ./startup -i=10 -j=20 -k=30 -v=5)
add_test(startup ${SHM_TEST_CMD_LIST} ./startup -i=50 -j=50 -k=50 -v=5)

# The headnode lack GPUs so we need MPI in order to get the test to run on
# one of the nodes.
if (CUDA_FOUND AND MPI_C_FOUND)
endif (CUDA_FOUND AND MPI_C_FOUND)

#
# Distributed Memory Testings
#
if( MPI_C_FOUND )
  # Check MPI
  add_test(mpi_test   ${MPI_TEST_CMD_LIST} -np 8 /bin/true)

endif( MPI_C_FOUND )

