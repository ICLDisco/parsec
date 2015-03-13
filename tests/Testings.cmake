#
# Shared Memory Testings
#

add_test(unit_startup1 ${SHM_TEST_CMD_LIST} ./startup -i=10 -j=10 -k=10 -v=5)
add_test(unit_startup2 ${SHM_TEST_CMD_LIST} ./startup -i=10 -j=20 -k=30 -v=5)
add_test(unit_startup3 ${SHM_TEST_CMD_LIST} ./startup -i=30 -j=30 -k=30 -v=5)
add_test(unit_reduce ${SHM_TEST_CMD_LIST} ./reduce -v=5)

add_test(unit_touch    ${SHM_TEST_CMD_LIST} ./touch_ex -v=5)
if(CMAKE_Fortran_COMPILER_WORKS)
  if(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
    add_test(unit_touchf  ${SHM_TEST_CMD_LIST} ./touch_ex -v=5)
  endif(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
endif(CMAKE_Fortran_COMPILER_WORKS)


# The headnode lack GPUs so we need MPI in order to get the test to run on
# one of the nodes.
if (CUDA_FOUND AND MPI_C_FOUND)
endif (CUDA_FOUND AND MPI_C_FOUND)

#
# Distributed Memory Testings
#
if( MPI_C_FOUND )
  find_program(BINTRUE true)
  # Check MPI
  add_test(unit_mpi_test ${MPI_TEST_CMD_LIST} -np 8 ${BINTRUE})

endif( MPI_C_FOUND )

