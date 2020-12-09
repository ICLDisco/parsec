#
# Shared Memory Testings
#

parsec_addtest_cmd(unit_startup1_shm ${SHM_TEST_CMD_LIST} ./startup -i=10 -j=10 -k=10 -v=5)
parsec_addtest_cmd(unit_startup2_shm ${SHM_TEST_CMD_LIST} ./startup -i=10 -j=20 -k=30 -v=5)
parsec_addtest_cmd(unit_startup3_shm ${SHM_TEST_CMD_LIST} ./startup -i=30 -j=30 -k=30 -v=5)
parsec_addtest_cmd(unit_reduce_shm ${SHM_TEST_CMD_LIST} ./reduce)
parsec_addtest_cmd(unit_strange_shm ${SHM_TEST_CMD_LIST} ./strange)
parsec_addtest_cmd(unit_touch_shm  ${SHM_TEST_CMD_LIST} ./touch_ex -v=5)
parsec_addtest_cmd(unit_touch_inline_shm  ${SHM_TEST_CMD_LIST} ./touch_ex_inline -v=5)
parsec_addtest_cmd(compose ${SHM_TEST_CMD_LIST} ./compose)

#
# Fortran Testings
#
if(MPI_Fortran_FOUND AND CMAKE_Fortran_COMPILER_WORKS)
  if(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
    parsec_addtest_cmd(unit_touch_fortran_shm  ${SHM_TEST_CMD_LIST} ./touch_exf -v=5)
  endif(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
endif(MPI_Fortran_FOUND AND CMAKE_Fortran_COMPILER_WORKS)

#
# Distributed Memory Testings
#
if( MPI_C_FOUND )
  parsec_addtest_cmd(compose ${MPI_TEST_CMD_LIST} 4 ./compose)
endif( MPI_C_FOUND )

#
# CUDA Testings
#
if (PARSEC_HAVE_CUDA AND MPI_C_FOUND)
# The headnode lack GPUs so we need MPI in order to get the test to run on
# one of the nodes.
endif (PARSEC_HAVE_CUDA AND MPI_C_FOUND)


