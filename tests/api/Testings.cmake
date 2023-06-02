parsec_addtest_cmd(api/init_fini ${SHM_TEST_CMD_LIST} api/init_fini)
parsec_addtest_cmd(api/touch  ${SHM_TEST_CMD_LIST} api/touch_ex -v=5)
parsec_addtest_cmd(api/touch:inline  ${SHM_TEST_CMD_LIST} api/touch_ex_inline -v=5)

# Fortran Testings
if(MPI_Fortran_FOUND AND CMAKE_Fortran_COMPILER_WORKS AND CMAKE_Fortran_COMPILER_SUPPORTS_F90)
  parsec_addtest_cmd(api/touchf:fortran  ${SHM_TEST_CMD_LIST} api/touch_exf -v=5)
endif()

parsec_addtest_cmd(api/compose ${SHM_TEST_CMD_LIST} api/compose)

if( MPI_C_FOUND )
  parsec_addtest_cmd(api/init_fini:mp ${MPI_TEST_CMD_LIST} 4 api/init_fini)
  parsec_addtest_cmd(api/compose:mp ${MPI_TEST_CMD_LIST} 4 api/compose)
  parsec_addtest_cmd(api/operator:mp ${MPI_TEST_CMD_LIST} 4 api/operator)
endif( MPI_C_FOUND )

parsec_addtest_cmd(api/taskpool_wait ${SHM_TEST_CMD_LIST} api/taskpool_wait/taskpool_wait)
if( MPI_C_FOUND )
  parsec_addtest_cmd(api/taskpool_wait:mp ${MPI_TEST_CMD_LIST} 8 api/taskpool_wait/taskpool_wait)
endif()
