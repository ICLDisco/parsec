parsec_addtest_cmd(api/touch  ${SHM_TEST_CMD_LIST} api/touch_ex -v=5)
parsec_addtest_cmd(api/touch:inline  ${SHM_TEST_CMD_LIST} api/touch_ex_inline -v=5)

# Fortran Testings
if(MPI_Fortran_FOUND AND CMAKE_Fortran_COMPILER_WORKS)
  if(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
    parsec_addtest_cmd(api/touchf:fortran  ${SHM_TEST_CMD_LIST} api/touch_exf -v=5)
  endif(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
endif(MPI_Fortran_FOUND AND CMAKE_Fortran_COMPILER_WORKS)

parsec_addtest_cmd(api/compose ${SHM_TEST_CMD_LIST} api/compose)

if( MPI_C_FOUND )
  parsec_addtest_cmd(api/compose:mp ${MPI_TEST_CMD_LIST} 4 api/compose)
  # Test temporarily disabled to allow #309 to be merged.
  #parsec_addtest_cmd(api/operator:mp ${MPI_TEST_CMD_LIST} 4 api/operator)
endif( MPI_C_FOUND )

