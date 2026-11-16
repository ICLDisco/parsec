include(${CMAKE_CURRENT_LIST_DIR}/ptgpp/Testings.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/user-defined-functions/Testings.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/branching/Testings.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/multisize_bcast/Testings.cmake)

parsec_addtest_cmd(dsl/ptg/startup1 ${SHM_TEST_CMD_LIST} dsl/ptg/startup -i=10 -j=10 -k=10 -v=5)
parsec_addtest_cmd(dsl/ptg/startup2 ${SHM_TEST_CMD_LIST} dsl/ptg/startup -i=10 -j=20 -k=30 -v=5)
parsec_addtest_cmd(dsl/ptg/startup3 ${SHM_TEST_CMD_LIST} dsl/ptg/startup -i=30 -j=30 -k=30 -v=5)
parsec_addtest_cmd(dsl/ptg/strange ${SHM_TEST_CMD_LIST} dsl/ptg/strange)
if(PARSEC_HAVE_DEV_CAPABILITY_BATCH)
  parsec_addtest_cmd(dsl/ptg/batch_cpu ${SHM_TEST_CMD_LIST} dsl/ptg/batch_cpu)
endif(PARSEC_HAVE_DEV_CAPABILITY_BATCH)
# the inline C code of a dependency property sees its parent and the size of
# the incoming data: the receive count is computed from it
parsec_addtest_cmd(dsl/ptg/receive_ctx ${SHM_TEST_CMD_LIST} dsl/ptg/receive_ctx)
if( MPI_C_FOUND )
  parsec_addtest_cmd(dsl/ptg/receive_ctx:mp ${MPI_TEST_CMD_LIST} 2 dsl/ptg/receive_ctx)
  parsec_addtest_cmd(dsl/ptg/receive_ctx:rdv:mp ${MPI_TEST_CMD_LIST} 2 dsl/ptg/receive_ctx -d=4096 -- --mca runtime_comm_short_limit 0)
endif( MPI_C_FOUND )
