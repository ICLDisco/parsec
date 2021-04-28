
#
# Shared Memory Testings
#
parsec_addtest_cmd(dsl/dtd/task_generation ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_task_generation)
parsec_addtest_cmd(dsl/dtd/task_inserting_task ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_task_inserting_task)
parsec_addtest_cmd(dsl/dtd/task_insertion ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_task_insertion)
parsec_addtest_cmd(dsl/dtd/war ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_war)

#
# Distributed Memory Testings
#
if( MPI_C_FOUND )
  parsec_addtest_cmd(dsl/dtd/pingpong:mp ${MPI_TEST_CMD_LIST} 2 dsl/dtd/dtd_test_pingpong)
  parsec_addtest_cmd(dsl/dtd/task_inserting_task:mp ${MPI_TEST_CMD_LIST} 4 dsl/dtd/dtd_test_task_inserting_task)
  parsec_addtest_cmd(dsl/dtd/task_insertion:mp ${MPI_TEST_CMD_LIST} 4 dsl/dtd/dtd_test_task_insertion)
  parsec_addtest_cmd(dsl/dtd/war:mp ${MPI_TEST_CMD_LIST} 4 dsl/dtd/dtd_test_war)
  parsec_addtest_cmd(dsl/dtd/interleave_actions:mp ${MPI_TEST_CMD_LIST} 4 dsl/dtd/dtd_test_interleave_actions)
  parsec_addtest_cmd(dsl/dtd/allreduce:mp ${MPI_TEST_CMD_LIST} 4 dsl/dtd/dtd_test_allreduce)
endif( MPI_C_FOUND )
