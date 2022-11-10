
#
# Shared Memory Testings
#
parsec_addtest_cmd(parsec/dsl/dtd/task_generation ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_task_generation)
parsec_addtest_cmd(parsec/dsl/dtd/task_inserting_task ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_task_inserting_task)
parsec_addtest_cmd(parsec/dsl/dtd/task_insertion ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_task_insertion)
parsec_addtest_cmd(parsec/dsl/dtd/war ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_war)
parsec_addtest_cmd(parsec/dsl/dtd/new_tile:cpu ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_new_tile --mca device_cuda_enabled 0)
if(PARSEC_HAVE_CUDA AND CMAKE_CUDA_COMPILER)
# How do we run CUDA tests? Is there a SHM_TEST_CMD_LIST_CUDA?
  parsec_addtest_cmd(parsec/dsl/dtd/new_tile:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} dsl/dtd/dtd_test_new_tile --mca device_cuda_enabled 1 --mca device cuda)
endif(PARSEC_HAVE_CUDA AND CMAKE_CUDA_COMPILER)

#
# Distributed Memory Testings
#
if( MPI_C_FOUND )
  parsec_addtest_cmd(parsec/dsl/dtd/pingpong:mp ${MPI_TEST_CMD_LIST} 2 dsl/dtd/dtd_test_pingpong)
  parsec_addtest_cmd(parsec/dsl/dtd/task_inserting_task:mp ${MPI_TEST_CMD_LIST} 4 dsl/dtd/dtd_test_task_inserting_task)
  parsec_addtest_cmd(parsec/dsl/dtd/task_insertion:mp ${MPI_TEST_CMD_LIST} 4 dsl/dtd/dtd_test_task_insertion)
  parsec_addtest_cmd(parsec/dsl/dtd/war:mp ${MPI_TEST_CMD_LIST} 4 dsl/dtd/dtd_test_war)
  parsec_addtest_cmd(parsec/dsl/dtd/interleave_actions:mp ${MPI_TEST_CMD_LIST} 4 dsl/dtd/dtd_test_interleave_actions)
  parsec_addtest_cmd(parsec/dsl/dtd/allreduce:mp ${MPI_TEST_CMD_LIST} 4 dsl/dtd/dtd_test_allreduce)
  parsec_addtest_cmd(parsec/dsl/dtd/new_tile:mp:cpu ${MPI_TEST_CMD_LIST} 2 dsl/dtd/dtd_test_new_tile --mca device_cuda_enabled 0)
  if(PARSEC_HAVE_CUDA AND CMAKE_CUDA_COMPILER)
    parsec_addtest_cmd(parsec/dsl/dtd/new_tile:mp:gpu ${MPI_TEST_CMD_LIST} 2 ${CTEST_CUDA_LAUNCHER_OPTIONS} dsl/dtd/dtd_test_new_tile --mca device_cuda_enabled 1 --mca device cuda)
  endif(PARSEC_HAVE_CUDA AND CMAKE_CUDA_COMPILER)
endif( MPI_C_FOUND )
