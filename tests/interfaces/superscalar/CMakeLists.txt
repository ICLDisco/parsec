include_directories(${CMAKE_CURRENT_BINARY_DIR})
set(COMMON_DATA "common_data.c")
parsec_addtest(C dtd_test_pingpong "dtd_test_pingpong.c;${COMMON_DATA}")
parsec_addtest(C dtd_test_task_generation "dtd_test_task_generation.c;${COMMON_DATA}")
parsec_addtest(C dtd_test_war "dtd_test_war.c;${COMMON_DATA}")
parsec_addtest(C dtd_test_task_insertion "dtd_test_task_insertion.c")
parsec_addtest(C dtd_test_null_as_tile "dtd_test_null_as_tile.c")
parsec_addtest(C dtd_test_task_inserting_task "dtd_test_task_inserting_task.c")
parsec_addtest(C dtd_test_flag_dont_track "dtd_test_flag_dont_track.c;${COMMON_DATA}")
parsec_addtest(C dtd_test_multiple_handle_wait "dtd_test_multiple_handle_wait.c")
parsec_addtest(C dtd_test_broadcast "dtd_test_broadcast.c;${COMMON_DATA}")
parsec_addtest(C dtd_test_reduce "dtd_test_reduce.c;${COMMON_DATA}")
parsec_addtest(C dtd_test_allreduce "dtd_test_allreduce.c;${COMMON_DATA}")
parsec_addtest(C dtd_test_template_counter "dtd_test_template_counter.c;${COMMON_DATA}")
parsec_addtest(C dtd_test_untie "dtd_test_untie.c;${COMMON_DATA}")


#
# Shared Memory Testings
#
add_test(shm_dtd_test_task_generation ${SHM_TEST_CMD_LIST} ./dtd_test_task_generation)
add_test(shm_dtd_test_task_inserting_task ${SHM_TEST_CMD_LIST} ./dtd_test_task_inserting_task)
add_test(shm_dtd_test_task_insertion ${SHM_TEST_CMD_LIST} ./dtd_test_task_insertion)
add_test(shm_dtd_test_war ${SHM_TEST_CMD_LIST} ./dtd_test_war)

#
# Distributed Memory Testings
#
if( MPI_C_FOUND )
  add_test(mpi_dtd_test_pingpong ${MPI_TEST_CMD_LIST} -np 2 ./dtd_test_pingpong)
  add_test(mpi_dtd_test_task_generation ${MPI_TEST_CMD_LIST} -np 1 ./dtd_test_task_generation)
  add_test(mpi_dtd_test_task_inserting_task ${MPI_TEST_CMD_LIST} -np 1 ./dtd_test_task_inserting_task)
  add_test(mpi_dtd_test_task_insertion ${MPI_TEST_CMD_LIST} -np 1 ./dtd_test_task_insertion)
  add_test(mpi_dtd_test_war ${MPI_TEST_CMD_LIST} -np 1 ./dtd_test_war)
endif( MPI_C_FOUND )
