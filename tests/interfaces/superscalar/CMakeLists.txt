include_directories(${CMAKE_CURRENT_BINARY_DIR})
set(COMMON_DATA "common_data.c")
parsec_addtest_executable(C dtd_test_pingpong SOURCES dtd_test_pingpong.c ${COMMON_DATA})
parsec_addtest_executable(C dtd_test_task_generation SOURCES dtd_test_task_generation.c ${COMMON_DATA})
parsec_addtest_executable(C dtd_test_war SOURCES dtd_test_war.c ${COMMON_DATA})
parsec_addtest_executable(C dtd_test_task_insertion SOURCES dtd_test_task_insertion.c)
parsec_addtest_executable(C dtd_test_null_as_tile SOURCES dtd_test_null_as_tile.c)
parsec_addtest_executable(C dtd_test_task_inserting_task SOURCES dtd_test_task_inserting_task.c)
parsec_addtest_executable(C dtd_test_flag_dont_track SOURCES dtd_test_flag_dont_track.c ${COMMON_DATA})
parsec_addtest_executable(C dtd_test_multiple_handle_wait SOURCES dtd_test_multiple_handle_wait.c)
parsec_addtest_executable(C dtd_test_broadcast SOURCES dtd_test_broadcast.c ${COMMON_DATA})
parsec_addtest_executable(C dtd_test_reduce SOURCES dtd_test_reduce.c ${COMMON_DATA})
parsec_addtest_executable(C dtd_test_allreduce SOURCES dtd_test_allreduce.c ${COMMON_DATA})
parsec_addtest_executable(C dtd_test_template_counter SOURCES dtd_test_template_counter.c ${COMMON_DATA})
parsec_addtest_executable(C dtd_test_untie SOURCES dtd_test_untie.c ${COMMON_DATA})
parsec_addtest_executable(C dtd_test_hierarchy SOURCES dtd_test_hierarchy.c ${COMMON_DATA})
parsec_addtest_executable(C dtd_test_task_placement SOURCES dtd_test_task_placement.c ${COMMON_DATA})
parsec_addtest_executable(C dtd_test_insert_task_interface SOURCES dtd_test_insert_task_interface.c ${COMMON_DATA})
parsec_addtest_executable(C dtd_test_data_flush SOURCES dtd_test_data_flush.c ${COMMON_DATA})
parsec_addtest_executable(C dtd_test_global_id_for_dc_assumed SOURCES dtd_test_global_id_for_dc_assumed.c ${COMMON_DATA})
parsec_addtest_executable(C dtd_test_explicit_task_creation SOURCES dtd_test_explicit_task_creation.c ${COMMON_DATA})
parsec_addtest_executable(C dtd_test_tp_enqueue_dequeue SOURCES dtd_test_tp_enqueue_dequeue.c)
parsec_addtest_executable(C dtd_test_interleave_actions SOURCES dtd_test_interleave_actions.c)

#
# Shared Memory Testings
#
parsec_addtest_cmd(unit_dtd_task_generation_shm ${SHM_TEST_CMD_LIST} ./dtd_test_task_generation)
parsec_addtest_cmd(unit_dtd_task_inserting_task_shm ${SHM_TEST_CMD_LIST} ./dtd_test_task_inserting_task)
parsec_addtest_cmd(unit_dtd_task_insertion_shm ${SHM_TEST_CMD_LIST} ./dtd_test_task_insertion)
parsec_addtest_cmd(unit_dtd_war_shm ${SHM_TEST_CMD_LIST} ./dtd_test_war)

#
# Distributed Memory Testings
#
if( MPI_C_FOUND )
  parsec_addtest_cmd(unit_dtd_pingpong_mpi ${MPI_TEST_CMD_LIST} 2 ./dtd_test_pingpong)
  parsec_addtest_cmd(unit_dtd_task_inserting_task_mpi ${MPI_TEST_CMD_LIST} 4 ./dtd_test_task_inserting_task)
  parsec_addtest_cmd(unit_dtd_task_insertion_mpi ${MPI_TEST_CMD_LIST} 4 ./dtd_test_task_insertion)
  parsec_addtest_cmd(unit_dtd_war_mpi ${MPI_TEST_CMD_LIST} 4 ./dtd_test_war)
  parsec_addtest_cmd(unit_dtd_interleave_actions_mpi ${MPI_TEST_CMD_LIST} 4 ./dtd_test_interleave_actions)
endif( MPI_C_FOUND )
