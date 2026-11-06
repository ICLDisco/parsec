# Copyright (c) 2026 NVIDIA Corporation.  All rights reserved.

#
# Shared Memory Testings
#
parsec_addtest_cmd(dsl/dtd/empty ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_empty)
parsec_addtest_cmd(dsl/dtd/task_generation ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_task_generation)
parsec_addtest_cmd(dsl/dtd/task_inserting_task ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_task_inserting_task)
parsec_addtest_cmd(dsl/dtd/task_insertion ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_task_insertion)
parsec_addtest_cmd(dsl/dtd/war ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_war)
parsec_addtest_cmd(dsl/dtd/interleave_actions ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_interleave_actions)
parsec_addtest_cmd(dsl/dtd/allreduce ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_allreduce)
parsec_addtest_cmd(dsl/dtd/broadcast ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_broadcast)
parsec_addtest_cmd(dsl/dtd/data_flush ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_data_flush)
parsec_addtest_cmd(dsl/dtd/global_id_for_dc_assumed ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_global_id_for_dc_assumed)
parsec_addtest_cmd(dsl/dtd/hierarchy ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_hierarchy)
parsec_addtest_cmd(dsl/dtd/insert_task_interface ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_insert_task_interface)
parsec_addtest_cmd(dsl/dtd/flag_dont_track ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_flag_dont_track)
parsec_addtest_cmd(dsl/dtd/multiple_handle_wait ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_multiple_handle_wait)
parsec_addtest_cmd(dsl/dtd/null_as_tile ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_null_as_tile)
parsec_addtest_cmd(dsl/dtd/reduce ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_reduce)
parsec_addtest_cmd(dsl/dtd/explicit_task_creation ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_explicit_task_creation)
parsec_addtest_cmd(dsl/dtd/task_class_release ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_task_class_release)
parsec_addtest_cmd(dsl/dtd/template_counter ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_template_counter)
parsec_addtest_cmd(dsl/dtd/untie ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_untie)
parsec_addtest_cmd(dsl/dtd/new_tile:cpu ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_new_tile --mca device_cuda_enabled 0)
if(PARSEC_HAVE_CUDA AND CMAKE_CUDA_COMPILER)
  parsec_addtest_cmd(dsl/dtd/new_tile:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} dsl/dtd/dtd_test_new_tile --require-gpu --mca device_cuda_enabled 1 --mca device cuda)
  set_tests_properties(dsl/dtd/new_tile:gpu PROPERTIES FIXTURES_REQUIRED parsec_cuda_kernels_run)
  parsec_addtest_cmd(dsl/dtd/cuda_batch_status:gpu ${SHM_TEST_CMD_LIST}
                    ${CTEST_CUDA_LAUNCHER_OPTIONS}
                    dsl/dtd/dtd_test_cuda_again_async
                    --mca device_cuda_enabled 1
                    --mca device_cuda_mask 1
                    --mca device_enable_batching 1
                    --mca device cuda
                    # A batched hook here waits for a peer task that only
                    # another thread can submit, so one core livelocks.
                    --mca runtime_num_cores 2)
  # Only built when cuBLAS is available, hence the target test.
  if(TARGET dtd_test_simple_gemm)
    parsec_addtest_cmd(dsl/dtd/simple_gemm:gpu ${SHM_TEST_CMD_LIST}
                       ${CTEST_CUDA_LAUNCHER_OPTIONS}
                       dsl/dtd/dtd_test_simple_gemm
                       -M 2048 -N 2048 -K 2048 -m 512 -n 512 -k 512 -t 2
                       -- --mca device_cuda_enabled 1 --mca device cuda)
  endif(TARGET dtd_test_simple_gemm)
  # A single execution stream, where a hook returning AGAIN until a peer task
  # gets submitted must not hold back that peer, with and without delaying
  # batch-capable tasks behind work in flight.
  foreach(delay 0 1)
    parsec_addtest_cmd(dsl/dtd/cuda_batch_status:1stream:delay${delay}:gpu ${SHM_TEST_CMD_LIST}
                       ${CTEST_CUDA_LAUNCHER_OPTIONS}
                       dsl/dtd/dtd_test_cuda_again_async
                       --mca device_cuda_enabled 1
                       --mca device_cuda_mask 1
                       --mca device_enable_batching 1
                       --mca device_cuda_max_streams 3
                       --mca device_cuda_exec_delay_batch ${delay}
                       --mca device cuda
                       --mca runtime_num_cores 2)
  endforeach()
endif(PARSEC_HAVE_CUDA AND CMAKE_CUDA_COMPILER)
if(PARSEC_HAVE_DEV_CAPABILITY_BATCH)
  parsec_addtest_cmd(dsl/dtd/batch_cpu ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_batch_cpu)
  parsec_addtest_cmd(dsl/dtd/batch_cpu:disabled ${SHM_TEST_CMD_LIST} dsl/dtd/dtd_test_batch_cpu --mca device_enable_batching 0)
endif(PARSEC_HAVE_DEV_CAPABILITY_BATCH)

#
# Distributed Memory Testings
#
if( MPI_C_FOUND )
  parsec_addtest_cmd(dsl/dtd/empty:mp ${MPI_TEST_CMD_LIST} 2 dsl/dtd/dtd_test_empty)
  parsec_addtest_cmd(dsl/dtd/pingpong:mp ${MPI_TEST_CMD_LIST} 2 dsl/dtd/dtd_test_pingpong)
  parsec_addtest_cmd(dsl/dtd/task_inserting_task:mp ${MPI_TEST_CMD_LIST} 4 dsl/dtd/dtd_test_task_inserting_task)
  parsec_addtest_cmd(dsl/dtd/task_insertion:mp ${MPI_TEST_CMD_LIST} 4 dsl/dtd/dtd_test_task_insertion)
  parsec_addtest_cmd(dsl/dtd/war:mp ${MPI_TEST_CMD_LIST} 4 dsl/dtd/dtd_test_war)
  parsec_addtest_cmd(dsl/dtd/interleave_actions:mp ${MPI_TEST_CMD_LIST} 4 dsl/dtd/dtd_test_interleave_actions)
  parsec_addtest_cmd(dsl/dtd/allreduce:mp ${MPI_TEST_CMD_LIST} 4 dsl/dtd/dtd_test_allreduce)
  # A chain and a binomial tree only differ from a star once a peer has to
  # forward what it received, which takes more than two destinations. The
  # scenarios are what such a peer has to cope with, see the test itself. The
  # options of the test come after the -- that ends the ones of PaRSEC: the
  # command line parsing of PaRSEC stops at the first one it does not know.
  foreach(topology 0:star 1:chain 2:binomial)
    string(REPLACE ":" ";" topology_parts ${topology})
    list(GET topology_parts 0 topology_id)
    list(GET topology_parts 1 topology_name)
    foreach(scenario plain late writer multi multiset multisingle)
      parsec_addtest_cmd(dsl/dtd/broadcast:mp:${topology_name}:${scenario} ${MPI_TEST_CMD_LIST} 4 dsl/dtd/dtd_test_broadcast --mca runtime_comm_coll_bcast ${topology_id} -- --scenario ${scenario})
    endforeach()
    # the tree is laid out relative to the root, which is not rank 0 here
    parsec_addtest_cmd(dsl/dtd/broadcast:mp:${topology_name}:root3 ${MPI_TEST_CMD_LIST} 6 dsl/dtd/dtd_test_broadcast --mca runtime_comm_coll_bcast ${topology_id} -- --root 3)
    # a broadcast to more destinations than the limit is split in groups, each with its own tree
    if( NOT topology_id EQUAL 0 )
      parsec_addtest_cmd(dsl/dtd/broadcast:mp:${topology_name}:groups ${MPI_TEST_CMD_LIST} 6 dsl/dtd/dtd_test_broadcast --mca runtime_comm_coll_bcast ${topology_id} --mca runtime_comm_coll_bcast_group 2 -- --root 3)
      parsec_addtest_cmd(dsl/dtd/broadcast:mp:${topology_name}:groups:multiset ${MPI_TEST_CMD_LIST} 6 dsl/dtd/dtd_test_broadcast --mca runtime_comm_coll_bcast ${topology_id} --mca runtime_comm_coll_bcast_group 2 -- --root 3 --scenario multiset)
    endif()
  endforeach()
  # The topology can also be selected for a taskpool, whatever the default is.
  foreach(topology star binomial)
    parsec_addtest_cmd(dsl/dtd/broadcast:mp:taskpool:${topology} ${MPI_TEST_CMD_LIST} 6 dsl/dtd/dtd_test_broadcast --mca runtime_comm_coll_bcast 1 -- --topology ${topology} --root 3)
  endforeach()
  # The destinations of a broadcast travel as a list of ranks rather than as a
  # bitmask when there are fewer of them than words in the bitmask, which takes
  # more than 64 ranks for the smallest groups.
  parsec_addtest_cmd(dsl/dtd/broadcast:mp:chain:list ${MPI_TEST_CMD_LIST} 70 dsl/dtd/dtd_test_broadcast 1 --mca runtime_comm_coll_bcast 1 --mca runtime_comm_coll_bcast_group 2)
  parsec_addtest_cmd(dsl/dtd/data_flush:mp:2 ${MPI_TEST_CMD_LIST} 2 dsl/dtd/dtd_test_data_flush)
  parsec_addtest_cmd(dsl/dtd/data_flush:mp:3 ${MPI_TEST_CMD_LIST} 3 dsl/dtd/dtd_test_data_flush)
  parsec_addtest_cmd(dsl/dtd/global_id_for_dc_assumed:mp ${MPI_TEST_CMD_LIST} 2 dsl/dtd/dtd_test_global_id_for_dc_assumed)
  parsec_addtest_cmd(dsl/dtd/hierarchy:mp ${MPI_TEST_CMD_LIST} 2 dsl/dtd/dtd_test_hierarchy)
  parsec_addtest_cmd(dsl/dtd/multiple_handle_wait:mp ${MPI_TEST_CMD_LIST} 2 dsl/dtd/dtd_test_multiple_handle_wait)
  parsec_addtest_cmd(dsl/dtd/null_as_tile:mp ${MPI_TEST_CMD_LIST} 2 dsl/dtd/dtd_test_null_as_tile)
  parsec_addtest_cmd(dsl/dtd/reduce:mp ${MPI_TEST_CMD_LIST} 4 dsl/dtd/dtd_test_reduce)
  parsec_addtest_cmd(dsl/dtd/task_placement:mp ${MPI_TEST_CMD_LIST} 2 dsl/dtd/dtd_test_task_placement)
  parsec_addtest_cmd(dsl/dtd/template_counter:mp ${MPI_TEST_CMD_LIST} 2 dsl/dtd/dtd_test_template_counter)
  parsec_addtest_cmd(dsl/dtd/new_tile:mp:cpu ${MPI_TEST_CMD_LIST} 2 dsl/dtd/dtd_test_new_tile --mca device_cuda_enabled 0)
  if(PARSEC_HAVE_CUDA AND CMAKE_CUDA_COMPILER)
    parsec_addtest_cmd(dsl/dtd/new_tile:mp:gpu ${MPI_TEST_CMD_LIST} 2 ${CTEST_CUDA_LAUNCHER_OPTIONS} dsl/dtd/dtd_test_new_tile --require-gpu --mca device_cuda_enabled 1 --mca device cuda)
    set_tests_properties(dsl/dtd/new_tile:mp:gpu PROPERTIES FIXTURES_REQUIRED parsec_cuda_kernels_run)
  endif(PARSEC_HAVE_CUDA AND CMAKE_CUDA_COMPILER)
endif( MPI_C_FOUND )
