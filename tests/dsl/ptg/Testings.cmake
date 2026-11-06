include(${CMAKE_CURRENT_LIST_DIR}/ptgpp/Testings.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/user-defined-functions/Testings.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/branching/Testings.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/multisize_bcast/Testings.cmake)

parsec_addtest_cmd(dsl/ptg/startup1 ${SHM_TEST_CMD_LIST} dsl/ptg/startup -i=10 -j=10 -k=10 -v=5)
parsec_addtest_cmd(dsl/ptg/startup2 ${SHM_TEST_CMD_LIST} dsl/ptg/startup -i=10 -j=20 -k=30 -v=5)
parsec_addtest_cmd(dsl/ptg/startup3 ${SHM_TEST_CMD_LIST} dsl/ptg/startup -i=30 -j=30 -k=30 -v=5)
parsec_addtest_cmd(dsl/ptg/strange ${SHM_TEST_CMD_LIST} dsl/ptg/strange)
# [type_remote = AUTO] only matters when data crosses processes. The single
# process flavor checks the generated code still runs when everything is local.
if( MPI_C_FOUND )
  parsec_addtest_cmd(dsl/ptg/receive_auto ${SHM_TEST_CMD_LIST} dsl/ptg/receive_auto)
  parsec_addtest_cmd(dsl/ptg/receive_auto:mp ${MPI_TEST_CMD_LIST} 2 dsl/ptg/receive_auto)
  # rendezvous only, no data embedded in the activation message
  parsec_addtest_cmd(dsl/ptg/receive_auto:rdv:mp ${MPI_TEST_CMD_LIST} 2 dsl/ptg/receive_auto -- --mca runtime_comm_short_limit 0)
  # payloads larger than the largest caching bucket
  parsec_addtest_cmd(dsl/ptg/receive_auto:large:mp ${MPI_TEST_CMD_LIST} 2 dsl/ptg/receive_auto -n=8 -b=3145728)
  if(PARSEC_HAVE_CUDA)
    # SEND and RECV have CUDA incarnations: produce the data on a GPU, check it
    # once staged in GPU memory, and keep checking the datatypes on the CPU
    # incarnation. -G=1 gives each process its own GPU (none is shared).
    parsec_addtest_cmd(dsl/ptg/receive_auto:gpu:mp ${MPI_TEST_CMD_LIST} 2 ${CTEST_CUDA_LAUNCHER_OPTIONS} dsl/ptg/receive_auto -g=1 -G=1 -- --mca device_cuda_enabled 1)
    parsec_addtest_cmd(dsl/ptg/receive_auto:cpu:mp ${MPI_TEST_CMD_LIST} 2 dsl/ptg/receive_auto -- --mca device_cuda_enabled 0)
    if(PARSEC_MPI_IS_GPU_AWARE)
      # the tiles prefer the GPU: the data is received straight into GPU memory
      parsec_addtest_cmd(dsl/ptg/receive_auto:gpudirect:mp ${MPI_TEST_CMD_LIST} 2 ${CTEST_CUDA_LAUNCHER_OPTIONS} dsl/ptg/receive_auto -g=1 -G=1 -d=1 -- --mca device_cuda_enabled 1 --mca mpi_gpu_aware 3 --mca runtime_comm_short_limit 0)
    endif(PARSEC_MPI_IS_GPU_AWARE)
  endif(PARSEC_HAVE_CUDA)
  parsec_addtest_cmd(dsl/ptg/receive_auto_mixed ${SHM_TEST_CMD_LIST} dsl/ptg/receive_auto_mixed)
  parsec_addtest_cmd(dsl/ptg/receive_auto_mixed:mp ${MPI_TEST_CMD_LIST} 2 dsl/ptg/receive_auto_mixed)
  parsec_addtest_cmd(dsl/ptg/receive_auto_mixed:rdv:mp ${MPI_TEST_CMD_LIST} 2 dsl/ptg/receive_auto_mixed -- --mca runtime_comm_short_limit 0)
endif( MPI_C_FOUND )
if(PARSEC_HAVE_DEV_CAPABILITY_BATCH)
  parsec_addtest_cmd(dsl/ptg/batch_cpu ${SHM_TEST_CMD_LIST} dsl/ptg/batch_cpu)
endif(PARSEC_HAVE_DEV_CAPABILITY_BATCH)
