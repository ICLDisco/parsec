#
# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
#

# Taskpools given to one context, merged tile by tile in the order they were
# added: map, in-place chain, read-only broadcast, map. Any task that sees a
# tile out of order, or any tile with a wrong final value, fails the test.
parsec_addtest_cmd(dsl/ptg/auto_merge ${SHM_TEST_CMD_LIST} dsl/ptg/auto_merge/auto_merge -g 0 -d 500 -r 2)
# Same, every taskpool added before the context starts
parsec_addtest_cmd(dsl/ptg/auto_merge:added_first ${SHM_TEST_CMD_LIST} dsl/ptg/auto_merge/auto_merge -g 0 -d 500 -a -s MCBMBCM)
if( MPI_C_FOUND )
  # The users of a broadcast run on other ranks: the tile is sent while a later
  # taskpool waits to overwrite it
  parsec_addtest_cmd(dsl/ptg/auto_merge:mp ${MPI_TEST_CMD_LIST} 4 dsl/ptg/auto_merge/auto_merge -g 0 -d 500 -M 8 -N 4 -b 64 -r 2)
endif( MPI_C_FOUND )
if( PARSEC_HAVE_CUDA AND CMAKE_CUDA_COMPILER )
  # The maps run on the GPU, the chain and the broadcast on the CPU: every tile
  # crosses between the device and the host from one taskpool to the next
  parsec_addtest_cmd(dsl/ptg/auto_merge:gpu ${SHM_TEST_CMD_LIST} ${CTEST_CUDA_LAUNCHER_OPTIONS} dsl/ptg/auto_merge/auto_merge -g 1 -d 500 -s MMCMBMM -r 2 -- --mca device_cuda_enabled 1 --mca device_show_statistics 1)
  if( MPI_C_FOUND )
    parsec_addtest_cmd(dsl/ptg/auto_merge:mp:gpu ${MPI_TEST_CMD_LIST} 2 ${CTEST_CUDA_LAUNCHER_OPTIONS} dsl/ptg/auto_merge/auto_merge -g 1 -d 500 -M 8 -N 4 -b 64 -s MMCMBMM -- --mca device_cuda_enabled 1 --mca device_show_statistics 1)
  endif( MPI_C_FOUND )
endif()
