#
# Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
#

include(runtime/scheduling/Testings.cmake)
include(runtime/cuda/Testings.cmake)

if( MPI_C_FOUND )
  parsec_addtest_cmd(runtime/multichain:mp ${MPI_TEST_CMD_LIST} 4 runtime/multichain -l=1 -c=2)
endif( MPI_C_FOUND )
# an arena caching chunks of several elements by power-of-two class
parsec_addtest_cmd(runtime/arena_classes ${SHM_TEST_CMD_LIST} runtime/arena_classes)
