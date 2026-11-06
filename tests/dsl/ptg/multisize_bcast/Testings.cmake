#
# Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
#

# Broadcasting tiles of different sizes reshapes them, which a build without a
# communication backend cannot represent: every derived type collapses to
# PARSEC_DATATYPE_NULL. Even the single process flavor needs a distributed build.
if( MPI_C_FOUND )
  parsec_addtest_cmd(dsl/ptg/multisize_bcast ${SHM_TEST_CMD_LIST} dsl/ptg/multisize_bcast/check_multisize_bcast)
  parsec_addtest_cmd(dsl/ptg/multisize_bcast:mp ${MPI_TEST_CMD_LIST} 4 dsl/ptg/multisize_bcast/check_multisize_bcast)
endif( MPI_C_FOUND)
