#
# Copyright (c) 2026      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
#

# Payloads smaller than the receive type, forwarded by the chain and binomial
# broadcasts. The consumers live on other processes and the receive types are
# derived types, so this needs the MPI backend and several processes.
if( MPI_C_FOUND )
  parsec_addtest_cmd(dsl/ptg/partial_bcast:mp:star ${MPI_TEST_CMD_LIST} 4 dsl/ptg/partial_bcast/partial_bcast --mca runtime_comm_coll_bcast 0)
  if( PARSEC_DIST_COLLECTIVES )
    parsec_addtest_cmd(dsl/ptg/partial_bcast:mp:chain ${MPI_TEST_CMD_LIST} 4 dsl/ptg/partial_bcast/partial_bcast --mca runtime_comm_coll_bcast 1)
    parsec_addtest_cmd(dsl/ptg/partial_bcast:mp:binomial ${MPI_TEST_CMD_LIST} 4 dsl/ptg/partial_bcast/partial_bcast --mca runtime_comm_coll_bcast 2)
  endif( PARSEC_DIST_COLLECTIVES )
endif( MPI_C_FOUND )
