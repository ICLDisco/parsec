#
# Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
#

add_subdirectory(scheduling)
add_Subdirectory(cuda)

if( MPI_C_FOUND )
  parsec_addtest_executable(C multichain)
  target_ptg_sources(multichain PRIVATE "multichain.jdf")
endif( MPI_C_FOUND )

parsec_addtest_executable(C dtt_bug_replicator SOURCES dtt_bug_replicator_ex.c)
target_link_libraries(dtt_bug_replicator PRIVATE tests_runtime_common)
target_ptg_sources(dtt_bug_replicator PRIVATE "dtt_bug_replicator.jdf")
