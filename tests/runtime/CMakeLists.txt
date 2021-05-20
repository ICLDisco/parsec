add_subdirectory(scheduling)

if( MPI_C_FOUND )
  parsec_addtest_executable(C multichain)
  target_ptg_sources(multichain PRIVATE "multichain.jdf")
endif( MPI_C_FOUND )

parsec_addtest_executable(C dtt_bug_replicator SOURCES dtt_bug_replicator_ex.c)
target_ptg_sources(dtt_bug_replicator PRIVATE "dtt_bug_replicator.jdf")


