FOREACH(_sched ${MCA_sched})
    parsec_addtest_cmd(runtime/scheduling:sp:${_sched} ${MPI_TEST_CMD_LIST} 1 runtime/scheduling/schedmicro -t 10 -l 8 -n 512 -- --mca mca_sched ${_sched})
ENDFOREACH()

if( MPI_C_FOUND )
  FOREACH(_sched ${MCA_sched})
        parsec_addtest_cmd(runtime/scheduling:mp:${_sched} ${MPI_TEST_CMD_LIST} 2 runtime/scheduling/schedmicro -t 10 -l 8 -n 512 -- --mca mca_sched ${_sched})
    ENDFOREACH()
endif( MPI_C_FOUND )
