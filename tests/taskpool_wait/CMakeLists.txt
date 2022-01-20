include(ParsecCompilePTG)

parsec_addtest_executable(C taskpool_wait SOURCES main.c dtd_tp.c)
target_include_directories(taskpool_wait PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
target_ptg_sources(taskpool_wait PRIVATE "ptg_tp.jdf")

if( MPI_C_FOUND )
    parsec_addtest_cmd(shm_two_dim_band ${MPI_TEST_CMD_LIST} 8 ./taskpool_wait)
else( MPI_C_FOUND )
    parsec_addtest_cmd(shm_two_dim_band ${SHM_TEST_CMD_LIST} ./taskpool_wait)
endif( MPI_C_FOUND )
