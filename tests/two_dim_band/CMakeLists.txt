include(ParsecCompilePTG)

parsec_addtest_executable(C testing_band SOURCES main.c)
target_include_directories(testing_band PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
target_ptg_sources(testing_band PRIVATE "two_dim_band.jdf;two_dim_band_free.jdf")

if( MPI_C_FOUND )
  parsec_addtest_cmd(shm_two_dim_band ${MPI_TEST_CMD_LIST} 8 ./testing_band -N 3200 -T 160 -P 4 -s 5 -S 10 -p 2 -f 2 -F 10 -b 2)
else( MPI_C_FOUND )
  parsec_addtest_cmd(shm_two_dim_band ${SHM_TEST_CMD_LIST} ./testing_band -N 3200 -T 160 -b 2)
endif( MPI_C_FOUND )
