include(ParsecCompilePTG)

parsec_addtest(C testing_band "main.c")
target_include_directories(testing_band PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
target_ptg_sources(testing_band PRIVATE "two_dim_band.jdf;two_dim_band_free.jdf")

add_test(shm_two_dim_band ${MPI_TEST_CMD_LIST} 8 ./testing_band -N 3200 -T 160 -P 4 -s 5 -S 10 -p 2 -f 2 -F 10 -b 2)
