include(ParsecCompilePTG)
include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(BEFORE "${CMAKE_CURRENT_BINARY_DIR}")

parsec_compile_ptg(two_dim_band_generated "${CMAKE_CURRENT_SOURCE_DIR}/two_dim_band.jdf")
parsec_compile_ptg(two_dim_band_free_generated "${CMAKE_CURRENT_SOURCE_DIR}/two_dim_band_free.jdf")

parsec_addtest(C testing_band "${two_dim_band_generated};${two_dim_band_free_generated};main.c")

add_test(shm_two_dim_band ${MPI_TEST_CMD_LIST} 8 ./testing_band -N 3200 -T 160 -P 4 -s 5 -S 10 -p 2 -f 2 -F 10 -b 2)
