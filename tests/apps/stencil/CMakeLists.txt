file(COPY loop_gen_1D DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

parsec_addtest_executable(C testing_stencil_1D SOURCES stencil_internal.c testing_stencil_1D.c)
target_include_directories(testing_stencil_1D PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
target_ptg_sources(testing_stencil_1D PRIVATE "stencil_1D.jdf")
target_link_libraries(testing_stencil_1D PRIVATE m)

add_test(stencil_1D_shm ${SHM_TEST_CMD_LIST} ./testing_stencil_1D -t 100 -T 100 -N 1000 -M 1000 -I 10 -R 2 -m 1)
if( MPI_C_FOUND )
    add_test(stencil_1D_mpi ${MPI_TEST_CMD_LIST} 8 ./testing_stencil_1D -t 100 -T 100 -N 1000 -M 1000 -I 10 -R 2 -m 1)
endif( MPI_C_FOUND )
