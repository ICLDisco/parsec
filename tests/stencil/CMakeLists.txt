file(COPY loop_gen_1D DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

parsec_addtest(C testing_stencil_1D "stencil_internal.c;testing_stencil_1D.c")
target_include_directories(testing_stencil_1D PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
target_ptg_sources(testing_stencil_1D PRIVATE "stencil_1D.jdf")
target_link_libraries(testing_stencil_1D PRIVATE m)

if( MPI_C_FOUND )
  add_test(mpi_stencil_1D ${MPI_TEST_CMD_LIST} 4 ./testing_stencil_1D -M 64 -N 64 -t 4 -I 1)
else( MPI_C_FOUND )
  add_test(shm_stencil_1D ${SHM_TEST_CMD_LIST} ./testing_stencil_1D -M 64 -N 64 -t 4 -I 1)
endif( MPI_C_FOUND )
