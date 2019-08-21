include(ParsecCompilePTG)

parsec_compile_ptg(project_generated "${CMAKE_CURRENT_SOURCE_DIR}/project.jdf;${CMAKE_CURRENT_SOURCE_DIR}/walk.jdf")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

parsec_addtest(C project "main.c;${project_generated};tree_dist.c")
target_link_libraries(project PRIVATE Threads::Threads)

add_test(unit_haar_tree_shm ${SHM_TEST_CMD_LIST} ./project -x)
if( MPI_C_FOUND )
  add_test(unit_haar_tree_mpi ${MPI_TEST_CMD_LIST} 4 ./project -x)
  set_tests_properties(unit_haar_tree_mpi PROPERTIES DEPENDS launch_mpi)
endif( MPI_C_FOUND )

