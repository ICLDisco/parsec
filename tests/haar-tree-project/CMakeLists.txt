include(ParsecCompilePTG)

parsec_addtest_executable(C project SOURCES main.c ${project_generated} tree_dist.c)
target_include_directories(project PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
target_ptg_sources(project PRIVATE "project.jdf;walk.jdf")
target_link_libraries(project PRIVATE Threads::Threads m)

parsec_addtest_cmd(unit_haar_tree_shm ${SHM_TEST_CMD_LIST} ./project -x)
if( MPI_C_FOUND )
  parsec_addtest_cmd(unit_haar_tree_mpi ${MPI_TEST_CMD_LIST} 4 ./project -x)
  set_tests_properties(unit_haar_tree_mpi PROPERTIES DEPENDS launch_mpi)
endif( MPI_C_FOUND )

