include(RulesJDF)

jdf_rules(project_generated "${CMAKE_CURRENT_SOURCE_DIR}/project.jdf;${CMAKE_CURRENT_SOURCE_DIR}/walk.jdf")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

parsec_addtest(C project "main.c;${project_generated};tree_dist.c")

add_test(shm_haar_tree "./project" "-x")
if( MPI_C_FOUND )
  find_program(BINTRUE true)
  set(PROCS "-np;4")

  # Check MPI
  add_test(mpi_test ${MPI_TEST_CMD_LIST} ${PROCS} ${BINTRUE})
  add_test(mpi_haar_tree ${MPI_TEST_CMD_LIST} ${PROCS}  ./project -x)
  set_tests_properties(mpi_haar_tree PROPERTIES DEPENDS mpi_test)
endif( MPI_C_FOUND )
