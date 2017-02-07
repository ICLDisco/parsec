include(RulesJDF)

# Test that NULL as output returns an error
add_test(NAME jdf_output_NULL
  COMMAND ${parsec_ptgpp_EXE} ${PARSEC_PTGPP_CFLAGS} ${ADDITIONAL_PTGPP_CFLAGS} -E -i ${CMAKE_CURRENT_SOURCE_DIR}/output_NULL.jdf -o output_NULL -f output_NULL)

set_tests_properties(jdf_output_NULL
  PROPERTIES
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output_NULL.jdf)

add_test(NAME jdf_output_NULL_true
  COMMAND ${parsec_ptgpp_EXE} ${PARSEC_PTGPP_CFLAGS} ${ADDITIONAL_PTGPP_CFLAGS} -E -i ${CMAKE_CURRENT_SOURCE_DIR}/output_NULL_true.jdf -o output_NULL_true -f output_NULL_true)

set_tests_properties(jdf_output_NULL_true
  PROPERTIES
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output_NULL_true.jdf)

add_test(NAME jdf_output_NULL_false
  COMMAND ${parsec_ptgpp_EXE} ${PARSEC_PTGPP_CFLAGS} ${ADDITIONAL_PTGPP_CFLAGS} -E -i ${CMAKE_CURRENT_SOURCE_DIR}/output_NULL_false.jdf -o output_NULL_false -f output_NULL_false)

set_tests_properties(jdf_output_NULL_false
  PROPERTIES
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output_NULL_false.jdf)

set_tests_properties(jdf_output_NULL jdf_output_NULL_true jdf_output_NULL_false
  PROPERTIES
  DEPENDS "${parsec_ptgpp_EXE}"
  PASS_REGULAR_EXPRESSION "NULL data only supported in IN dependencies."
  )

# Test that NEW as output returns an error
add_test(NAME jdf_output_NEW
  COMMAND ${parsec_ptgpp_EXE} ${PARSEC_PTGPP_CFLAGS} ${ADDITIONAL_PTGPP_CFLAGS} -E -i ${CMAKE_CURRENT_SOURCE_DIR}/output_NEW.jdf -o output_NEW -f output_NEW)

set_tests_properties(jdf_output_NEW
  PROPERTIES
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output_NEW.jdf)

add_test(NAME jdf_output_NEW_true
  COMMAND ${parsec_ptgpp_EXE} ${PARSEC_PTGPP_CFLAGS} ${ADDITIONAL_PTGPP_CFLAGS} -E -i ${CMAKE_CURRENT_SOURCE_DIR}/output_NEW_true.jdf -o output_NEW_true -f output_NEW_true)

set_tests_properties(jdf_output_NEW_true
  PROPERTIES
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output_NEW_true.jdf)

add_test(NAME jdf_output_NEW_false
  COMMAND ${parsec_ptgpp_EXE} ${PARSEC_PTGPP_CFLAGS} ${ADDITIONAL_PTGPP_CFLAGS} -E -i ${CMAKE_CURRENT_SOURCE_DIR}/output_NEW_false.jdf -o output_NEW_false -f output_NEW_false)

set_tests_properties(jdf_output_NEW_false
  PROPERTIES
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/output_NEW_false.jdf)

set_tests_properties(jdf_output_NEW jdf_output_NEW_true jdf_output_NEW_false
  PROPERTIES
  DEPENDS "${parsec_ptgpp_EXE}"
  PASS_REGULAR_EXPRESSION "Automatic data allocation with NEW only supported in IN dependencies."
  )

# Test that a NULL cannot be forwarded

jdf_rules(forward_RW_NULL "${CMAKE_CURRENT_SOURCE_DIR}/forward_RW_NULL.jdf")
parsec_addtest(C jdf_forward_RW_NULL "${forward_RW_NULL}")

jdf_rules(forward_READ_NULL "${CMAKE_CURRENT_SOURCE_DIR}/forward_READ_NULL.jdf")
parsec_addtest(C jdf_forward_READ_NULL "${forward_READ_NULL}")

add_test(shm_jdf_forward_RW_NULL   ${SHM_TEST_CMD_LIST}       ./jdf_forward_RW_NULL)
add_test(shm_jdf_forward_READ_NULL ${SHM_TEST_CMD_LIST}       ./jdf_forward_READ_NULL)
if( MPI_C_FOUND )
  add_test(mpi_jdf_forward_RW_NULL   ${MPI_TEST_CMD_LIST} -np 2 ./jdf_forward_RW_NULL)
  add_test(mpi_jdf_forward_READ_NULL ${MPI_TEST_CMD_LIST} -np 2 ./jdf_forward_READ_NULL)
  set_tests_properties(
    mpi_jdf_forward_RW_NULL
    mpi_jdf_forward_READ_NULL
    PROPERTIES
    PASS_REGULAR_EXPRESSION "A NULL is forwarded"
  )
endif( MPI_C_FOUND )

set_tests_properties(
  shm_jdf_forward_RW_NULL
  shm_jdf_forward_READ_NULL
  PROPERTIES
  PASS_REGULAR_EXPRESSION "A NULL is forwarded"
)

