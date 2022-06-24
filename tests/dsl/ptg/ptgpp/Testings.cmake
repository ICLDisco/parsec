include(ParsecCompilePTG)

#
# Test that NULL as output returns an error
#
parsec_addtest_cmd(dsl/ptg/ptgpp/output_NULL
  COMMAND ${PARSEC_PTGPP_EXECUTABLE} ${PARSEC_PTGPP_FLAGS} -E -i ${CMAKE_CURRENT_SOURCE_DIR}/dsl/ptg/ptgpp/output_NULL.jdf -o output_NULL -f output_NULL)

parsec_addtest_cmd(dsl/ptg/ptgpp/output_NULL_true
  COMMAND ${PARSEC_PTGPP_EXECUTABLE} ${PARSEC_PTGPP_FLAGS} -E -i ${CMAKE_CURRENT_SOURCE_DIR}/dsl/ptg/ptgpp/output_NULL_true.jdf -o output_NULL_true -f output_NULL_true)

parsec_addtest_cmd(dsl/ptg/ptgpp/output_NULL_false
  COMMAND ${PARSEC_PTGPP_EXECUTABLE} ${PARSEC_PTGPP_FLAGS} -E -i ${CMAKE_CURRENT_SOURCE_DIR}/dsl/ptg/ptgpp/output_NULL_false.jdf -o output_NULL_false -f output_NULL_false)

set_tests_properties(dsl/ptg/ptgpp/output_NULL dsl/ptg/ptgpp/output_NULL_true dsl/ptg/ptgpp/output_NULL_false
  PROPERTIES
  DEPENDS "${PARSEC_PTGPP_EXECUTABLE}"
  PASS_REGULAR_EXPRESSION "NULL data only supported in IN dependencies.")

#
# Test that NEW as output returns an error
#
parsec_addtest_cmd(dsl/ptg/ptgpp/output_NEW
  COMMAND ${PARSEC_PTGPP_EXECUTABLE} ${PARSEC_PTGPP_FLAGS} -E -i ${CMAKE_CURRENT_SOURCE_DIR}/dsl/ptg/ptgpp/output_NEW.jdf -o output_NEW -f output_NEW)

parsec_addtest_cmd(dsl/ptg/ptgpp/output_NEW_true
  COMMAND ${PARSEC_PTGPP_EXECUTABLE} ${PARSEC_PTGPP_FLAGS} -E -i ${CMAKE_CURRENT_SOURCE_DIR}/dsl/ptg/ptgpp/output_NEW_true.jdf -o output_NEW_true -f output_NEW_true)

parsec_addtest_cmd(dsl/ptg/ptgpp/output_NEW_false
  COMMAND ${PARSEC_PTGPP_EXECUTABLE} ${PARSEC_PTGPP_FLAGS} -E -i ${CMAKE_CURRENT_SOURCE_DIR}/dsl/ptg/ptgpp/output_NEW_false.jdf -o output_NEW_false -f output_NEW_false)

set_tests_properties(dsl/ptg/ptgpp/output_NEW dsl/ptg/ptgpp/output_NEW_true dsl/ptg/ptgpp/output_NEW_false
  PROPERTIES
  DEPENDS "${PARSEC_PTGPP_EXECUTABLE}"
  PASS_REGULAR_EXPRESSION "Automatic data allocation with NEW only supported in IN dependencies."
  )

#
# Test that a NULL cannot be forwarded
#
parsec_addtest_cmd(dsl/ptg/ptgpp/forward_RW_NULL   ${SHM_TEST_CMD_LIST}       dsl/ptg/ptgpp/jdf_forward_RW_NULL)
parsec_addtest_cmd(dsl/ptg/ptgpp/forward_READ_NULL ${SHM_TEST_CMD_LIST}       dsl/ptg/ptgpp/jdf_forward_READ_NULL)
set_tests_properties(
  dsl/ptg/ptgpp/forward_RW_NULL
  dsl/ptg/ptgpp/forward_READ_NULL
  PROPERTIES
  PASS_REGULAR_EXPRESSION "A NULL is forwarded"
)

parsec_addtest_cmd(dsl/ptg/ptgpp/write_check ${SHM_TEST_CMD_LIST} dsl/ptg/ptgpp/write_check)

if( MPI_C_FOUND )
  parsec_addtest_cmd(dsl/ptg/ptgpp/forward_RW_NULL:mp   ${MPI_TEST_CMD_LIST} 2 dsl/ptg/ptgpp/jdf_forward_RW_NULL)
  parsec_addtest_cmd(dsl/ptg/ptgpp/forward_READ_NULL:mp ${MPI_TEST_CMD_LIST} 2 dsl/ptg/ptgpp/jdf_forward_READ_NULL)
  set_tests_properties(
    dsl/ptg/ptgpp/forward_RW_NULL:mp
    dsl/ptg/ptgpp/forward_READ_NULL:mp
    PROPERTIES
    PASS_REGULAR_EXPRESSION "A NULL is forwarded"
  )
  parsec_addtest_cmd(dsl/ptg/ptgpp/write_check:mp ${MPI_TEST_CMD_LIST} 4 dsl/ptg/ptgpp/write_check)
endif( MPI_C_FOUND )

#
# Test to validate the number of input and output dependencies.
# Should fail is PaRSEC is compiled without support for more than 20
# output dependencies.
#
parsec_addtest_cmd(dsl/ptg/ptgpp/must_fail_too_many_in_deps
  COMMAND ${CMAKE_COMMAND} --build . --target must_fail_too_many_in_deps ${SHM_TEST_CMD_LIST} dsl/ptg/ptgpp/too_many_in_deps
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(dsl/ptg/ptgpp/must_fail_too_many_in_deps PROPERTIES WILL_FAIL TRUE)

parsec_addtest_cmd(dsl/ptg/ptgpp/must_fail_too_many_out_deps
          COMMAND ${CMAKE_COMMAND} --build . --target must_fail_too_many_out_deps ${SHM_TEST_CMD_LIST} dsl/ptg/ptgpp/too_many_out_deps
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(dsl/ptg/ptgpp/must_fail_too_many_out_deps PROPERTIES WILL_FAIL TRUE)

parsec_addtest_cmd(dsl/ptg/ptgpp/must_fail_too_many_read_flows
          COMMAND ${CMAKE_COMMAND} --build . --target must_fail_too_many_read_flows ${SHM_TEST_CMD_LIST} dsl/ptg/ptgpp/too_many_read_flows
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        set_tests_properties(dsl/ptg/ptgpp/must_fail_too_many_read_flows PROPERTIES WILL_FAIL TRUE)

parsec_addtest_cmd(dsl/ptg/ptgpp/must_fail_too_many_write_flows
                    COMMAND ${CMAKE_COMMAND} --build . --target must_fail_too_many_write_flows ${SHM_TEST_CMD_LIST} dsl/ptg/ptgpp/too_many_write_flows
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(dsl/ptg/ptgpp/must_fail_too_many_write_flows PROPERTIES WILL_FAIL TRUE)

parsec_addtest_cmd(dsl/ptg/ptgpp/must_fail_too_many_local_vars
                    COMMAND ${CMAKE_COMMAND} --build . --target must_fail_too_many_local_vars ${SHM_TEST_CMD_LIST} dsl/ptg/ptgpp/too_many_local_vars
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(dsl/ptg/ptgpp/must_fail_too_many_local_vars PROPERTIES WILL_FAIL TRUE)
