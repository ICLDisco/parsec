include(ParsecCompilePTG)

#
# Test that NULL as output returns an error
#
add_test(NAME unit_jdf_output_NULL_ptgpp
  COMMAND ${PARSEC_PTGPP_EXECUTABLE} ${PARSEC_PTGFLAGS} -E -i ${CMAKE_CURRENT_SOURCE_DIR}/output_NULL.jdf -o output_NULL -f output_NULL)

add_test(NAME unit_jdf_output_NULL_true_ptgpp
  COMMAND ${PARSEC_PTGPP_EXECUTABLE} ${PARSEC_PTGFLAGS} -E -i ${CMAKE_CURRENT_SOURCE_DIR}/output_NULL_true.jdf -o output_NULL_true -f output_NULL_true)

add_test(NAME unit_jdf_output_NULL_false_ptgpp
  COMMAND ${PARSEC_PTGPP_EXECUTABLE} ${PARSEC_PTGFLAGS} -E -i ${CMAKE_CURRENT_SOURCE_DIR}/output_NULL_false.jdf -o output_NULL_false -f output_NULL_false)

set_tests_properties(unit_jdf_output_NULL_ptgpp unit_jdf_output_NULL_true_ptgpp unit_jdf_output_NULL_false_ptgpp
  PROPERTIES
  DEPENDS "${PARSEC_PTGPP_EXECUTABLE}"
  PASS_REGULAR_EXPRESSION "NULL data only supported in IN dependencies.")

#
# Test that NEW as output returns an error
#
add_test(NAME unit_jdf_output_NEW_ptgpp
  COMMAND ${PARSEC_PTGPP_EXECUTABLE} ${PARSEC_PTGFLAGS} -E -i ${CMAKE_CURRENT_SOURCE_DIR}/output_NEW.jdf -o output_NEW -f output_NEW)

add_test(NAME unit_jdf_output_NEW_true_ptgpp
  COMMAND ${PARSEC_PTGPP_EXECUTABLE} ${PARSEC_PTGFLAGS} -E -i ${CMAKE_CURRENT_SOURCE_DIR}/output_NEW_true.jdf -o output_NEW_true -f output_NEW_true)

add_test(NAME unit_jdf_output_NEW_false_ptgpp
  COMMAND ${PARSEC_PTGPP_EXECUTABLE} ${PARSEC_PTGFLAGS} -E -i ${CMAKE_CURRENT_SOURCE_DIR}/output_NEW_false.jdf -o output_NEW_false -f output_NEW_false)

set_tests_properties(unit_jdf_output_NEW_ptgpp unit_jdf_output_NEW_true_ptgpp unit_jdf_output_NEW_false_ptgpp
  PROPERTIES
  DEPENDS "${PARSEC_PTGPP_EXECUTABLE}"
  PASS_REGULAR_EXPRESSION "Automatic data allocation with NEW only supported in IN dependencies."
  )

#
# Test that a NULL cannot be forwarded
#
parsec_addtest(C jdf_forward_RW_NULL)
target_ptg_sources(jdf_forward_RW_NULL PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/forward_RW_NULL.jdf")

parsec_addtest(C jdf_forward_READ_NULL)
target_ptg_sources(jdf_forward_READ_NULL PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/forward_READ_NULL.jdf")

add_test(unit_jdf_forward_RW_NULL_shm   ${SHM_TEST_CMD_LIST}       ./jdf_forward_RW_NULL)
add_test(unit_jdf_forward_READ_NULL_shm ${SHM_TEST_CMD_LIST}       ./jdf_forward_READ_NULL)
set_tests_properties(
  unit_jdf_forward_RW_NULL_shm
  unit_jdf_forward_READ_NULL_shm
  PROPERTIES
  PASS_REGULAR_EXPRESSION "A NULL is forwarded"
)

parsec_addtest(C write_check "vector.c")
target_ptg_sources(write_check PRIVATE "write_check.jdf")
add_test( write_check_shm ${SHM_TEST_CMD_LIST} ./write_check)

if( MPI_C_FOUND )
  add_test(unit_jdf_forward_RW_NULL_mpi   ${MPI_TEST_CMD_LIST} 2 ./jdf_forward_RW_NULL)
  add_test(unit_jdf_forward_READ_NULL_mpi ${MPI_TEST_CMD_LIST} 2 ./jdf_forward_READ_NULL)
  set_tests_properties(
    unit_jdf_forward_RW_NULL_mpi
    unit_jdf_forward_READ_NULL_mpi
    PROPERTIES
    PASS_REGULAR_EXPRESSION "A NULL is forwarded"
  )
endif( MPI_C_FOUND )

#
# Test to validate the number of input and output dependencies.
# Should fail is PaRSEC is compiled without support for more than 20
# output dependencies.
#
parsec_addtest(C must_fail_too_many_in_deps)
target_ptg_sources(must_fail_too_many_in_deps PRIVATE "too_many_in_deps.jdf")
# Avoid building these targets normally
set_target_properties(must_fail_too_many_in_deps PROPERTIES
                      EXCLUDE_FROM_ALL TRUE
                      EXCLUDE_FROM_DEFAULT_BUILD TRUE)
add_test( NAME must_fail_too_many_in_deps_shm
          COMMAND ${CMAKE_COMMAND} --build . --target must_fail_too_many_in_deps ${SHM_TEST_CMD_LIST} ./too_many_in_deps
          WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set_tests_properties(must_fail_too_many_in_deps_shm PROPERTIES WILL_FAIL TRUE)

parsec_addtest(C must_fail_too_many_out_deps)
target_ptg_sources(must_fail_too_many_out_deps PRIVATE "too_many_out_deps.jdf")
# Avoid building these targets normally
set_target_properties(must_fail_too_many_out_deps PROPERTIES
                      EXCLUDE_FROM_ALL TRUE
                      EXCLUDE_FROM_DEFAULT_BUILD TRUE)
add_test( NAME must_fail_too_many_out_deps_shm
          COMMAND ${CMAKE_COMMAND} --build . --target must_fail_too_many_out_deps ${SHM_TEST_CMD_LIST} ./too_many_out_deps
          WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set_tests_properties(must_fail_too_many_out_deps_shm PROPERTIES WILL_FAIL TRUE)

parsec_addtest(C must_fail_too_many_read_flows)
target_ptg_sources(must_fail_too_many_read_flows PRIVATE "too_many_read_flows.jdf")
# Avoid building these targets normally
set_target_properties(must_fail_too_many_read_flows PROPERTIES
                      EXCLUDE_FROM_ALL TRUE
                      EXCLUDE_FROM_DEFAULT_BUILD TRUE)
add_test( NAME must_fail_too_many_read_flows_shm
          COMMAND ${CMAKE_COMMAND} --build . --target must_fail_too_many_read_flows ${SHM_TEST_CMD_LIST} ./too_many_read_flows
          WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
        set_tests_properties(must_fail_too_many_read_flows_shm PROPERTIES WILL_FAIL TRUE)

parsec_addtest(C must_fail_too_many_write_flows)
target_ptg_sources(must_fail_too_many_write_flows PRIVATE "too_many_write_flows.jdf")
# Avoid building these targets normally
set_target_properties(must_fail_too_many_write_flows PROPERTIES
                      EXCLUDE_FROM_ALL TRUE
                      EXCLUDE_FROM_DEFAULT_BUILD TRUE)
add_test( NAME must_fail_too_many_write_flows_shm
          COMMAND ${CMAKE_COMMAND} --build . --target must_fail_too_many_write_flows ${SHM_TEST_CMD_LIST} ./too_many_write_flows
          WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set_tests_properties(must_fail_too_many_write_flows_shm PROPERTIES WILL_FAIL TRUE)

parsec_addtest(C must_fail_too_many_local_vars)
target_ptg_sources(must_fail_too_many_local_vars PRIVATE "too_many_local_vars.jdf")
# Avoid building these targets normally
set_target_properties(must_fail_too_many_local_vars PROPERTIES
                      EXCLUDE_FROM_ALL TRUE
                      EXCLUDE_FROM_DEFAULT_BUILD TRUE)
add_test( NAME must_fail_too_many_local_vars_shm
          COMMAND ${CMAKE_COMMAND} --build . --target must_fail_too_many_local_vars ${SHM_TEST_CMD_LIST} ./too_many_local_vars
          WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set_tests_properties(must_fail_too_many_local_vars_shm PROPERTIES WILL_FAIL TRUE)
