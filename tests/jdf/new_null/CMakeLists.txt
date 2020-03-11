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

