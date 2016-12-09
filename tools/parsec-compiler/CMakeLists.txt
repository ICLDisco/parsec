# when crosscompiling the executable is imported from the
# export file.
IF(NOT CMAKE_CROSSCOMPILING)

  BISON_TARGET(parsec_yacc parsec.y ${CMAKE_CURRENT_BINARY_DIR}/parsec.y.c)
  FLEX_TARGET(parsec_flex parsec.l  ${CMAKE_CURRENT_BINARY_DIR}/parsec.l.c)
  ADD_FLEX_BISON_DEPENDENCY(parsec_flex parsec_yacc)

  # Bison and Flex are supposed to generate good code.
  # But they don't.
  # This approach is damageable, because we can't catch C errors in our .l or .y code
  # But if we don't do that, we'll keep having reports of compilation warnings forever.
  SET_SOURCE_FILES_PROPERTIES(${BISON_parsec_yacc_OUTPUTS} PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS} -w")
  SET_SOURCE_FILES_PROPERTIES(${FLEX_parsec_flex_OUTPUTS} PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS} -w")

  include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
  include_directories(AFTER ${CMAKE_CURRENT_BINARY_DIR})

  add_executable(parsecpp jdf.c jdf2c.c jdf_unparse.c ${BISON_parsec_yacc_OUTPUTS} ${FLEX_parsec_flex_OUTPUTS})
  set_target_properties(parsecpp PROPERTIES LINKER_LANGUAGE C)
  set_target_properties(parsecpp PROPERTIES LINK_FLAGS "${LOCAL_C_LINK_FLAGS}")
  target_link_libraries(parsecpp -lm)
  target_link_libraries(parsecpp parsec-base)

  install(TARGETS parsecpp RUNTIME DESTINATION bin)

  #
  # Generate the EXPORT file for external projects.
  #
  EXPORT(TARGETS parsecpp FILE "${CMAKE_BINARY_DIR}/ImportExecutables.cmake" NAMESPACE native-)
ENDIF(NOT CMAKE_CROSSCOMPILING)

