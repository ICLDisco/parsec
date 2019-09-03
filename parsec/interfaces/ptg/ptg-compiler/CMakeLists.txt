# when crosscompiling the executable is imported from the
# export file.
IF(NOT CMAKE_CROSSCOMPILING)

  BISON_TARGET(parsec_yacc parsec.y ${CMAKE_CURRENT_BINARY_DIR}/parsec.y.c)
  FLEX_TARGET(parsec_flex parsec.l  ${CMAKE_CURRENT_BINARY_DIR}/parsec.l.c)
  ADD_FLEX_BISON_DEPENDENCY(parsec_flex parsec_yacc)
  # Bison and Flex are supposed to generate good code.  But they don't.
  # This approach is damageable, because we can't catch C errors in our .l or .y code
  # But if we don't do that, we'll keep having reports of compilation warnings forever.
  set_property(SOURCE ${BISON_parsec_yacc_OUTPUTS} APPEND PROPERTY COMPILE_OPTIONS "-w")
  set_property(SOURCE ${FLEX_parsec_flex_OUTPUTS} APPEND PROPERTY COMPILE_OPTIONS "-w")

  if(NOT TARGET parsec_ptgpp)
    add_executable(parsec_ptgpp jdf.c jdf2c.c jdf_unparse.c ${BISON_parsec_yacc_OUTPUTS} ${FLEX_parsec_flex_OUTPUTS})
  endif(NOT TARGET parsec_ptgpp)
  target_include_directories(parsec_ptgpp PRIVATE BEFORE
      ${CMAKE_CURRENT_SOURCE_DIR})
  target_include_directories(parsec_ptgpp PRIVATE
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(parsec_ptgpp PRIVATE m parsec-base)

  install(TARGETS parsec_ptgpp
    EXPORT parsec-targets
    RUNTIME DESTINATION bin)

  #
  # Generate the EXPORT file for external projects.
  #
  EXPORT(TARGETS parsec_ptgpp FILE "${CMAKE_BINARY_DIR}/ImportExecutables.cmake")
ENDIF(NOT CMAKE_CROSSCOMPILING)

