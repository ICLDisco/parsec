### Look for OMEGA
option(PARSEC_Q2J
  "Build the JDF generator" ON)

if(BUILD_TOOLS AND PARSEC_Q2J AND NOT CMAKE_CROSSCOMPILING)
  find_package(Omega)
  if(NOT OMEGA_FOUND)
    return()
  endif(NOT OMEGA_FOUND)

  BISON_TARGET(parsec_q2j_yacc src/q2j.y ${CMAKE_CURRENT_BINARY_DIR}/q2j.y.c)
  FLEX_TARGET(parsec_q2j_flex src/q2j.l  ${CMAKE_CURRENT_BINARY_DIR}/q2j.l.c)
  ADD_FLEX_BISON_DEPENDENCY(parsec_q2j_flex parsec_q2j_yacc)

  add_executable(q2j
      src/omega_interface.cpp
      src/jdfoutput.cpp
      src/jdfregister.cpp
      src/driver.c
      src/utility.c
      src/symtab.c
      ../ptg-compiler/jdf_unparse.c
      ${BISON_parsec_q2j_yacc_OUTPUTS}
      ${FLEX_parsec_q2j_flex_OUTPUTS}
  )
  target_include_directories(q2j BEFORE PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(q2j PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../ptg-compiler
    ${OMEGA_INCLUDE_DIRS})
  set_target_properties(q2j PROPERTIES LINKER_LANGUAGE CXX)
  target_link_libraries(q2j PRIVATE m parsec-base ${OMEGA_LIBRARIES})

  install(TARGETS q2j RUNTIME DESTINATION bin)
endif(BUILD_TOOLS AND PARSEC_Q2J AND NOT CMAKE_CROSSCOMPILING)
