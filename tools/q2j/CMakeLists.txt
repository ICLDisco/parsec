### Look for OMEGA
option(DAGUE_Q2J
  "Build the JDF generator" ON)
set(FOUND_OMEGA false)
if(DAGUE_Q2J)
  message(STATUS "Looking for DAGUE_OMEGA_DIR to enable DAGUE_Q2J")
  if(DAGUE_OMEGA_DIR)
    message(STATUS "DAGUE_OMEGA_DIR found in ${DAGUE_OMEGA_DIR}")
    set(FOUND_OMEGA true)
  else(DAGUE_OMEGA_DIR)
    message(WARNING "DAGUE_OMEGA_DIR not set: disabling DAGUE_Q2J")
    set(DAGUE_Q2J false)
    set(FOUND_OMEGA false)
  endif(DAGUE_OMEGA_DIR)
endif(DAGUE_Q2J)

if(FOUND_OMEGA)
  BISON_TARGET(dague_q2j_yacc src/q2j.y ${CMAKE_CURRENT_BINARY_DIR}/q2j.y.c)
  FLEX_TARGET(dague_q2j_flex src/q2j.l  ${CMAKE_CURRENT_BINARY_DIR}/q2j.l.c)
  ADD_FLEX_BISON_DEPENDENCY(dague_q2j_flex dague_q2j_yacc)

  # Bison and Flex are supposed to generate good code.
  # But they don't.
  # This approach is damageable, because we can't catch C errors in our .l or .y code
  # But if we don't do that, we'll keep having reports of compilation warnings forever.
  SET_SOURCE_FILES_PROPERTIES(${BISON_dague_q2j_yacc_OUTPUTS} PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS} -w")
  SET_SOURCE_FILES_PROPERTIES(${FLEX_dague_q2j_flex_OUTPUTS} PROPERTIES COMPILE_FLAGS "${CMAKE_C_FLAGS} -w")

  include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../dague-compiler)

  add_executable(q2j 
      src/omega_interface.cpp
      src/jdfoutput.cpp
      src/driver.c
      src/utility.c
      src/symtab.c      
      ../dague-compiler/jdf_unparse.c
      ${BISON_dague_q2j_yacc_OUTPUTS}
      ${FLEX_dague_q2j_flex_OUTPUTS}
  )
  set_target_properties(q2j PROPERTIES COMPILE_FLAGS "-I${DAGUE_OMEGA_DIR}/omega_lib/include -I${DAGUE_OMEGA_DIR}/basic/include")
  set_target_properties(q2j PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(q2j PROPERTIES LINK_FLAGS "${LOCAL_C_LINK_FLAGS}")
  target_link_libraries(q2j -lm "${DAGUE_OMEGA_DIR}/omega_lib/obj/libomega.a")
  
  install(TARGETS q2j RUNTIME DESTINATION bin)
endif(FOUND_OMEGA)
