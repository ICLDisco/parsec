macro(parsec_addexample lang target input)
  list(APPEND EXTRA_LIBS m)

  if( MPI_C_FOUND )
    set(${target}_${lang}FLAGS  "${MPI_C_COMPILE_FLAGS} ${${target}_${lang}FLAGS}")
    set(${target}_LDFLAGS "${MPI_C_LINK_FLAGS} ${${target}_LDFLAGS}")
  endif( MPI_C_FOUND )
  set(${target}_LIBS    parsec ${EXTRA_LIBS} ${${target}_LIBS})

  add_executable(${target} ${input})
  #  message(STATUS "${target} -> ${input}")
  set_target_properties(${target} PROPERTIES
    LINKER_LANGUAGE ${lang}
    COMPILE_FLAGS "${${target}_${lang}FLAGS} ${LOCAL_${lang}_LINK_FLAGS}"
    LINK_FLAGS "${${target}_LDFLAGS}")
  target_link_libraries(${target} ${${target}_LIBS})
  #  install(TARGETS ${target} RUNTIME DESTINATION bin)

endmacro(parsec_addexample)

parsec_addexample(C ex00 Ex00_StartStop.c)
parsec_compile_ptg(ex01_generated "${CMAKE_CURRENT_SOURCE_DIR}/Ex01_HelloWorld.jdf")
parsec_addexample(C ex01 "${ex01_generated}")
parsec_compile_ptg(ex02_generated "${CMAKE_CURRENT_SOURCE_DIR}/Ex02_Chain.jdf")
parsec_addexample(C ex02 "${ex02_generated}")
parsec_compile_ptg(ex03_generated "${CMAKE_CURRENT_SOURCE_DIR}/Ex03_ChainMPI.jdf")
parsec_addexample(C ex03 "${ex03_generated}")
parsec_compile_ptg(ex04_generated "${CMAKE_CURRENT_SOURCE_DIR}/Ex04_ChainData.jdf")
parsec_addexample(C ex04 "${ex04_generated}")
parsec_compile_ptg(ex05_generated "${CMAKE_CURRENT_SOURCE_DIR}/Ex05_Broadcast.jdf")
parsec_addexample(C ex05 "${ex05_generated}")
parsec_compile_ptg(ex06_generated "${CMAKE_CURRENT_SOURCE_DIR}/Ex06_RAW.jdf")
parsec_addexample(C ex06 "${ex06_generated}")
parsec_compile_ptg(ex07_generated "${CMAKE_CURRENT_SOURCE_DIR}/Ex07_RAW_CTL.jdf")
parsec_addexample(C ex07 "${ex07_generated}")

add_Subdirectory(interfaces/superscalar)
