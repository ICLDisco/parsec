macro(parsec_addexample lang target input)
  list(APPEND EXTRA_LIBS m)

  if( MPI_C_FOUND )
    set(${target}_${lang}FLAGS  "${MPI_C_COMPILE_FLAGS} ${${target}_${lang}FLAGS}")
    set(${target}_LDFLAGS "${MPI_C_LINK_FLAGS} ${${target}_LDFLAGS}")
  endif( MPI_C_FOUND )
  set(${target}_LIBS    parsec parsec_distribution_matrix ${EXTRA_LIBS} ${${target}_LIBS})

  add_executable(${target} ${input})
  set_target_properties(${target} PROPERTIES
    LINKER_LANGUAGE ${lang}
    COMPILE_FLAGS "${${target}_${lang}FLAGS} ${LOCAL_${lang}_LINK_FLAGS}"
    LINK_FLAGS "${${target}_LDFLAGS}")
  target_link_libraries(${target} ${${target}_LIBS})
  #  install(TARGETS ${target} RUNTIME DESTINATION bin)

endmacro(parsec_addexample)

add_Subdirectory(interfaces/superscalar)
