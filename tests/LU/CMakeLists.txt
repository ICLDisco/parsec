if( PLASMA_FOUND )
  include_directories( ${PLASMA_INCLUDE_DIR} ${PLASMA_INCLUDE_DIR}/../src )
  
  add_executable(dgesv dgesv.c LU.c)
  set_target_properties(dgesv PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(dgesv PROPERTIES COMPILE_FLAGS "-DDPLASMA_EXECUTE")
  target_link_libraries(dgesv dplasma ${PLASMA_LIBRARIES} ${LAPACK_LIBRARIES} ${EXTRA_LIBS})

  add_executable(dgesv.plasma dgesv.c)
  set_target_properties(dgesv.plasma PROPERTIES LINKER_LANGUAGE Fortran)
  target_link_libraries(dgesv.plasma ${PLASMA_LIBRARIES} ${LAPACK_LIBRARIES} ${EXTRA_LIBS})
endif( PLASMA_FOUND )

# add the command to generate the source code
add_custom_command (
  OUTPUT LU.c
  COMMAND ../../tools/dpc LU.jdf LU.c
  MAIN_DEPENDENCY LU.jdf
  DEPENDS dpc
  )
