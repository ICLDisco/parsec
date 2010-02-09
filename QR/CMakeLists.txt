if( PLASMA_FOUND )
  include_directories( ${PLASMA_INCLUDE_DIR} ${PLASMA_INCLUDE_DIR}/../src )
  
  add_executable(dgels dgels.c QR.c)
  set_target_properties(dgels PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(dgels PROPERTIES COMPILE_FLAGS "-DDPLASMA_EXECUTE")
  target_link_libraries(dgels dplasma ${PLASMA_LIBRARIES} ${BLAS_cblas_LIBRARY} ${LAPACK_LIBRARIES})

  add_executable(dgels.plasma dgels.c)
  set_target_properties(dgels.plasma PROPERTIES LINKER_LANGUAGE Fortran)
  target_link_libraries(dgels.plasma ${PLASMA_LIBRARIES} ${BLAS_cblas_LIBRARY} ${LAPACK_LIBRARIES})
endif( PLASMA_FOUND )

# add the command to generate the source code
add_custom_command (
  OUTPUT QR.c
  COMMAND ../dpc QR.jdf QR.c
  DEPENDS dpc
  )
