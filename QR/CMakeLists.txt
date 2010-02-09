if( PLASMA_FOUND )
  include_directories( ${PLASMA_INCLUDE_PATH} )
  
  add_executable(dgels dgels.c QR.c)
  set_target_properties(dgels PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(dgels PROPERTIES COMPILE_FLAGS "-DDPLASMA_EXECUTE")

  add_executable(dgels.plasma dgels.c)
  set_target_properties(dgels.plasma PROPERTIES LINKER_LANGUAGE Fortran)

  target_link_libraries(dgels dplasma ${PLASMA_LIBRARY} ${BLAS_cblas_LIBRARY} ${LAPACK_LIBRARIES})
  target_link_libraries(dgels.plasma ${PLASMA_LIBRARY} ${BLAS_cblas_LIBRARY} ${LAPACK_LIBRARIES})
endif( PLASMA_FOUND )

# add the command to generate the source code
add_custom_command (
  OUTPUT QR.c
  COMMAND ../dpc QR.jdf QR.c
  DEPENDS dpc
  )
