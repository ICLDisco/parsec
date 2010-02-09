if( PLASMA_FOUND )
  include_directories( ${PLASMA_INCLUDE_PATH} )
  
  add_executable(dposv_rl dposv.c cholesky_rl.c)
  set_target_properties(dposv_rl PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(dposv_rl PROPERTIES COMPILE_FLAGS "-DDPLASMA_EXECUTE")

  add_executable(dposv_ll dposv.c cholesky_ll.c)
  set_target_properties(dposv_ll PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(dposv_ll PROPERTIES COMPILE_FLAGS "-DDPLASMA_EXECUTE")

  add_executable(dposv.plasma dposv.c)
  set_target_properties(dposv.plasma PROPERTIES LINKER_LANGUAGE Fortran)

  target_link_libraries(dposv_ll dplasma ${PLASMA_LIBRARY} ${BLAS_cblas_LIBRARY} ${LAPACK_LIBRARIES})
  target_link_libraries(dposv_rl dplasma ${PLASMA_LIBRARY} ${BLAS_cblas_LIBRARY} ${LAPACK_LIBRARIES})
  target_link_libraries(dposv.plasma ${PLASMA_LIBRARY} ${BLAS_cblas_LIBRARY} ${LAPACK_LIBRARIES})
endif( PLASMA_FOUND )

# add the command to generate the source code
add_custom_command (
  OUTPUT cholesky_rl.c
  COMMAND ../dpc cholesky_rl.jdf cholesky_rl.c
  DEPENDS dpc
  )
 
add_custom_command (
  OUTPUT cholesky_ll.c
  COMMAND ../dpc cholesky_ll.jdf cholesky_ll.c
  DEPENDS dpc
  )
 
