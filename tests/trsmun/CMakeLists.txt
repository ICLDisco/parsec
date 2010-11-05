if( PLASMA_FOUND )

  include_directories(${PLASMA_INCLUDE_DIRS})
  
  if( MPI_FOUND )
    add_executable(pdtrsmun dtrsmun.c data_management.c trsmun_jdf.c)
    set_target_properties(pdtrsmun PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(pdtrsmun PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_")
    set_target_properties(pdtrsmun PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(pdtrsmun dplasma-mpi ${PLASMA_LDFLAGS} ${MPI_LIBRARIES} ${EXTRA_LIBS})
  endif( MPI_FOUND )

  add_executable(dtrsmun dtrsmun.c data_management.c trsmun_jdf.c)
  set_target_properties(dtrsmun PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(dtrsmun PROPERTIES COMPILE_FLAGS "-DADD_")
  set_target_properties(dtrsmun PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
  target_link_libraries(dtrsmun dplasma ${PLASMA_LDFLAGS} ${EXTRA_LIBS})

endif( PLASMA_FOUND )

# add the command to generate the source code
add_custom_command (
  OUTPUT trsmun_jdf.c
  COMMAND ../../tools/dague-compiler/daguepp trsmun_jdf.c
  MAIN_DEPENDENCY trsmun.jdf
  DEPENDS daguepp
  )
 
