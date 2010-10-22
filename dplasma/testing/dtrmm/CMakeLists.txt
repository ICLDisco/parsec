include(precision_generation)
include(JDFsupport)

set(SOURCES
  testing_ztrmm.c
)
precisions_rules(generated_files "+;d" "${SOURCES}") 

add_executable(testing_dtrmm 
  ${generated_files}
  )
set_target_properties(testing_dtrmm PROPERTIES LINKER_LANGUAGE Fortran)

if( PLASMA_FOUND )

  if( DAGUE_MPI AND MPI_FOUND )
    
    set_target_properties(testing_dtrmm PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(testing_dtrmm PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    set_target_properties(testing_dtrmm PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(testing_dtrmm 
      dplasma-mpi 
      dplasmatest-mpi 
      dague-mpi 
      dague_distribution_matrix-mpi 
      ${PLASMA_LIBRARIES} 
      ${BLAS_LIBRARIES} 
      ${MPI_LIBRARIES} 
      ${EXTRA_LIBS}
      )

  else ( DAGUE_MPI AND MPI_FOUND )

    set_target_properties(testing_dtrmm PROPERTIES COMPILE_FLAGS "-DADD_")
    set_target_properties(testing_dtrmm PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(testing_dtrmm 
      dplasma
      dplasmatest
      dague
      dague_distribution_matrix
      ${PLASMA_LIBRARIES} 
      ${BLAS_LIBRARIES} 
      ${MPI_LIBRARIES} 
      ${EXTRA_LIBS}
      )
    #target_link_libraries(testing_dtrmm dplasma dague dague_distribution_matrix ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})
 
    endif( DAGUE_MPI AND MPI_FOUND )
endif( PLASMA_FOUND )

