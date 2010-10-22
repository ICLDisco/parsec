include(precision_generation)
include(JDFsupport)

set(EXTRA_SOURCES 
  testscommon.c
)

set(SOURCES
  # Level 3 Blas
  testing_ztrmm.c
#  zgemm_wrapper.c 
#  ztrmm_wrapper.c 
  # Lapack
#  zpotrf_wrapper.c 
#  zgetrf_wrapper.c 
#  zgeqrf_wrapper.c
)
precisions_rules(generated_files "+;d" "${SOURCES}") 

if( PLASMA_FOUND )

  # if( DAGUE_MPI AND MPI_FOUND )

  #   add_executable(mpi_trmm 
  #                  ${generated_files} 
  #                  ${EXTRA_SOURCES})
  #   set_target_properties(mpi_trsm PROPERTIES LINKER_LANGUAGE Fortran)
  #   set_target_properties(mpi_trsm PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
  #   set_target_properties(mpi_trsm PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
  #   target_link_libraries(mpi_trsm dplasma-mpi dague-mpi ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS})

  # endif( DAGUE_MPI AND MPI_FOUND )

    add_executable(testing_dtrmm 
      ${generated_files}
      ${EXTRA_SOURCES}
      )
    set_target_properties(testing_dtrmm PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(testing_dtrmm PROPERTIES COMPILE_FLAGS "-DADD_")
    set_target_properties(testing_dtrmm PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(testing_dtrmm dplasma-mpi dague-mpi dague_distribution_matrix-mpi ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS})
    #target_link_libraries(testing_dtrmm dplasma dague dague_distribution_matrix ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})
 
endif( PLASMA_FOUND )

