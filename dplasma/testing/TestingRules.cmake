include(precision_generation)
include(JDFsupport)

macro(testings_addexec OUTPUTLIST PRECISIONS ZSOURCES)
  include_directories(common)

  # Set flags for compilation
  if( DAGUE_MPI AND MPI_FOUND )
    set(testings_addexec_CFLAGS  "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    set(testings_addexec_LDFLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    set(testings_addexec_LIBS   
      dplasma-mpi  dplasma_testscommon-mpi dague-mpi  dague_distribution_matrix-mpi 
      ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS}
    )
  else ( DAGUE_MPI AND MPI_FOUND )
    set(testings_addexec_CFLAGS  "-DADD")
    set(testings_addexec_LDFLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
    set(testings_addexec_LIBS   
      dplasma dplasma_testscommon dague dague_distribution_matrix 
      ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS}
    )
  endif()

  set(testing_addexec_GENFILES "")
  precisions_rules(testings_addexec_GENFILES "${PRECISIONS}" "${ZSOURCES}")
  
  foreach(testings_addexec_GENFILE ${testings_addexec_GENFILES})
    message(STATUS "${testings_addexec_GENFILE}")
    string(REGEX REPLACE "\\.[scdz]" "" testings_addexec_EXEC ${testings_addexec_GENFILE})
    string(REGEX REPLACE "generated/" "" testings_addexec_EXEC ${testings_addexec_EXEC})

    add_executable(${testings_addexec_EXEC} ${testings_addexec_GENFILE})
    set_target_properties(${testings_addexec_EXEC} PROPERTIES
                            LINKER_LANGUAGE Fortran
                            COMPILE_FLAGS "${testings_addexec_CFLAGS}"
                            LINK_FLAGS "${testings_addexec_LDFLAGS}")
    target_link_libraries(${testings_addexec_EXEC} ${testings_addexec_LIBS})
    list(APPEND ${OUTPUTLIST} ${testings_addexec_EXEC})
  endforeach()

endmacro(testings_addexec)

