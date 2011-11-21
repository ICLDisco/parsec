include(RulesPrecisions)

macro(testings_addexec OUTPUTLIST PRECISIONS ZSOURCES)
  include_directories(.)

  set(testings_addexec_CFLAGS "-DADD_")
  foreach(arg ${PLASMA_CFLAGS})
    set(testings_addexec_CFLAGS "${testings_addexec_CFLAGS} ${arg}")
  endforeach(arg ${PLASMA_CFLAGS})

  set(testings_addexec_LDFLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
  set(testings_addexec_LIBS    "${EXTRA_LIBS}")
  # Set flags for compilation
  if( MPI_FOUND )
    set(testings_addexec_CFLAGS  "${MPI_COMPILE_FLAGS} ${testings_addexec_CFLAGS} -DUSE_MPI")
    set(testings_addexec_LDFLAGS "${MPI_LINK_FLAGS} ${testings_addexec_LDFLAGS}")
    set(testings_addexec_LIBS   
      common-mpi dplasma-mpi dplasma_cores dague-mpi dague_distribution_matrix-mpi 
      ${testings_addexec_LIBS} ${MPI_LIBRARIES} 
      )
  else ( MPI_FOUND )
    set(testings_addexec_LIBS   
      common dplasma dplasma_cores dague dague_distribution_matrix 
      ${testings_addexec_LIBS}
      )
  endif()

  set(testings_addexec_GENFILES "")
  precisions_rules(testings_addexec_GENFILES 
    "${ZSOURCES}"
    PRECISIONS "${PRECISIONS}")
  foreach(testings_addexec_GENFILE ${testings_addexec_GENFILES})
    string(REGEX REPLACE "\\.c" "" testings_addexec_EXEC ${testings_addexec_GENFILE})

    add_executable(${testings_addexec_EXEC} ${testings_addexec_GENFILE})
    set_target_properties(${testings_addexec_EXEC} PROPERTIES
                            LINKER_LANGUAGE Fortran
                            COMPILE_FLAGS "${testings_addexec_CFLAGS}"
                            LINK_FLAGS "${testings_addexec_LDFLAGS}")
    target_link_libraries(${testings_addexec_EXEC} ${testings_addexec_LIBS} ${PLASMA_LDFLAGS} ${PLASMA_LIBRARIES})
    install(TARGETS ${testings_addexec_EXEC} RUNTIME DESTINATION bin)
    list(APPEND ${OUTPUTLIST} ${testings_addexec_EXEC})
  endforeach()

endmacro(testings_addexec)

