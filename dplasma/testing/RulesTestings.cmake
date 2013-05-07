include(RulesPrecisions)

macro(testings_addexec OUTPUTLIST PRECISIONS ZSOURCES)
  include_directories(. ${PLASMA_INCLUDE_DIRS})

  #set(testings_addexec_CFLAGS "-DADD_")
  #foreach(arg ${PLASMA_CFLAGS})
  #  set(testings_addexec_CFLAGS "${testings_addexec_CFLAGS} ${arg}")
  #endforeach(arg ${PLASMA_CFLAGS})

  # Set flags for compilation
  if( MPI_FOUND )
    set(testings_addexec_CFLAGS  "${MPI_COMPILE_FLAGS} ${testings_addexec_CFLAGS}")
    set(testings_addexec_LDFLAGS "${MPI_LINK_FLAGS} ${testings_addexec_LDFLAGS}")
    set(testings_addexec_LIBS
      common-mpi dplasma-mpi dplasma_cores dague-mpi dague_distribution_matrix-mpi
      ${MPI_LIBRARIES} ${EXTRA_LIBS}
      )
  else ( MPI_FOUND )
    set(testings_addexec_LIBS
      common dplasma dplasma_cores dague dague_distribution_matrix
      ${EXTRA_LIBS}
      )
  endif()

  set(testings_addexec_GENFILES "")
  precisions_rules_py(testings_addexec_GENFILES
    "${ZSOURCES}"
    PRECISIONS "${PRECISIONS}")
  foreach(testings_addexec_GENFILE ${testings_addexec_GENFILES})
    string(REGEX REPLACE "\\.c" "" testings_addexec_EXEC ${testings_addexec_GENFILE})

    add_executable(${testings_addexec_EXEC} ${testings_addexec_GENFILE})
    set_target_properties(${testings_addexec_EXEC} PROPERTIES
                            LINKER_LANGUAGE Fortran
                            COMPILE_FLAGS "${testings_addexec_CFLAGS}"
                            LINK_FLAGS "${testings_addexec_LDFLAGS} ${LOCAL_FORTRAN_LINK_FLAGS} ${CMAKE_EXE_EXPORTS_C_FLAG} ${PLASMA_LDFLAGS}")
    target_link_libraries(${testings_addexec_EXEC} ${testings_addexec_LIBS} ${PLASMA_LIBRARIES})
    install(TARGETS ${testings_addexec_EXEC} RUNTIME DESTINATION bin)
    list(APPEND ${OUTPUTLIST} ${testings_addexec_EXEC})
  endforeach()

endmacro(testings_addexec)

