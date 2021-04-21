if( BUILD_PARSEC )
  LIST(APPEND EXTRA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/superscalar/parsec_dtd_data_flush.c
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/superscalar/overlap_strategies.c
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/superscalar/insert_function.c)

  INSTALL(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/superscalar/insert_function.h
    DESTINATION ${PARSEC_INSTALL_INCLUDEDIR}/parsec/interfaces/superscalar/)

  if( PARSEC_WITH_DEVEL_HEADERS )
    install(FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/superscalar/insert_function_internal.h
            DESTINATION ${PARSEC_INSTALL_INCLUDEDIR}/parsec/interfaces/superscalar/)
  endif( PARSEC_WITH_DEVEL_HEADERS )

endif( BUILD_PARSEC )
