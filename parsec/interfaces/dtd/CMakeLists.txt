if( BUILD_PARSEC )
  LIST(APPEND EXTRA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/dtd/parsec_dtd_data_flush.c
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/dtd/overlap_strategies.c
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/dtd/insert_function.c)

  INSTALL(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/dtd/insert_function.h
    DESTINATION ${PARSEC_INSTALL_INCLUDEDIR}/parsec/interfaces/dtd/)

  if( PARSEC_WITH_DEVEL_HEADERS )
    set_property(TARGET parsec
                 APPEND PROPERTY
                        PUBLIC_HEADER_H interfaces/dtd/insert_function_internal.h)
  endif( PARSEC_WITH_DEVEL_HEADERS )

endif( BUILD_PARSEC )
