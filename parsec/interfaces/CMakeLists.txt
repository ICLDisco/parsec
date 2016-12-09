if( BUILD_PARSEC )
  LIST(APPEND EXTRA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/interface.c)

  INSTALL(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/interface.h
    DESTINATION include/parsec)

  include(interfaces/superscalar/CMakeLists.txt)
#  include(jdf/CMakeLists.txt)

endif( BUILD_PARSEC )
