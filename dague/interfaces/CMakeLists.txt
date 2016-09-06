if( BUILD_DAGUE )
  LIST(APPEND EXTRA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/interface.c)

  INSTALL(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/interface.h
    DESTINATION include/dague)

  include(interfaces/superscalar/CMakeLists.txt)
#  include(jdf/CMakeLists.txt)

endif( BUILD_DAGUE )
