if( BISON_FOUND AND FLEX_FOUND )
  Add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/interfaces/ptg)
endif( BISON_FOUND AND FLEX_FOUND )

if( BUILD_PARSEC )
  LIST(APPEND EXTRA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/interface.c)

  include(interfaces/superscalar/CMakeLists.txt)

endif( BUILD_PARSEC )
