Add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/interfaces/ptg)

if( BUILD_PARSEC )
  LIST(APPEND EXTRA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/interfaces/interface.c)

  include(interfaces/dtd/CMakeLists.txt)

endif( BUILD_PARSEC )

set_property(TARGET parsec
             APPEND PROPERTY
                    PUBLIC_HEADER_H interfaces/interface.h)
