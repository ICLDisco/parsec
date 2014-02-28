#
#Use Doxygen to create the API documentation
#

option(BUILD_DOCUMENTATION "Generate API documentation during the build process." OFF)

find_package(Doxygen)

if (DOXYGEN_FOUND)

  #  set(PARSEC_DOX_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src")
  #  set(PARSEC_DOX_SRCS "${PARSEC_DOX_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/include")
  #  set(PARSEC_DOX_SRCS "${PARSEC_DOX_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/tools")
  #  set(PARSEC_DOX_SRCS "${PARSEC_DOX_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/data_dist")
  foreach( _file ${PARSEC_ALL_SRCS} )
    set(PARSEC_DOX_SRCS "${PARSEC_DOX_SRCS} ${_file}" )
  endforeach()

  set(PARSEC_DOX_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/doxygen/groups.dox ${PARSEC_DOX_SRCS}" )

  #-- Configure the Template Doxyfile for our specific project
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxygen/Doxyfile.in
                 ${CMAKE_CURRENT_BINARY_DIR}/doxygen/Doxyfile @ONLY IMMEDIATE )

  #-- Add a custom target to run Doxygen when ever the project is built
  if(BUILD_DOCUMENTATION)
    add_custom_target (docs ALL
      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doxygen/Doxyfile
      SOURCES ${CMAKE_CURRENT_BINARY_DIR}/doxygen/Doxyfile
      COMMENT "Generating API documentation with Doxygen" VERBATIM)
  else(BUILD_DOCUMENTATION)
    add_custom_target (docs
      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doxygen/Doxyfile
      SOURCES ${CMAKE_CURRENT_BINARY_DIR}/doxygen/Doxyfile
      COMMENT "Generating API documentation with Doxygen" VERBATIM)
  endif(BUILD_DOCUMENTATION)

endif (DOXYGEN_FOUND)

