option(BUILD_DOCUMENTATION "Use Doxygen to create the HTML based API documentation" OFF)

if(BUILD_DOCUMENTATION)
  find_package(Doxygen)

  if (NOT DOXYGEN_FOUND)
    message(FATAL_ERROR
      "Doxygen is needed to build the documentation. Please install it correctly")
  endif()

  set(PARSEC_DOX_SRCS )
  foreach( _file ${PARSEC_ALL_SRCS} )
    #message(STATUS "File: ${_file}" )
    set(PARSEC_DOX_SRCS "${PARSEC_DOX_SRCS} ${_file}" )
  endforeach()

  #-- Configure the Template Doxyfile for our specific project
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxygen/Doxyfile.in
                 ${CMAKE_CURRENT_BINARY_DIR}/doxygen/Doxyfile @ONLY IMMEDIATE )
  #-- Add a custom target to run Doxygen when ever the project is built
  add_custom_target (docs
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doxygen/Doxyfile
    SOURCES ${CMAKE_CURRENT_BINARY_DIR}/doxygen/Doxyfile )

endif()

