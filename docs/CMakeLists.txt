#
#Use Doxygen to create the API documentation
#

option(BUILD_DOCUMENTATION "Generate API documentation during the build process." OFF)

if (BUILD_DOCUMENTATION)

  find_package(Doxygen)

  if (DOXYGEN_FOUND)
    if(NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
      message(STATUS "PaRSEC is pulled as a submodule -- Not generating documentation")
      return()
    elseif(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
      message(FATAL_ERROR
        "Because PaRSEC generates many source files at compile time, "
        "no documentation will be generated when building directly within "
        "the source directory. To generate the documentation, re-configure "
        "and compile from an out-of-source directory (look for "
        "Out-of-source build trees with CMake for documentation)")
    else(NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
      get_property(SRCS GLOBAL PROPERTY PARSEC_DOX_SRCS)
      string (REPLACE ";" " " PARSEC_DOX_SRCS "${SRCS}")

      #-- Configure the Template Doxyfile for our specific project
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxygen/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/doxygen/Doxyfile @ONLY IMMEDIATE )

      #-- Add a custom command for the documentation output
      add_custom_command( OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/doxygen/doxygen.stamp
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/doxygen/Doxyfile
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doxygen/Doxyfile
        COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/doxygen/doxygen.stamp
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
      #-- Add a custom target to run Doxygen when the project is built
      add_custom_target(docs-parsec DEPENDS parsec ${CMAKE_CURRENT_BINARY_DIR}/doxygen/doxygen.stamp )
      message(STATUS "PaRSEC documentation can be generated with target 'docs-parsec' after the library is compiled")
    endif(NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  else (DOXYGEN_FOUND)
    message(FATAL_ERROR "Documentation generation requested but Doxygen application could not be found. Either install Doygen or disable the documentation generation (BUILD_DOCUMENTATION to OFF)")
  endif (DOXYGEN_FOUND)
else (BUILD_DOCUMENTATION)
  add_custom_target(docs-parsec
    COMMAND true
    COMMAND true
    COMMENT "No documentation will be generated. Configure with BUILD_DOCUMENTATION set to ON" VERBATIM)
endif (BUILD_DOCUMENTATION)

# if "global" docs target exists, add dependency on docs-parsec
if (TARGET docs)
  add_dependencies(docs docs-parsec)
else(TARGET docs)
  add_custom_target(docs DEPENDS docs-parsec)
endif(TARGET docs)
