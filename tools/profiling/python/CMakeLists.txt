#
# Based on http://bloerg.net/2012/11/10/cmake-and-distutils.html
#
set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/pbt2ptt.pyx"
                "${CMAKE_CURRENT_SOURCE_DIR}/pbt2ptt.pxd"
                "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in"
                "${CMAKE_CURRENT_SOURCE_DIR}/../dbpreader.c"
                "${CMAKE_CURRENT_SOURCE_DIR}/../dbpreader.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/../../../parsec/parsec_binary_profile.h"
                parsec-base)
set(SETUP_PY    "setup.py")
set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/pbt2ptt.c")

#
# Always generate the setup.py so it can be used by hand
#
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${SETUP_PY}.in
               ${CMAKE_CURRENT_BINARY_DIR}/${SETUP_PY} @ONLY )
# Create FISH environment PaRSEC python support
configure_file(utilities/fish.env.in
               ${CMAKE_CURRENT_BINARY_DIR}/utilities/fish.env @ONLY )
# Create bash environment PaRSEC python support
configure_file(utilities/bash.env.in
               ${CMAKE_CURRENT_BINARY_DIR}/utilities/bash.env @ONLY )
# Create csh and friends environment PaRSEC python support
configure_file(utilities/csh.env.in
               ${CMAKE_CURRENT_BINARY_DIR}/utilities/csh.env @ONLY )

if( NOT PYTHON_EXECUTABLE )
  FIND_PACKAGE(PythonInterp QUIET)
endif( NOT PYTHON_EXECUTABLE )
if( PYTHONINTERP_FOUND )

  # Copy all Python files to the build directory before we
  # potentially modify them with 2to3
  file(GLOB_RECURSE pyfiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
  foreach(file ${pyfiles})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${file}
                   ${CMAKE_CURRENT_BINARY_DIR}/${file} COPYONLY)
  endforeach()

  if(PYTHON_VERSION_MAJOR GREATER 2)
    # If we are using python 3.x we need to convert our scripts from
    # python 2 to python 3. Look for the 2to3 helper script
    get_filename_component(PYTHON_EXE_DIR ${PYTHON_EXECUTABLE} PATH)
    find_program(PYTHON_2TO3_EXECUTABLE
                 NAMES 2to3
                 HINTS ${PYTHON_EXE_DIR})

    if(NOT PYTHON_2TO3_EXECUTABLE)
      message(WARNING "2to3 python utility not found. The profiling scripts will not be converted for use with selected Python version ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
    else()
      execute_process(OUTPUT_QUIET
                      COMMAND ${PYTHON_2TO3_EXECUTABLE} -w -n ${CMAKE_CURRENT_BINARY_DIR})
    endif(NOT PYTHON_2TO3_EXECUTABLE)
  endif(PYTHON_VERSION_MAJOR GREATER 2)

  add_custom_command(OUTPUT pbt2ptt.c
                     COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} build_ext --inplace --quiet
                     DEPENDS ${DEPS})

  add_custom_target(target ALL DEPENDS ${OUTPUT})

  install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install)")
  install( FILES ${CMAKE_CURRENT_BINARY_DIR}/pbt2ptt.so
           DESTINATION libexec/parsec )

  # Install the files from the binary directory after being translated
  # by the 2to3 tool.
  foreach(file ${pyfiles})
    if( NOT ${file} EQUAL "setup.py" )
      install( FILES ${CMAKE_CURRENT_BINARY_DIR}/${file}
               DESTINATION libexec/parsec )
    endif( NOT ${file} EQUAL "setup.py" )
  endforeach()
endif( PYTHONINTERP_FOUND )
