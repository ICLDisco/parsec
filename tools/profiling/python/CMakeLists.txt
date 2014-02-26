#
# Based on http://bloerg.net/2012/11/10/cmake-and-distutils.html
#
if( DAGUE_PROF_TRACE )
set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(FISH_ENV_IN "${CMAKE_CURRENT_SOURCE_DIR}/utilities/fish.env.in")
set(BASH_ENV_IN "${CMAKE_CURRENT_SOURCE_DIR}/utilities/bash.env.in")
set(FISH_ENV    "${CMAKE_CURRENT_BINARY_DIR}/utilities/fish.env")
set(BASH_ENV    "${CMAKE_CURRENT_BINARY_DIR}/utilities/bash.env")
set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/pbt2ptt.pyx"
                "${CMAKE_CURRENT_SOURCE_DIR}/pbt2ptt.pxd"
                "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in"
                "${CMAKE_CURRENT_SOURCE_DIR}/pbt_info_parser.pxd"
                "${CMAKE_CURRENT_SOURCE_DIR}/pbt_info_parser.pxi"
                "${CMAKE_CURRENT_SOURCE_DIR}/../dbpreader.c"
                "${CMAKE_CURRENT_SOURCE_DIR}/../dbpreader.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/../../../include/dbp.h"
                dague-base dague)
set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/pbt2ptt.c")

#
# Always generate the setup.py so it can be used by hand
#
configure_file(${SETUP_PY_IN} ${SETUP_PY} @ONLY )
configure_file(${FISH_ENV_IN} ${FISH_ENV} @ONLY )
configure_file(${BASH_ENV_IN} ${BASH_ENV} @ONLY )

if( NOT PYTHON_EXECUTABLE )
    FIND_PACKAGE(PythonInterp QUIET)
endif( NOT PYTHON_EXECUTABLE )
if( PYTHONINTERP_FOUND )
    add_custom_command(OUTPUT ${OUTPUT}
                       COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} build
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                       DEPENDS ${DEPS})

    add_custom_target(target ALL DEPENDS ${OUTPUT})

    install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install)")
endif( PYTHONINTERP_FOUND )
endif( DAGUE_PROF_TRACE )
