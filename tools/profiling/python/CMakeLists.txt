#
# Based on http://bloerg.net/2012/11/10/cmake-and-distutils.html
#

set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/parsec_binprof.pyx"
                "${CMAKE_CURRENT_SOURCE_DIR}/parsec_binprof.pxd"
		"${CMAKE_CURRENT_SOURCE_DIR}/../dbpreader.c"
		"${CMAKE_CURRENT_SOURCE_DIR}/../dbpreader.h"
		"${CMAKE_CURRENT_SOURCE_DIR}/../../../include/dbp.h"
                dague-base dague)
set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp"
                "${CMAKE_CURRENT_BINARY_DIR}/parsec_binprof.c")

#
# Always generate the setup.py so it can be used by hand
#
configure_file(${SETUP_PY_IN} ${SETUP_PY} @ONLY )

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

