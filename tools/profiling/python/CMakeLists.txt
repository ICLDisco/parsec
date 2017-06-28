#
# Based on http://bloerg.net/2012/11/10/cmake-and-distutils.html
#
set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
# Create FISH environment
set(FISH_ENV_IN "${CMAKE_CURRENT_SOURCE_DIR}/utilities/fish.env.in")
set(FISH_ENV    "${CMAKE_CURRENT_BINARY_DIR}/utilities/fish.env")
# Create bash environment
set(BASH_ENV_IN "${CMAKE_CURRENT_SOURCE_DIR}/utilities/bash.env.in")
set(BASH_ENV    "${CMAKE_CURRENT_BINARY_DIR}/utilities/bash.env")
# Create csh and friends environment
set(CSH_ENV_IN  "${CMAKE_CURRENT_SOURCE_DIR}/utilities/csh.env.in")
set(CSH_ENV     "${CMAKE_CURRENT_BINARY_DIR}/utilities/csh.env")

set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/pbt2ptt.pyx"
                "${CMAKE_CURRENT_SOURCE_DIR}/pbt2ptt.pxd"
                "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in"
                "${CMAKE_CURRENT_SOURCE_DIR}/../dbpreader.c"
                "${CMAKE_CURRENT_SOURCE_DIR}/../dbpreader.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/../../../parsec/parsec_binary_profile.h"
                parsec-base)
set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/pbt2ptt.c")

#
# Always generate the setup.py so it can be used by hand
#
configure_file(${SETUP_PY_IN} ${SETUP_PY} @ONLY )
configure_file(${FISH_ENV_IN} ${FISH_ENV} @ONLY )
configure_file(${BASH_ENV_IN} ${BASH_ENV} @ONLY )
configure_file(${CSH_ENV_IN} ${CSH_ENV} @ONLY )

if( NOT PYTHON_EXECUTABLE )
    FIND_PACKAGE(PythonInterp QUIET)
endif( NOT PYTHON_EXECUTABLE )
if( PYTHONINTERP_FOUND )
    # Copy all Python files to the build directory before we 
    # potentially modify them with 2to3
    file(GLOB_RECURSE pyfiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
    foreach(file ${pyfiles})
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file} COPYONLY)
    endforeach()

    if(PYTHON_VERSION_MAJOR GREATER 2)
        get_filename_component(PYTHON_EXE_DIR ${PYTHON_EXECUTABLE} PATH)
        find_program(PYTHON_2TO3_EXECUTABLE
            NAMES 2to3
            HINTS ${PYTHON_EXE_DIR})
        if(NOT PYTHON_2TO3_EXECUTABLE)
            message(WARNING "2to3 python utility not found. The profiling scripts will not be converted for use with selected Python version ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
        else()
            execute_process(OUTPUT_QUIET
                COMMAND ${PYTHON_2TO3_EXECUTABLE} -w -n ${CMAKE_CURRENT_BINARY_DIR})
        endif()
    endif()

    add_custom_command(OUTPUT ${OUTPUT}
                       COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} build_ext --inplace --quiet
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                       DEPENDS ${DEPS})

    add_custom_target(target ALL DEPENDS ${OUTPUT})

    install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install)")
endif( PYTHONINTERP_FOUND )
