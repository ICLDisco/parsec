if( NOT BUILD_SHARED_LIBS )
  message(WARNING "Python support requires the building of shared libraries, but this PaRSEC installation only supports static builds. Python profiling will be disabled.")
  set(PARSEC_PYTHON_TOOLS OFF CACHE BOOL "True iff Python tools are enabled in PaRSEC")
  return()
endif( NOT BUILD_SHARED_LIBS )

FIND_PACKAGE(Python COMPONENTS Interpreter Development
             QUIET)

if( NOT Python_Development_FOUND )
  message(WARNING "Python Development not found. Disabling the profiling tools")
  set(PARSEC_PYTHON_TOOLS OFF CACHE BOOL "True iff Python tools are enabled in PaRSEC")
  return()
endif( NOT Python_Development_FOUND )

# Python support for profiling requires
# - Cython 0.21.2
# - pandas
# the latter can be installed AFTER building parsec tools, but since PaRSEC Python module is not usable without it,
# do not build it without pandas
find_package(Cython 0.21.2)
if( NOT CYTHON_EXECUTABLE )
  message(WARNING "Cython > 0.21.2 not found. Disabling the profiling tools")
  set(PARSEC_PYTHON_TOOLS OFF CACHE BOOL "True iff Python tools are enabled in PaRSEC")
  return()
endif( NOT CYTHON_EXECUTABLE )
execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import Cython; import pandas; import setuptools"
        RESULT_VARIABLE PARSEC_PYTHON_CAN_LOAD_PREREQUISITE_MODULES_IF_THIS_IS_ZERO
        ERROR_QUIET
)
if ( NOT (PARSEC_PYTHON_CAN_LOAD_PREREQUISITE_MODULES_IF_THIS_IS_ZERO EQUAL 0) )
  message(WARNING "Prerequisite Python modules (Cython, pandas, setuptools) not found. Disabling the profiling tools")
  set(PARSEC_PYTHON_TOOLS OFF CACHE BOOL "True iff Python tools are enabled in PaRSEC")
  return()
endif ( NOT (PARSEC_PYTHON_CAN_LOAD_PREREQUISITE_MODULES_IF_THIS_IS_ZERO EQUAL 0) )

if (NOT BUILD_TOOLS)
  message(STATUS "All prerequisite available but python tools have been disabled (by command line option)")
  return()
endif(NOT BUILD_TOOLS)

set(SRC_PYTHON_SUPPORT ${CMAKE_CURRENT_SOURCE_DIR}/common_utils.py ${CMAKE_CURRENT_SOURCE_DIR}/parsec_trace_tables.py ${CMAKE_CURRENT_SOURCE_DIR}/ptt_utils.py ${CMAKE_CURRENT_SOURCE_DIR}/profile2h5.py ${CMAKE_CURRENT_SOURCE_DIR}/pbt2ptt.pyx ${CMAKE_CURRENT_SOURCE_DIR}/pbt2ptt.pxd)
set(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/build/pbt2ptt.timestamp)

if(Python_VERSION_MAJOR EQUAL 2)
  # We do not support python 2 anymore. If you really need it you can go back
  # to an older version of parsec
  message(WARNING "The profiling scripts will not be installed for use with selected Python version ${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}. Update to python3.")
endif(Python_VERSION_MAJOR EQUAL 2)

# Always generate the setup.py first, to include it in the 2to3 conversion
# if necessary
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/setup.py @ONLY )
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml.in
               ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml @ONLY )

# Move the files into the build directory (to prevent the pollution of the source directory
# due to cython compilation) and compile the python module
# Also fake a local installation of the python module to be able to use it
# for testing purposes. With this we can simply set the PYTHONPATH to the correct
# location and import the pbt2ptt module.
add_custom_command(OUTPUT ${OUTPUT}
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC_PYTHON_SUPPORT} ${CMAKE_CURRENT_BINARY_DIR}
                   COMMAND ${Python_EXECUTABLE} -m pip install --quiet --upgrade --use-pep517 --no-build-isolation --no-deps --target=${CMAKE_CURRENT_BINARY_DIR}/python.test .
                   COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                   DEPENDS parsec-base ${SRC_PYTHON_SUPPORT}
                           ${CMAKE_CURRENT_BINARY_DIR}/setup.py
                           ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml)
add_custom_target(pbt2ptt ALL DEPENDS ${OUTPUT})

# Ask pip to install all python support in the right location
# (aka. according to the OS demands). Prepare to reconfigure the shell
# helper scripts to point to the right location
install(CODE "
set(_parsec_python_pip_args --quiet --upgrade --use-pep517 --no-build-isolation --no-deps --prefix=${CMAKE_INSTALL_PREFIX})
if(DEFINED ENV{DESTDIR} AND NOT \"\$ENV{DESTDIR}\" STREQUAL \"\")
  list(APPEND _parsec_python_pip_args --root=\$ENV{DESTDIR})
endif()
execute_process(COMMAND ${Python_EXECUTABLE} -m pip install \${_parsec_python_pip_args} .
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_ERROR_IS_FATAL ANY)")

# Create bash environment PaRSEC python support
configure_file(utilities/bash.env.in
               ${CMAKE_CURRENT_BINARY_DIR}/utilities/parsec.env.sh @ONLY )
# Create csh and friends environment PaRSEC python support
configure_file(utilities/csh.env.in
               ${CMAKE_CURRENT_BINARY_DIR}/utilities/parsec.env.csh @ONLY )
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/utilities/parsec.env.sh ${CMAKE_CURRENT_BINARY_DIR}/utilities/parsec.env.csh
         DESTINATION ${PARSEC_INSTALL_BINDIR}
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

file(GLOB pyfiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "examples/*.py" "profile2h5.py")
foreach(file ${pyfiles})
  install( FILES ${CMAKE_CURRENT_SOURCE_DIR}/${file}
           DESTINATION ${PARSEC_INSTALL_LIBEXECDIR}/parsec
           PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
  get_filename_component(filenoext "${file}" NAME_WE)
  get_filename_component(filenodir "${file}" NAME)
  install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink \$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${PARSEC_INSTALL_LIBEXECDIR}/parsec/${filenodir} \$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${PARSEC_INSTALL_BINDIR}/${filenoext})")
endforeach()

set(PARSEC_PYTHON_TOOLS ON CACHE BOOL "True if Python tools are enabled in PaRSEC")

