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

# Python support for profiling (requires Cython 0.21.2)
find_package(Cython 0.21.2)
if( NOT CYTHON_EXECUTABLE )
  message(WARNING "Cython > 0.21.2 not found. Disabling the profiling tools")
  set(PARSEC_PYTHON_TOOLS OFF CACHE BOOL "True iff Python tools are enabled in PaRSEC")
  return()
endif( NOT CYTHON_EXECUTABLE )

set(SRC_PYTHON_SUPPORT ${CMAKE_CURRENT_SOURCE_DIR}/common_utils.py ${CMAKE_CURRENT_SOURCE_DIR}/parsec_trace_tables.py ${CMAKE_CURRENT_SOURCE_DIR}/ptt_utils.py ${CMAKE_CURRENT_SOURCE_DIR}/profile2h5.py ${CMAKE_CURRENT_SOURCE_DIR}/pbt2ptt.pyx ${CMAKE_CURRENT_SOURCE_DIR}/pbt2ptt.pxd)
set(SETUP_PY setup.py)

set(PYTHON_TOOLS_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(OUTPUT ${PYTHON_TOOLS_BIN_DIR}/build/pbt2ptt.timestamp)

if(Python_VERSION_MAJOR EQUAL 2)
  # We do not support python 2 anymore. If you really need it you can go back
  # to an older version of parsec
  message(WARNING "The profiling scripts will not be installed for use with selected Python version ${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}. Update to python3.")
endif(Python_VERSION_MAJOR EQUAL 2)

# Always generate the setup.py first, to include it in the 2to3 conversion
# if necessary
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${SETUP_PY}.in
               ${PYTHON_TOOLS_BIN_DIR}/${SETUP_PY} @ONLY )

# Move the files into the build directory (to prevent the pollution of the source directory
# due to cython compilation) and compile the python module
# Also fake a local installation of the python module to be able to use it
# for testing purposes. With this we can simply set the PYTHONPATH to the correct
# location and import the pbt2ptt module.
add_custom_command(OUTPUT ${OUTPUT}
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC_PYTHON_SUPPORT} ${CMAKE_CURRENT_BINARY_DIR}
                   COMMAND ${Python_EXECUTABLE} -m pip install --quiet --upgrade --target=${CMAKE_CURRENT_BINARY_DIR}/python.test .
                   COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                   DEPENDS parsec-base ${SRC_PYTHON_SUPPORT})
add_custom_target(pbt2ptt ALL DEPENDS ${OUTPUT})

# Call python distutils to install all python support in the right location
# (aka. according to the OS demands). Prepare to reconfigure the shell
# helper scripts to point to the right location
install(CODE "execute_process(COMMAND ${Python_EXECUTABLE} -m pip install --quiet --prefix=\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX} .
        WORKING_DIRECTORY ${PYTHON_TOOLS_BIN_DIR})")

# Create bash environment PaRSEC python support
configure_file(utilities/bash.env.in
               ${PYTHON_TOOLS_BIN_DIR}/utilities/parsec.env.sh @ONLY )
# Create csh and friends environment PaRSEC python support
configure_file(utilities/csh.env.in
               ${PYTHON_TOOLS_BIN_DIR}/utilities/parsec.env.csh @ONLY )
install( FILES ${PYTHON_TOOLS_BIN_DIR}/utilities/parsec.env.sh ${PYTHON_TOOLS_BIN_DIR}/utilities/parsec.env.csh
         DESTINATION ${PARSEC_INSTALL_BINDIR}
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

file(GLOB pyfiles RELATIVE ${PYTHON_TOOLS_BIN_DIR} "examples/*.py" "profile2h5.py")
foreach(file ${pyfiles})
  install( FILES ${PYTHON_TOOLS_BIN_DIR}/${file}
           DESTINATION ${PARSEC_INSTALL_LIBEXECDIR}/parsec
           PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
  get_filename_component(filenoext "${file}" NAME_WE)
  get_filename_component(filenodir "${file}" NAME)
  install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink \$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${PARSEC_INSTALL_LIBEXECDIR}/parsec/${filenodir} \$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${PARSEC_INSTALL_BINDIR}/${filenoext})")
endforeach()

set(PARSEC_PYTHON_TOOLS ON CACHE BOOL "True if Python tools are enabled in PaRSEC")


