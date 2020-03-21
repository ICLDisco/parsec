if( NOT BUILD_SHARED_LIBS )
  message(WARNING "Python support requires the building of shared libraries, but this PaRSEC installation only supports static builds. Python profiling will be disabled.")
  return()
endif( NOT BUILD_SHARED_LIBS )

FIND_PACKAGE(Python COMPONENTS Interpreter Development QUIET)

if( NOT Python_Development_FOUND )
  return()
endif( NOT Python_Development_FOUND )

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
set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

# Create FISH environment PaRSEC python support
configure_file(utilities/fish.env.in
               ${CMAKE_CURRENT_BINARY_DIR}/utilities/fish.env @ONLY )
# Create bash environment PaRSEC python support
configure_file(utilities/bash.env.in
               ${CMAKE_CURRENT_BINARY_DIR}/utilities/bash.env @ONLY )
# Create csh and friends environment PaRSEC python support
configure_file(utilities/csh.env.in
               ${CMAKE_CURRENT_BINARY_DIR}/utilities/csh.env @ONLY )
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/utilities/fish.env ${CMAKE_CURRENT_BINARY_DIR}/utilities/bash.env ${CMAKE_CURRENT_BINARY_DIR}/utilities/csh.env
         DESTINATION ${PARSEC_INSTALL_BINDIR}
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

# Always generate the setup.py first, to include it in the 2to3 conversion
# if necessary
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${SETUP_PY}.in
               ${CMAKE_CURRENT_BINARY_DIR}/${SETUP_PY} @ONLY )

# Copy all Python files to the build directory before we
# potentially modify them with 2to3
file(GLOB_RECURSE pyfiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.py")
foreach(file ${pyfiles})
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${file}
                 ${CMAKE_CURRENT_BINARY_DIR}/${file} COPYONLY)
endforeach()

if(PYTHON_VERSION_MAJOR GREATER 2)
  # If we are using python 3.x we need to convert our scripts from
  # python 2 to python 3. Look for the 2to3 helper script
  get_filename_component(PYTHON_EXE_DIR ${Python_EXECUTABLE} PATH)
  find_program(PYTHON_2TO3_EXECUTABLE
               NAMES 2to3
               HINTS ${PYTHON_EXE_DIR})

  if(NOT PYTHON_2TO3_EXECUTABLE)
    message(WARNING "2to3 python utility not found. The profiling scripts will not be converted for use with selected Python version ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
  else()
    execute_process(OUTPUT_QUIET
                    COMMAND ${PYTHON_2TO3_EXECUTABLE} -w -n ${CMAKE_CURRENT_BINARY_DIR}/examples)
  endif(NOT PYTHON_2TO3_EXECUTABLE)
endif(PYTHON_VERSION_MAJOR GREATER 2)

add_custom_command(OUTPUT ${OUTPUT}
                   COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/${SETUP_PY} build_ext --quiet
                   COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                   DEPENDS ${DEPS}
                   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(pbt2ptt ALL DEPENDS ${OUTPUT})

install(CODE "execute_process(COMMAND ${Python_EXECUTABLE} \"${CMAKE_CURRENT_BINARY_DIR}/${SETUP_PY}\" install --skip-build --prefix=${CMAKE_INSTALL_PREFIX})")

# We need to install the rest of the python profiling support file into the correct
# python location (same as the generated python module). This should be
# ${CMAKE_INSTALL_PREFIX}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/common_utils.py ${CMAKE_CURRENT_BINARY_DIR}/parsec_trace_tables.py ${CMAKE_CURRENT_BINARY_DIR}/ptt_utils.py
         DESTINATION ${PARSEC_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages)

# Install the files from the binary directory after being translated
# by the 2to3 tool.
file(GLOB pyfiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "examples/*.py")
foreach(file ${pyfiles})
  install( FILES ${CMAKE_CURRENT_BINARY_DIR}/${file}
           DESTINATION ${PARSEC_INSTALL_LIBEXECDIR}/parsec
           PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
  get_filename_component(filenoext "${file}" NAME_WE)
  get_filename_component(filenodir "${file}" NAME)
  install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_INSTALL_PREFIX}/${PARSEC_INSTALL_LIBEXECDIR}/parsec/${filenodir} ${CMAKE_INSTALL_PREFIX}/${PARSEC_INSTALL_BINDIR}/${filenoext})")
endforeach()

