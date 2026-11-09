#
# Python bindings (py_parsec), enabled with -DPARSEC_PYTHON_BINDINGS=ON.
#
# The package is built with pip against this build tree into
# ${CMAKE_CURRENT_BINARY_DIR}/python.test for the tests, and installed with
# pip under CMAKE_INSTALL_PREFIX (lib/pythonX.Y/site-packages) at install time.
#
if( NOT BUILD_SHARED_LIBS )
  message(WARNING "The Python bindings need a shared libparsec. Python bindings disabled.")
  return()
endif( NOT BUILD_SHARED_LIBS )

find_package(Python COMPONENTS Interpreter Development.Module)
if( NOT Python_FOUND )
  message(WARNING "Python interpreter and development files not found. Python bindings disabled.")
  return()
endif( NOT Python_FOUND )
execute_process(COMMAND ${Python_EXECUTABLE} -c "import Cython, numpy, pip"
                RESULT_VARIABLE PARSEC_PYTHON_BINDINGS_MISSING_MODULES
                OUTPUT_QUIET ERROR_QUIET)
if( NOT PARSEC_PYTHON_BINDINGS_MISSING_MODULES EQUAL 0 )
  message(WARNING "The Python bindings need the Cython, numpy and pip modules for ${Python_EXECUTABLE}. Python bindings disabled.")
  return()
endif()

# Cython writes its C files next to the sources: build from a copy.
set(PY_PARSEC_STAGE "${CMAKE_CURRENT_BINARY_DIR}/stage")
set(PY_PARSEC_TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/python.test")
file(GLOB PY_PARSEC_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/py_parsec/*.py"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/py_parsec/*.pyx"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/py_parsec/*.pxd"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/py_parsec/*.h")
# The PaRSEC headers, and those of the packages PaRSEC exposes in its own
# headers (MPI, hwloc, accelerator runtimes).
set(PY_PARSEC_INCLUDE_DIRS
    "${PROJECT_SOURCE_DIR}/parsec/include" "${PROJECT_SOURCE_DIR}"
    "${PROJECT_BINARY_DIR}/parsec/include" "${PROJECT_BINARY_DIR}"
    ${EXTRA_INCLUDES})
foreach(dep IN ITEMS MPI::MPI_C HWLOC::hwloc CUDA::cudart hip::host)
  if( TARGET ${dep} )
    list(APPEND PY_PARSEC_INCLUDE_DIRS "$<TARGET_PROPERTY:${dep},INTERFACE_INCLUDE_DIRECTORIES>")
  endif()
endforeach()
set(PY_PARSEC_INCLUDE_PATH "$<JOIN:$<REMOVE_DUPLICATES:${PY_PARSEC_INCLUDE_DIRS}>,:>")
set(PY_PARSEC_PIP_ARGS --quiet --upgrade --no-build-isolation --no-deps)

add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/py_parsec.timestamp"
                   COMMAND ${CMAKE_COMMAND} -E rm -rf "${PY_PARSEC_STAGE}"
                   COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/src" "${PY_PARSEC_STAGE}/src"
                   COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/setup.py"
                                                    "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml"
                                                    "${CMAKE_CURRENT_SOURCE_DIR}/README.md" "${PY_PARSEC_STAGE}"
                   COMMAND ${CMAKE_COMMAND} -E env "CC=${CMAKE_C_COMPILER}"
                           "PARSEC_INCLUDE_DIRS=${PY_PARSEC_INCLUDE_PATH}"
                           "PARSEC_LIBRARY_DIRS=$<TARGET_FILE_DIR:parsec>"
                           ${Python_EXECUTABLE} -m pip install ${PY_PARSEC_PIP_ARGS}
                           "--target=${PY_PARSEC_TEST_DIR}" "${PY_PARSEC_STAGE}"
                   COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_CURRENT_BINARY_DIR}/py_parsec.timestamp"
                   DEPENDS parsec ${PY_PARSEC_SOURCES}
                           "${CMAKE_CURRENT_SOURCE_DIR}/setup.py" "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml"
                   COMMENT "Building the PaRSEC Python bindings"
                   VERBATIM)
add_custom_target(py_parsec ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/py_parsec.timestamp")

# Rebuild for the install prefix: linked against the build tree (the
# installed library may be staged under DESTDIR), with a run path to the
# installed library directory.
install(CODE "
file(REMOVE_RECURSE \"${PY_PARSEC_STAGE}/build\")
set(_pip_args ${PY_PARSEC_PIP_ARGS} --prefix=${CMAKE_INSTALL_PREFIX})
if(DEFINED ENV{DESTDIR} AND NOT \"\$ENV{DESTDIR}\" STREQUAL \"\")
  list(APPEND _pip_args --root=\$ENV{DESTDIR})
endif()
execute_process(COMMAND ${CMAKE_COMMAND} -E env \"CC=${CMAKE_C_COMPILER}\"
                        \"PARSEC_INCLUDE_DIRS=${PY_PARSEC_INCLUDE_PATH}\"
                        \"PARSEC_LIBRARY_DIRS=$<TARGET_FILE_DIR:parsec>\"
                        \"PARSEC_RUNTIME_LIBRARY_DIRS=${CMAKE_INSTALL_PREFIX}/${PARSEC_INSTALL_LIBDIR}\"
                        ${Python_EXECUTABLE} -m pip install \${_pip_args} \"${PY_PARSEC_STAGE}\"
                COMMAND_ERROR_IS_FATAL ANY)")

if( BUILD_TESTING )
  execute_process(COMMAND ${Python_EXECUTABLE} -c "import pytest"
                  RESULT_VARIABLE PARSEC_PYTHON_NO_PYTEST OUTPUT_QUIET ERROR_QUIET)
  if( PARSEC_PYTHON_NO_PYTEST EQUAL 0 )
    set(PY_PARSEC_PYTEST ${Python_EXECUTABLE} -m pytest -q -p no:cacheprovider "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    parsec_addtest_cmd(python/bindings ${SHM_TEST_CMD_LIST} ${PY_PARSEC_PYTEST})
    set(PY_PARSEC_TESTS python/bindings)
    if( PARSEC_HAVE_MPI )
      parsec_addtest_cmd(python/bindings:mp ${MPI_TEST_CMD_LIST} 4 ${PY_PARSEC_PYTEST})
      list(APPEND PY_PARSEC_TESTS python/bindings:mp)
    endif( PARSEC_HAVE_MPI )
    set_property(TEST ${PY_PARSEC_TESTS} APPEND PROPERTY ENVIRONMENT
                 "PYTHONPATH=${PY_PARSEC_TEST_DIR}" "PYTHONDONTWRITEBYTECODE=1")
  else()
    message(STATUS "pytest not found for ${Python_EXECUTABLE}: the Python bindings will not be tested")
  endif()
endif( BUILD_TESTING )
