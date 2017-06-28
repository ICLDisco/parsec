# Generate the Python3 precision generators if needed
# Note how this is done with an execute_process, during the configure
# rather than during the build: this is because GenDeps is invoked during
# configure, hence it needs to be created at this point
FIND_PACKAGE(PythonInterp REQUIRED)
if(PYTHON_VERSION_MAJOR GREATER 2)
  get_filename_component(PYTHON_EXE_DIR ${PYTHON_EXECUTABLE} PATH)
  find_program(PYTHON_2TO3_EXECUTABLE
    NAMES 2to3
    HINTS ${PYTHON_EXE_DIR})
  if(NOT PYTHON_2TO3_EXECUTABLE)
    message(FATAL_ERROR "2to3 python utility not found. Use Python 2.7 or provide the 2to3 utility")
  endif()

  execute_process(OUTPUT_QUIET
      COMMAND ${PYTHON_2TO3_EXECUTABLE} -w -W -n -o ${CMAKE_BINARY_DIR}/dplasma/tools/PrecisionGenerator ${CMAKE_SOURCE_DIR}/dplasma/tools/PrecisionGenerator)
endif()

