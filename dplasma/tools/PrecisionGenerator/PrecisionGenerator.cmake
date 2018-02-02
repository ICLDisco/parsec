#
# PaRSEC Internal: generation of various floating point precision files from a template.
#
include(ParseArguments)
FIND_PACKAGE(PythonInterp REQUIRED)
if(PYTHON_VERSION_MAJOR GREATER 2)
  get_filename_component(PYTHON_EXE_DIR ${PYTHON_EXECUTABLE} PATH)
  find_program(PYTHON_2TO3_EXECUTABLE
    NAMES 2to3
    HINTS ${PYTHON_EXE_DIR})
  if(NOT PYTHON_2TO3_EXECUTABLE)
    message(FATAL_ERROR "2to3 python utility not found. Use Python 2.7 or provide the 2to3 utility")
  endif()

  set(GENDEPENDENCIES  ${CMAKE_BINARY_DIR}/dplasma/tools/PrecisionGenerator/PrecisionDeps.py)
  set(PRECISIONPP      ${CMAKE_BINARY_DIR}/dplasma/tools/PrecisionGenerator/PrecisionGenerator.py)
  set(PRECISIONPP_subs ${CMAKE_BINARY_DIR}/dplasma/tools/PrecisionGenerator/subs.py)
else()
  set(GENDEPENDENCIES  ${CMAKE_SOURCE_DIR}/dplasma/tools/PrecisionGenerator/PrecisionDeps.py)
  set(PRECISIONPP      ${CMAKE_SOURCE_DIR}/dplasma/tools/PrecisionGenerator/PrecisionGenerator.py)
  set(PRECISIONPP_subs ${CMAKE_SOURCE_DIR}/dplasma/tools/PrecisionGenerator/subs.py)
endif()

#
# Generates a rule for every SOURCES file, to create the precisions in PRECISIONS. If TARGETDIR
# is not empty then all generated files will be prepended with the $TARGETDIR/.
# A new file is created, from a copy by default
# If the first precision is "/", all occurences of the basename in the file are remplaced by
# "pbasename" where p is the selected precision.
# the target receives a -DPRECISION_p in its cflags.
#
macro(precisions_rules_py)
  PARSE_ARGUMENTS(PREC_RULE
    "TARGETDIR;PRECISIONS"
    ""
    ${ARGN})

  MESSAGE(STATUS "Generate precision dependencies in ${CMAKE_CURRENT_SOURCE_DIR}")

  # The first is the output variable list
  CAR(OUTPUTLIST ${PREC_RULE_DEFAULT_ARGS})
  # Everything else should be source files.
  CDR(SOURCES ${PREC_RULE_DEFAULT_ARGS})
  # By default the TARGETDIR is the current binary directory
  if( "${PREC_RULE_TARGETDIR}" STREQUAL "" )
    set(PREC_RULE_TARGETDIR "./")
    set(PRECISIONPP_prefix "./")
    set(PRECISIONPP_arg "-P")
  else( "${PREC_RULE_TARGETDIR}" STREQUAL "" )
    if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/${PREC_RULE_TARGETDIR})
    else(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/${PREC_RULE_TARGETDIR})
      file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${PREC_RULE_TARGETDIR})
    endif(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/${PREC_RULE_TARGETDIR})
    set(PRECISIONPP_arg "-P")
    set(PRECISIONPP_prefix "${PREC_RULE_TARGETDIR}")
  endif( "${PREC_RULE_TARGETDIR}" STREQUAL "" )

  set(options_list "")
  foreach(prec_rules_PREC ${PREC_RULE_PRECISIONS})
    set(options_list "${options_list} ${prec_rules_PREC}")
  endforeach()

  set(sources_list "")
  foreach(_src ${SOURCES})
    set(sources_list "${sources_list} ${_src}")
  endforeach()

  set(gencmd ${PYTHON_EXECUTABLE} ${GENDEPENDENCIES} -f "${sources_list}" -p "${options_list}" -s "${CMAKE_CURRENT_SOURCE_DIR}" ${PRECISIONPP_arg} ${PRECISIONPP_prefix})
  execute_process(COMMAND ${gencmd} OUTPUT_VARIABLE dependencies_list)

  foreach(_dependency ${dependencies_list})

    string(STRIP "${_dependency}" _dependency)
    string(COMPARE NOTEQUAL "${_dependency}" "" not_empty)
    if( not_empty )

      string(REGEX REPLACE "^(.*),(.*),(.*)$" "\\1" _dependency_INPUT "${_dependency}")
      set(_dependency_PREC   "${CMAKE_MATCH_2}")
      set(_dependency_OUTPUT "${CMAKE_MATCH_3}")

      set(pythoncmd ${PYTHON_EXECUTABLE} ${PRECISIONPP} -f ${CMAKE_CURRENT_SOURCE_DIR}/${_dependency_INPUT} -p ${_dependency_PREC} ${PRECISIONPP_arg} ${PRECISIONPP_prefix})

      string(STRIP "${_dependency_OUTPUT}" _dependency_OUTPUT)
      string(COMPARE NOTEQUAL "${_dependency_OUTPUT}" "" got_file)

      # Force the copy of the original files in the binary_dir
      # for VPATH compilation
      if( NOT PARSEC_COMPILE_INPLACE )
        set(generate_out 1)
      else( NOT PARSEC_COMPILE_INPLACE )
        string(COMPARE NOTEQUAL "${_dependency_OUTPUT}" "${_dependency_INPUT}" generate_out )
      endif()

      # We generate a dependency only if a file will be generated
      if( got_file )
        if( generate_out )
          # the custom command is executed in CMAKE_CURRENT_BINARY_DIR
          add_custom_command(
            OUTPUT ${_dependency_OUTPUT}
            COMMAND ${CMAKE_COMMAND} -E remove -f ${_dependency_OUTPUT} && ${pythoncmd} && chmod a-w ${_dependency_OUTPUT}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${_dependency_INPUT} ${PRECISIONPP} ${PRECISIONPP_subs})

          set_source_files_properties(${_dependency_OUTPUT} PROPERTIES COMPILE_FLAGS "-DPRECISION_${_dependency_PREC}" GENERATED 1 IS_IN_BINARY_DIR 1 )

        else( generate_out )
          set_source_files_properties(${_dependency_OUTPUT} PROPERTIES COMPILE_FLAGS "-DPRECISION_${_dependency_PREC}" GENERATED 0 )
        endif( generate_out )

        list(APPEND ${OUTPUTLIST} ${_dependency_OUTPUT})
      endif( got_file )
    endif()
  endforeach()

  MESSAGE(STATUS "Generate precision dependencies in ${CMAKE_CURRENT_SOURCE_DIR} - Done")

endmacro(precisions_rules_py)
