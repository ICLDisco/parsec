# 
# DAGuE Internal: generation of various floating point precision files from a template.
#

set(PRECISIONPP ${CMAKE_SOURCE_DIR}/tools/precision_generator/codegen.py)
set(PRECISIONPP_subs ${CMAKE_SOURCE_DIR}/tools/precision_generator/subs.py)

include(ParseArguments)

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

  foreach(prec_rules_SOURCE ${SOURCES})
    foreach(prec_rules_PREC ${PREC_RULE_PRECISIONS})
    
      set(pythoncmd ${PRECISIONPP} -f ${CMAKE_CURRENT_SOURCE_DIR}/${prec_rules_SOURCE} -p ${prec_rules_PREC} ${PRECISIONPP_arg} ${PRECISIONPP_prefix})

      execute_process(COMMAND ${pythoncmd} --out
              OUTPUT_VARIABLE prec_rules_OSRC)
      string(STRIP "${prec_rules_OSRC}" prec_rules_OSRC)
      string(COMPARE NOTEQUAL "${prec_rules_OSRC}" "" got_file)
      # Force the copy of the original files in the binary_dir
      # for VPATH compilation
      if( NOT DAGUE_COMPILE_INPLACE )
        set(generate_out 1)
      else( NOT DAGUE_COMPILE_INPLACE )
        string(COMPARE NOTEQUAL "${prec_rules_OSRC}" "${prec_rules_SOURCE}" generate_out )
      endif()

      # We generate a dependency only if a file will be generated
      if( got_file )

        if( generate_out )
          # the custom command is executed in CMAKE_CURRENT_BINARY_DIR
          add_custom_command(
            OUTPUT ${prec_rules_OSRC}
            COMMAND rm -f ${prec_rules_OSRC} && ${pythoncmd} && chmod a-w ${prec_rules_OSRC}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${prec_rules_SOURCE} ${PRECISIONPP} ${PRECISIONPP_subs})
          set_source_files_properties(${prec_rules_OSRC} PROPERTIES COMPILE_FLAGS "-DPRECISION_${prec_rules_PREC}" GENERATED 1 IS_IN_BINARY_DIR 1 ) 
        else( generate_out )
          set_source_files_properties(${prec_rules_OSRC} PROPERTIES COMPILE_FLAGS "-DPRECISION_${prec_rules_PREC}" GENERATED 0 )
        endif( generate_out )

        list(APPEND ${OUTPUTLIST} ${prec_rules_OSRC})
      endif( got_file )
    endforeach()
  endforeach()
endmacro(precisions_rules_py)
