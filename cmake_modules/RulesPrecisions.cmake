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
macro(precisions_rules)
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
    set(PRECISIONPP_prefix "")
  else( "${PREC_RULE_TARGETDIR}" STREQUAL "" )
    if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/${PREC_RULE_TARGETDIR})
    else(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/${PREC_RULE_TARGETDIR})
      file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${PREC_RULE_TARGETDIR})
    endif(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/${PREC_RULE_TARGETDIR})
    set(PRECISIONPP_prefix "-P ${PREC_RULE_TARGETDIR}")
  endif( "${PREC_RULE_TARGETDIR}" STREQUAL "" )

  set(precisions_rules_SED 0)
  set(precisions_rules_PP 0)
  foreach(prec_rules_SOURCE ${SOURCES})
    string(REGEX REPLACE "^(.*/)*(.+)\\.(.+)$" "\\2;\\3" prec_rules_BSRCl ${prec_rules_SOURCE})
    set(prec_rules_BSRC "${CMAKE_MATCH_2}")
    set(prec_rules_ESRC "${CMAKE_MATCH_3}")
    foreach(prec_rules_PREC ${PREC_RULE_PRECISIONS})
      if("${prec_rules_PREC}" MATCHES "/")
        set(precisions_rules_SED 1)
      elseif("${prec_rules_PREC}" MATCHES "\\+")
        set(precisions_rules_PP 1)
      else()
        set(prec_rules_OSRC "${PREC_RULE_TARGETDIR}${prec_rules_PREC}${prec_rules_BSRC}.${prec_rules_ESRC}")
        
        if(${precisions_rules_SED})
          message(STATUS ${prec_rules_SOURCE})
          add_custom_command(
            OUTPUT ${prec_rules_OSRC}
            COMMAND sed 's/${prec_rules_BSRC}/${prec_rules_PREC}${prec_rules_BSRC}/g' ${CMAKE_CURRENT_SOURCE_DIR}/${prec_rules_SOURCE} >${PREC_RULE_TARGETDIR}${prec_rules_OSRC}
            MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/${prec_rules_SOURCE})
          
        elseif(${precisions_rules_PP})
          execute_process(COMMAND ${PRECISIONPP} --file ${CMAKE_CURRENT_SOURCE_DIR}/${prec_rules_SOURCE} --prec ${prec_rules_PREC} ${PRECISIONPP_prefix} --out
            OUTPUT_VARIABLE prec_rules_OSRC)
          string(STRIP "${prec_rules_OSRC}" prec_rules_OSRC)
          string(COMPARE NOTEQUAL "${prec_rules_OSRC}" "" got_file)
          # We generate a dependency only if a file will be generated
          if( ${got_file} )
	    #MESSAGE(STATUS "prec rule OSRC = ${prec_rules_OSRC}")
            add_custom_command(
              OUTPUT ${prec_rules_OSRC}
              COMMAND ${PRECISIONPP} -f ${CMAKE_CURRENT_SOURCE_DIR}/${prec_rules_SOURCE} -p ${prec_rules_PREC} ${PRECISIONPP_prefix}
              MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/${prec_rules_SOURCE}
              DEPENDS ${PRECISIONPP})
          endif()
        else()
          message(STATUS ${prec_rules_SOURCE})
          add_custom_command(
            OUTPUT ${prec_rules_OSRC}
            COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${PREC_RULE_TARGETDIR} && cp ${CMAKE_CURRENT_SOURCE_DIR}/${prec_rules_SOURCE} ${PREC_RULE_TARGETDIR}${prec_rules_OSRC}
            MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/${prec_rules_SOURCE})
        endif()
        set_source_files_properties(${prec_rules_OSRC} PROPERTIES COMPILE_FLAGS "-DPRECISION_${prec_rules_PREC}" GENERATED 1)
        list(APPEND ${OUTPUTLIST} ${prec_rules_OSRC})
      endif()
    endforeach()
  endforeach()
endmacro(precisions_rules)

