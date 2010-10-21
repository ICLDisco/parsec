set(DPLASMA_PRECISIONS "s;d") # "s;d;c;z"

macro(precisions_rules OUTPUTLIST PRECISIONS SOURCES)
 set(precisions_rules_SED 0)
 foreach(prec_rules_SOURCE ${SOURCES})
  string(REGEX REPLACE "^(.*/)*(.+)\\.(.+)$" "\\2;\\3" prec_rules_BSRCl ${prec_rules_SOURCE})
  set(prec_rules_BSRC "${CMAKE_MATCH_2}")
  set(prec_rules_ESRC "${CMAKE_MATCH_3}")
  foreach(prec_rules_PREC ${PRECISIONS})
   if("${prec_rules_PREC}" MATCHES "/") 
    set(precisions_rules_SED 1)
   else()
    set(prec_rules_OSRC "generated/${prec_rules_PREC}${prec_rules_BSRC}.${prec_rules_ESRC}")
    if(precisions_rules_SED)
      add_custom_command(
        OUTPUT "${prec_rules_OSRC}"
        COMMAND sed '{ s/${prec_rules_BSRC}/${prec_rules_PREC}${prec_rules_BSRC}/g }' ${CMAKE_CURRENT_SOURCE_DIR}/${prec_rules_SOURCE} >${CMAKE_CURRENT_BINARY_DIR}/${prec_rules_OSRC}
        MAIN_DEPENDENCY ${prec_rules_SOURCE}
        DEPENDS ${DAGUEPP})
    else()
      add_custom_command(
        OUTPUT "${prec_rules_OSRC}"
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${prec_rules_SOURCE} ${CMAKE_CURRENT_BINARY_DIR}/${prec_rules_OSRC}
        MAIN_DEPENDENCY ${prec_rules_SOURCE}
        DEPENDS ${DAGUEPP})
    endif()
    set_source_files_properties(${prec_rules_OSRC} PROPERTIES COMPILE_FLAGS "-DPRECISION_${prec_rules_PREC}")
    list(APPEND ${OUTPUTLIST} ${prec_rules_OSRC})
   endif()
  endforeach()
 endforeach()
endmacro(precisions_rules)

macro(jdf_rules jdf_rules_OUTPUTLIST jdf_rules_SOURCES)
  foreach(jdf_rules_SOURCE ${jdf_rules_SOURCES})
    string(REGEX REPLACE "^(.*/)*(.+)\\.*.*" "\\2" jdf_rules_BSRC ${jdf_rules_SOURCE})
    set(jdf_rules_OSRC "generated/${jdf_rules_BSRC}")
    add_custom_command(
        OUTPUT ${jdf_rules_OSRC}.h ${jdf_rules_OSRC}.c
        COMMAND mkdir -p generated && ${DAGUEPP} -i ${CMAKE_CURRENT_SOURCE_DIR}/${jdf_rules_SOURCE}.jdf -o ${CMAKE_CURRENT_BINARY_DIR}/${jdf_rules_OSRC} -f ${jdf_rules_BSRC}
        MAIN_DEPENDENCY ${jdf_rules_SOURCE}.jdf
        DEPENDS ${DAGUEPP})
    #        add_custom_target(${jdf_rules_BSRC} DEPENDS ${jdf_rules_OSRC}.h ${jdf_rules_OSRC}.c)
    #     set_source_files_properties(${jdf_rules_OSRC}.h ${jdf_rules_OSRC}.c PROPERTIES GENERATED 1)
    list(APPEND ${jdf_rules_OUTPUTLIST} "${jdf_rules_OSRC}.h;${jdf_rules_OSRC}.c")
    message(STATUS "rule to generate ${jdf_rules_OSRC}.[ch] from ${jdf_rules_SOURCE}.jdf")
  endforeach()
endmacro(jdf_rules)


set(EXTRA_SOURCES memory_pool.c)

### Generate .c files from .jdf
set(OLDJDF 
  dpotrf spotrf_ll spotrf_rl
  LU LU_sd
  QR TSQR sgeqrt 
)
jdf_rules(generated_files "${OLDJDF}")

set(JDF
  gemm_NN gemm_TN gemm_NT gemm_TT
  trsm_LLN trsm_LLT trsm_LUN trsm_LUT trsm_RLN trsm_RLT trsm_RUN trsm_RUT
  trmm_LLN trmm_LLT trmm_LUN trmm_LUT trmm_RLN trmm_RLT trmm_RUN trmm_RUT
)
jdf_rules(precisionstub_files "${JDF}")
precisions_rules(generated_files "/;${DPLASMA_PRECISIONS}" "${precisionstub_files}")

set(SOURCES trsm.c gemm.c trmm.c)
precisions_rules(generated_files "${DPLASMA_PRECISIONS}" "${SOURCES}") 

message(STATUS "${generated_files}")

add_library(dplasma ${generated_files} ${EXTRA_SOURCES})
#add_dependencies(dplasma ${OLDJDF})
set_target_properties(dplasma PROPERTIES COMPILE_FLAGS "-DADD_")
install(TARGETS dplasma ARCHIVE DESTINATION lib)

if (DAGUE_MPI AND MPI_FOUND)
  add_library(dplasma-mpi ${generated_files} ${EXTRA_SOURCES})
  set_target_properties(dplasma-mpi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
  install(TARGETS dplasma-mpi ARCHIVE DESTINATION lib)
endif (DAGUE_MPI AND MPI_FOUND)
