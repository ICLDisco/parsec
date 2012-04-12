include(RulesPrecisions)
include(RulesJDF)
# reset variables
set(generated_files "")
set(generated_jdf "")

set(EXTRA_SOURCES
  memory_pool.c
  dplasmatypes.c
  dplasmaaux.c
  pivgen.c
  pivgen_dbg.c
  map2_wrapper.c
  butterfly_map.c
)

### Generate .c files from .jdf
set(OLDJDF
  ${CMAKE_CURRENT_SOURCE_DIR}/TSQR.jdf
)
jdf_rules(generated_files "${OLDJDF}")

### Generate .c files from .jdf for all required precisions
set(JDF
  zprint.jdf
  zgemm_NN.jdf zgemm_TN.jdf zgemm_NT.jdf zgemm_TT.jdf
  zhemm.jdf zsymm.jdf
  zherk_LN.jdf zherk_LT.jdf zherk_UN.jdf zherk_UT.jdf
  ztrsm_LLN.jdf ztrsm_LLT.jdf ztrsm_LUN.jdf ztrsm_LUT.jdf ztrsm_RLN.jdf ztrsm_RLT.jdf ztrsm_RUN.jdf ztrsm_RUT.jdf
  ztrmm_LLN.jdf ztrmm_LLT.jdf ztrmm_LUN.jdf ztrmm_LUT.jdf ztrmm_RLN.jdf ztrmm_RLT.jdf ztrmm_RUN.jdf ztrmm_RUT.jdf
  zlaswp.jdf
  zpotrf_rl.jdf zpotrf_ll.jdf
  zpotrf_Url.jdf zpotrf_Lrl.jdf
  zgebut.jdf
  zhebut.jdf
  zgetrf.jdf zhetrf.jdf
  ztrdsm.jdf
  zgetrf_incpiv.jdf zgetrf_incpiv_sd.jdf
  ztrsmpl.jdf ztrsmpl_sd.jdf
  zgeqrf.jdf
  zgelqf.jdf
  zherbt_L.jdf
  zhbrdt.jdf
  zlaset.jdf zplrnt.jdf zplghe.jdf zplgsy.jdf zlacpy.jdf
  zgeqrf_param.jdf zungqr_param.jdf
  zungqr.jdf
  zlange_inf_cyclic.jdf
  zlange_one_cyclic.jdf
  zlange_max_cyclic.jdf
  zlange_frb_cyclic.jdf
)
precisions_rules_py(generated_jdf
                    "${JDF}"
                    PRECISIONS "${DPLASMA_PRECISIONS}")

if( NOT ${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR} )
  foreach(src_file ${EXTRA_SOURCES})
    set_source_files_properties(${src_file} PROPERTIES COMPILE_FLAGS "-I.") 
  endforeach()
endif( NOT ${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR} )

list(APPEND generated_jdf
  "${CMAKE_CURRENT_SOURCE_DIR}/map2.jdf")

jdf_rules(generated_files "${generated_jdf}")

### Generate the dplasma wrappers for all required precisions
set(SOURCES
#  dplasma_zf77.c
  zprint_wrapper.c
  # Level 3 Blas
  zgemm_wrapper.c
  zhemm_wrapper.c
  zherk_wrapper.c
  ztrsm_wrapper.c
  ztrmm_wrapper.c
  ztrsmpl_wrapper.c
  zsymm_wrapper.c
  # Lapack Auxiliary
  zgeadd_wrapper.c
  zlacpy_wrapper.c
  zlaset_wrapper.c
  zlaswp_wrapper.c
  zplrnt_wrapper.c
  zplghe_wrapper.c
  zplgsy_wrapper.c
  zlange_wrapper.c
  zlange2_wrapper.c
  zlanhe_wrapper.c
  zlansy_wrapper.c
  # Lapack
  zpotrf_wrapper.c
  zhebut_wrapper.c
  zhetrs_wrapper.c
  zpotrs_wrapper.c
  zposv_wrapper.c
  zgetrf_wrapper.c
  zhetrf_wrapper.c
  ztrdsm_wrapper.c
  zgetrs_wrapper.c
  zgesv_wrapper.c
  zgetrf_incpiv_wrapper.c
  zgetrs_incpiv_wrapper.c
  zgesv_incpiv_wrapper.c
  zgeqrf_wrapper.c
  zgeqrf_param_wrapper.c
  zungqr_param_wrapper.c
  zgelqf_wrapper.c
  zherbt_wrapper.c
  zhbrdt_wrapper.c
  zungqr_wrapper.c
)
precisions_rules_py(generated_files
                 "${SOURCES}"
                 PRECISIONS "${DPLASMA_PRECISIONS}")

### Generate the lib
if (MPI_FOUND)
  add_library(dplasma-mpi
    ${generated_files}
    ${EXTRA_SOURCES})
  set_target_properties(dplasma-mpi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
  target_link_libraries(dplasma-mpi dplasma_cores)
  add_dependencies(dplasma-mpi dplasma_includes)
  install(TARGETS dplasma-mpi ARCHIVE DESTINATION lib)
else (MPI_FOUND)
  add_library(dplasma
    ${generated_files}
    ${EXTRA_SOURCES})
  target_link_libraries(dplasma dplasma_cores)
  add_dependencies(dplasma dplasma_includes)
  install(TARGETS dplasma ARCHIVE DESTINATION lib)
endif (MPI_FOUND)
