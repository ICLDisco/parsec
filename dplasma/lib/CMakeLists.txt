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
  systolic.c
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
  zherk_LN.jdf zherk_LC.jdf zherk_UN.jdf zherk_UC.jdf
  zsyrk_LN.jdf zsyrk_LT.jdf zsyrk_UN.jdf zsyrk_UT.jdf
  zher2k_LN.jdf zher2k_LC.jdf zher2k_UN.jdf zher2k_UC.jdf
  zsyr2k_LN.jdf zsyr2k_LT.jdf zsyr2k_UN.jdf zsyr2k_UT.jdf
  ztrsm_LLN.jdf ztrsm_LLT.jdf ztrsm_LUN.jdf ztrsm_LUT.jdf ztrsm_RLN.jdf ztrsm_RLT.jdf ztrsm_RUN.jdf ztrsm_RUT.jdf
  ztrmm_LLN.jdf ztrmm_LLT.jdf ztrmm_LUN.jdf ztrmm_LUT.jdf ztrmm_RLN.jdf ztrmm_RLT.jdf ztrmm_RUN.jdf ztrmm_RUT.jdf
  zger.jdf
  zlaswp.jdf
  zpotrf_Url.jdf zpotrf_Lrl.jdf
  zgebmm.jdf
  zgebut.jdf
  zhebut.jdf
  zgetrf.jdf zhetrf.jdf
  ztrmdm.jdf
  ztrdsm.jdf
  zgetrf_incpiv.jdf zgetrf_incpiv_sd.jdf
  ztrsmpl.jdf ztrsmpl_sd.jdf
  zgeqrf.jdf
  zgelqf.jdf
  zherbt_L.jdf
  zhbrdt.jdf
  zlaset.jdf
  zplrnt.jdf zplrnt_perso.jdf zplrnt_toeppd.jdf
  zplghe.jdf zplgsy.jdf zlacpy.jdf
  zgeqrf_param.jdf zungqr_param.jdf
  zgetrf_hincpiv.jdf ztrsmpl_hincpiv.jdf
#  zgetrf_hpp.jdf     ztrsmpl_hpp.jdf
#  zgetrf_hpp2.jdf    ztrsmpl_hpp2.jdf
  zgetrf_qrf.jdf     ztrsmpl_qrf.jdf
  zungqr.jdf
  zunmqr_LC.jdf zunmqr_LN.jdf zunmqr_RN.jdf zunmqr_RC.jdf
  zunmqr_param_LC.jdf
  zlange_inf_cyclic.jdf
  zlange_one_cyclic.jdf
  zlange_max_cyclic.jdf
  zlange_frb_cyclic.jdf
  zlansy.jdf
)
precisions_rules_py(generated_jdf
                    "${JDF}"
                    PRECISIONS "${DPLASMA_PRECISIONS}")

include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}")
if( NOT ${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR} )
  include_directories(BEFORE "${CMAKE_CURRENT_BINARY_DIR}")
#  foreach(src_file ${EXTRA_SOURCES})
#    set_source_files_properties(${src_file} PROPERTIES COMPILE_FLAGS "-I.")
#  endforeach()
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
  zsyrk_wrapper.c
  zher2k_wrapper.c
  zsyr2k_wrapper.c
  ztrsm_wrapper.c
  ztrmm_wrapper.c
  ztrsmpl_wrapper.c
  zsymm_wrapper.c
  # Level 2 Blas
  zger_wrapper.c
  # Lapack Auxiliary
  zgeadd_wrapper.c
  zlacpy_wrapper.c
  zlaset_wrapper.c
  zlaswp_wrapper.c
  zplrnt_wrapper.c
  zplrnt_perso_wrapper.c
  zplghe_wrapper.c
  zplgsy_wrapper.c
  zlange_wrapper.c
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
  zgetrf_hincpiv_wrapper.c ztrsmpl_hincpiv_wrapper.c
#  zgetrf_hpp_wrapper.c     ztrsmpl_hpp_wrapper.c
#  zgetrf_hpp2_wrapper.c    ztrsmpl_hpp2_wrapper.c
  zgetrf_qrf_wrapper.c     ztrsmpl_qrf_wrapper.c
  zungqr_param_wrapper.c
  zgeqrs_wrapper.c
  zgeqrs_param_wrapper.c
  zgelqf_wrapper.c
  zherbt_wrapper.c
  zhbrdt_wrapper.c
  zungqr_wrapper.c
  zungqr_param_wrapper.c
  zunmqr_wrapper.c
  zunmqr_param_wrapper.c
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
