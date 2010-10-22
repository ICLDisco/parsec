include(precision_generation)
include(JDFsupport)
# reset variables
set(generated_files "")
set(generated_jdf "")

set(EXTRA_SOURCES memory_pool.c dplasmaaux.c)

### generate the dplasma headers for all possible precisions
set(HEADERS
  dplasma_z.h
)
precisions_rules(generated_headers "+;z;c;d;s" "${HEADERS}") 

### Generate .c files from .jdf
set(OLDJDF 
  TSQR.jdf sgeqrt.jdf 
)
jdf_rules(generated_files "${OLDJDF}")

### Generate .c files from .jdf for all required precisions
set(JDF
  zgemm_NN.jdf zgemm_TN.jdf zgemm_NT.jdf zgemm_TT.jdf
  ztrsm_LLN.jdf ztrsm_LLT.jdf ztrsm_LUN.jdf ztrsm_LUT.jdf ztrsm_RLN.jdf ztrsm_RLT.jdf ztrsm_RUN.jdf ztrsm_RUT.jdf
  ztrmm_LLN.jdf ztrmm_LLT.jdf ztrmm_LUN.jdf ztrmm_LUT.jdf ztrmm_RLN.jdf ztrmm_RLT.jdf ztrmm_RUN.jdf ztrmm_RUT.jdf
  zpotrf_rl.jdf zpotrf_ll.jdf 
  zgetrf.jdf zgetrf_sd.jdf 
  zgeqrf.jdf
)
precisions_rules(generated_jdf "+;${DPLASMA_PRECISIONS}" "${JDF}")
jdf_rules(generated_files "${generated_jdf}")

### Generate the dplasma wrappers for all required precisions 
set(SOURCES
  # Level 3 Blas
  ztrsm_wrapper.c 
  zgemm_wrapper.c 
  ztrmm_wrapper.c 
  # Lapack
  zpotrf_wrapper.c 
  zgetrf_wrapper.c 
  zgeqrf_wrapper.c
)
precisions_rules(generated_files "+;${DPLASMA_PRECISIONS}" "${SOURCES}") 

### Generate the lib 
if (DAGUE_MPI AND MPI_FOUND)
  add_library(dplasma-mpi
    ${generated_files} 
    ${EXTRA_SOURCES})
  set_target_properties(dplasma-mpi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
  install(TARGETS dplasma-mpi ARCHIVE DESTINATION lib)
  install(FILES dplasma.h ${generated_headers} DESTINATION include)
else (DAGUE_MPI AND MPI_FOUND)
  add_library(dplasma 
    ${generated_files}
    ${EXTRA_SOURCES})
  set_target_properties(dplasma PROPERTIES COMPILE_FLAGS "-DADD_")
  install(TARGETS dplasma ARCHIVE DESTINATION lib)
  install(FILES dplasma.h ${generated_headers} DESTINATION include)
endif (DAGUE_MPI AND MPI_FOUND)

