include(precision_generation)
include(JDFsupport)
# reset variables
set(generated_files "")
set(generated_jdf "")

set(EXTRA_SOURCES memory_pool.c dplasmaaux.c)

### Generate .c files from .jdf
set(OLDJDF 
  TSQR sgeqrt 
)
jdf_rules(generated_files "${OLDJDF}")

set(JDF
  zgemm_NN zgemm_TN zgemm_NT zgemm_TT
  ztrsm_LLN ztrsm_LLT ztrsm_LUN ztrsm_LUT ztrsm_RLN ztrsm_RLT ztrsm_RUN ztrsm_RUT
  ztrmm_LLN ztrmm_LLT ztrmm_LUN ztrmm_LUT ztrmm_RLN ztrmm_RLT ztrmm_RUN ztrmm_RUT
  zpotrf_rl zpotrf_ll 
  zgetrf zgetrf_sd 
  zgeqrf
)

# codegen.py need extension to detect if it's a file to parse or not
jdf_addextension( JSRC "${JDF}" )
precisions_rules(generated_jdf "+;${DPLASMA_PRECISIONS}" "${JSRC}")
jdf_rules(generated_files "${generated_jdf}")

set(HEADERS
  dplasma_z.h
)
precisions_rules(generated_headers "+;z;c;d;s" "${HEADERS}") 

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

if (DAGUE_MPI AND MPI_FOUND)
  add_library(dplasma-mpi
    ${generated_files} 
    ${EXTRA_SOURCES})
  set_target_properties(dplasma-mpi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
  install(TARGETS dplasma-mpi ARCHIVE DESTINATION lib)
  install(FILES dplasma.h ${generated_headers} ARCHIVE DESTINATION include)
else (DAGUE_MPI AND MPI_FOUND)
  add_library(dplasma 
    ${generated_files}
    ${EXTRA_SOURCES})
  set_target_properties(dplasma PROPERTIES COMPILE_FLAGS "-DADD_")
  install(TARGETS dplasma ARCHIVE DESTINATION lib)
  install(FILES dplasma.h ${generated_headers} ARCHIVE DESTINATION include)
endif (DAGUE_MPI AND MPI_FOUND)

