include(precision_generation)
include(JDFsupport)

set(EXTRA_SOURCES memory_pool.c dplasmaaux.c)

### Generate .c files from .jdf
set(OLDJDF 
  TSQR sgeqrt 
)
jdf_rules(generated_files "${OLDJDF}")

set(JDF
  gemm_NN gemm_TN gemm_NT gemm_TT
  trsm_LLN trsm_LLT trsm_LUN trsm_LUT trsm_RLN trsm_RLT trsm_RUN trsm_RUT
  trmm_LLN trmm_LLT trmm_LUT trmm_RLN trmm_RLT trmm_RUN trmm_RUT
  potrf_rl potrf_ll 
  getrf getrf_sd 
  geqrf
)
jdf_rules(precisionstub_files "${JDF}")
precisions_rules(generated_files "/;${DPLASMA_PRECISIONS}" "${precisionstub_files}")

set(NAMES
  ztrmm_LUN
)

# codegen.py need extension to detect if it's a file to parse or not
jdf_addextension( jdfSOURCES "${NAMES}" )
precisions_rules(generated_jdf "+;${DPLASMA_PRECISIONS}" "${jdfSOURCES}")
jdf_rules(precisionstub_files_new "${generated_jdf}")

set(SOURCES
  trsm_wrapper.c gemm_wrapper.c trmm_wrapper.c potrf_wrapper.c getrf_wrapper.c geqrf_wrapper.c
)
precisions_rules(generated_files "${DPLASMA_PRECISIONS}" "${SOURCES}") 

if (DAGUE_MPI AND MPI_FOUND)
  add_library(dplasma-mpi
    ${generated_files} 
    ${precisionstub_files_new}
    ${EXTRA_SOURCES}
    )
  set_target_properties(dplasma-mpi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
  install(TARGETS dplasma-mpi ARCHIVE DESTINATION lib)
else (DAGUE_MPI AND MPI_FOUND)
  add_library(dplasma 
    ${generated_files} 
    ${precisionstub_files_new}
    ${EXTRA_SOURCES}
    )
  set_target_properties(dplasma PROPERTIES COMPILE_FLAGS "-DADD_")
  install(TARGETS dplasma ARCHIVE DESTINATION lib)
endif (DAGUE_MPI AND MPI_FOUND)

