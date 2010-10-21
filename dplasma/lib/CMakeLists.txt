include(precision_generation)
include(JDFsupport)

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


if (DAGUE_MPI AND MPI_FOUND)
  add_library(dplasma-mpi ${generated_files} ${EXTRA_SOURCES})
  set_target_properties(dplasma-mpi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
  install(TARGETS dplasma-mpi ARCHIVE DESTINATION lib)
else (DAGUE_MPI AND MPI_FOUND)
  add_library(dplasma ${generated_files} ${EXTRA_SOURCES})
  set_target_properties(dplasma PROPERTIES COMPILE_FLAGS "-DADD_")
  install(TARGETS dplasma ARCHIVE DESTINATION lib)
endif (DAGUE_MPI AND MPI_FOUND)

