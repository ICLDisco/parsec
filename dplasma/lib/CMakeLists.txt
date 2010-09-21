set(SOURCES LU.jdf TSQR.jdf dgemm_TN.jdf trsm_LLN.jdf trsm_LUT.jdf trsm_RUN.jdf
  LU_sd.jdf dpotrf.jdf dgemm_NN.jdf dgemm_TT.jdf trsm_LLT.jdf trsm_RLN.jdf trsm_RUT.jdf
  QR.jdf dgemm_NT.jdf sgemm.jdf trsm_LUN.jdf trsm_RLT.jdf spotrf_ll.jdf spotrf_rl.jdf)
set(EXTRA_SOURCES memory_pool.c dtrsm.c pdgemm.c)

set(generated_files)
foreach(_file ${SOURCES})
  string(REPLACE ".jdf" "" file_tro ${_file})
  add_custom_command(
    OUTPUT ${file_tro}.c ${file_tro}.h
    COMMAND ../../tools/dague-compiler/dc -i ${_file} -o ${file_tro}
    MAIN_DEPENDENCY ${_file}
    DEPENDS dc
  ) 
  list(APPEND generated_files "${file_tro}.c")
endforeach()

add_library(dplasma ${generated_files} ${EXTRA_SOURCES})
set_target_properties(dplasma PROPERTIES COMPILE_FLAGS "-I${PLASMA_SRC_DIR} -DADD_")

add_library(dplasma-mpi ${generated_files} ${EXTRA_SOURCES})
set_target_properties(dplasma-mpi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -I${PLASMA_SRC_DIR} -DADD_ -DUSE_MPI")
