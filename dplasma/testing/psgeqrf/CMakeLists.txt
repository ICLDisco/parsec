if( PLASMA_FOUND )
  set(GPU_SUPPORT "")
  set(SGEQRF_REQUIRED_LIBRARIES ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES})
  if( CUDA_FOUND)
    set(CUDA_BUILD_CUBIN ON)
    set(CUDA_BUILD_EMULATION OFF)
    set(SGEQRF_REQUIRED_LIBRARIES ${SGEQRF_REQUIRED_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUDA_LIBRARY}
        ${CUDA_CUDART_LIBRARY})
    set(GPU_SUPPORT gpu_stsmqr.c)
  endif (CUDA_FOUND)

  precisions_rules(generated_files "+;${DPLASMA_PRECISIONS}" "testing_zgeqrf.c")

  if( DAGUE_MPI AND MPI_FOUND )
    add_executable(testing_sgeqrf_mpi generated/testing_sgeqrf.c ${GPU_SUPPORT})
    set_target_properties(testing_sgeqrf_mpi PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(testing_sgeqrf_mpi PROPERTIES COMPILE_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}/../common ${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI -DPRECISION_${prec_rules}")
    set_target_properties(testing_sgeqrf_mpi PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(testing_sgeqrf_mpi dplasma-mpi dplasma_testscommon-mpi dague-mpi dague_distribution_matrix-mpi
      ${MPI_LIBRARIES} ${EXTRA_LIBS} ${SGEQRF_REQUIRED_LIBRARIES})
  else( DAGUE_MPI AND MPI_FOUND )
    add_executable(testing_sgeqrf generated/testing_sgeqrf.c ${GPU_SUPPORT})
    set_target_properties(testing_sgeqrf PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(testing_sgeqrf PROPERTIES COMPILE_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}/../common -DADD_ -DLLT_LL -DPRECISION_${prec_rules}")
    set_target_properties(testing_sgeqrf PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(testing_sgeqrf dplasma dplasma_testscommon dague dague_distribution_matrix
      ${EXTRA_LIBS} ${SGEQRF_REQUIRED_LIBRARIES})
  endif( DAGUE_MPI AND MPI_FOUND )
endif( PLASMA_FOUND )

