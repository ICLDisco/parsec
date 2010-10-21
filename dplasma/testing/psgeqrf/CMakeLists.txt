if( PLASMA_FOUND )
  set(SGEQRF_REQUIRED_LIBRARIES ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})
  set(GPU_SUPPORT "")

  if( CUDA_FOUND)
    set(CUDA_BUILD_CUBIN ON)
    set(CUDA_BUILD_EMULATION OFF)
    set(SGEQRF_REQUIRED_LIBRARIES ${SGEQRF_REQUIRED_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUDA_LIBRARY}
        ${CUDA_CUDART_LIBRARY})
    set(GPU_SUPPORT gpu_stsmqr.c)
  endif (CUDA_FOUND)

  if( DAGUE_MPI AND MPI_FOUND )
    add_executable(psgeqrf sgeqrf.c ${GPU_SUPPORT})
    set_target_properties(psgeqrf PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(psgeqrf PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    set_target_properties(psgeqrf PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(psgeqrf dplasma-mpi dague-mpi dague_distribution_matrix-mpi
    ${MPI_LIBRARIES} ${EXTRA_LIBS} ${SGEQRF_REQUIRED_LIBRARIES})
  else( DAGUE_MPI AND MPI_FOUND )
    add_executable(sgeqrf sgeqrf.c ${GPU_SUPPORT})
    set_target_properties(sgeqrf PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(sgeqrf PROPERTIES COMPILE_FLAGS "-DADD_")
    set_target_properties(sgeqrf PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(sgeqrf dplasma dague dague_distribution_matrix ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS} ${SGEQRF_REQUIRED_LIBRARIES})
  endif( DAGUE_MPI AND MPI_FOUND )

endif( PLASMA_FOUND )

