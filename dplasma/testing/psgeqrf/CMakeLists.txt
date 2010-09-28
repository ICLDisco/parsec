if( PLASMA_FOUND )
 include_directories( ../../lib ${PLASMA_INCLUDE_DIR} ${PLASMA_INCLUDE_DIR}/../src )
  set(SGEQRF_REQUIRED_LIBRARIES ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})
  set(GPU_SUPPORT "")

  if( CUDA_FOUND)
   set(CUDA_BUILD_CUBIN ON)
    set(CUDA_BUILD_EMULATION OFF)
    set(SGEQRF_REQUIRED_LIBRARIES ${SGEQRF_REQUIRED_LIBRARIES}
   ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUDA_LIBRARY}
   ${CUDA_CUDART_LIBRARY} "-L./libmagma -lmagma -lmagmablas")

   set(GPU_SUPPORT gpu_stsmqr.c)
  endif (CUDA_FOUND)

  if( DAGUE_MPI AND MPI_FOUND )
   add_executable(psgeqrf sgeqrf.c ${GPU_SUPPORT})
    set_target_properties(psgeqrf PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(psgeqrf PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -I${PLASMA_SRC_DIR} -DADD_ -DUSE_MPI")
    set_target_properties(psgeqrf PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(psgeqrf dplasma-mpi dague-mpi
    ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS} ${SGEQRF_REQUIRED_LIBRARIES})
  endif( DAGUE_MPI AND MPI_FOUND )

 add_executable(sgeqrf sgeqrf.c ${GPU_SUPPORT})
  set_target_properties(sgeqrf PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(sgeqrf PROPERTIES COMPILE_FLAGS "-I${PLASMA_SRC_DIR} -DADD_")
  set_target_properties(sgeqrf PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
  target_link_libraries(sgeqrf dplasma dague ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS} ${SGEQRF_REQUIRED_LIBRARIES})

endif( PLASMA_FOUND )

