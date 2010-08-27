if( PLASMA_FOUND )

  include_directories( ${PLASMA_INCLUDE_DIR} ${PLASMA_INCLUDE_DIR}/../src)
  set(CUDA_BUILD_CUBIN ON)
  set(CUDA_HOST_COMPILATION_CPP OFF)
  set(SPOTRF_REQUIRED_LIBRARIES ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})
  set(GPU_SUPPORT "")

  if( CUDA_FOUND)
    cuda_add_library( mysgemm sgemm.cu )
    set(SPOTRF_REQUIRED_LIBRARIES ${SPOTRF_REQUIRED_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUDA_LIBRARY})
    set(GPU_SUPPORT gpu_gemm.c)
  endif (CUDA_FOUND)

  if( DAGUE_MPI AND MPI_FOUND )
    add_executable(mpi_sposv_ll cholesky_ll.c sposv.c ${GPU_SUPPORT})
    set_target_properties(mpi_sposv_ll PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(mpi_sposv_ll PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI -DLLT_LL")
    set_target_properties(mpi_sposv_ll PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(mpi_sposv_ll dague-mpi ${MPI_LIBRARIES} ${SPOTRF_REQUIRED_LIBRARIES})

    add_executable(mpi_sposv_rl cholesky_rl.c sposv.c ${GPU_SUPPORT})
    set_target_properties(mpi_sposv_rl PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(mpi_sposv_rl PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    set_target_properties(mpi_sposv_rl PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(mpi_sposv_rl dague-mpi ${MPI_LIBRARIES} ${SPOTRF_REQUIRED_LIBRARIES})
  endif( DAGUE_MPI AND MPI_FOUND )

  add_executable(sposv_ll cholesky_ll.c sposv.c ${GPU_SUPPORT})
  set_target_properties(sposv_ll PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(sposv_ll PROPERTIES COMPILE_FLAGS "-DADD_ -DLLT_LL")
  set_target_properties(sposv_ll PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
  target_link_libraries(sposv_ll dague ${SPOTRF_REQUIRED_LIBRARIES})

  add_executable(sposv_rl cholesky_rl.c sposv.c ${GPU_SUPPORT})
  set_target_properties(sposv_rl PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(sposv_rl PROPERTIES COMPILE_FLAGS "-DADD_")
  set_target_properties(sposv_rl PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
  target_link_libraries(sposv_rl dague ${SPOTRF_REQUIRED_LIBRARIES})

endif( PLASMA_FOUND )

# add the command to generate the source code
add_custom_command (
  OUTPUT cholesky_rl.c cholesky_rl.h
  COMMAND ../../tools/dague-compiler/dc -i cholesky_rl.jdf -o cholesky_rl
  MAIN_DEPENDENCY cholesky_rl.jdf
  DEPENDS dc
  )

add_custom_command (
  OUTPUT cholesky_ll.c cholesky_ll.h
  COMMAND ../../tools/dague-compiler/dc -i cholesky_ll.jdf -o cholesky_ll
  MAIN_DEPENDENCY cholesky_ll.jdf
  DEPENDS dc
  )
