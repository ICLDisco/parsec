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

  if( DPLASMA_MPI AND MPI_FOUND )
    add_library(mpi_sposv STATIC data_management.c ${GPU_SUPPORT} sposv.c)
    set_target_properties(mpi_sposv PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")

    add_executable(mpi_sposv_ll cholesky_ll.c)
    set_target_properties(mpi_sposv_ll PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(mpi_sposv_ll PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    set_target_properties(mpi_sposv_ll PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(mpi_sposv_ll dplasma-mpi mpi_sposv ${MPI_LIBRARIES} ${SPOTRF_REQUIRED_LIBRARIES})

    add_executable(mpi_sposv_rl cholesky_rl.c)
    set_target_properties(mpi_sposv_rl PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(mpi_sposv_rl PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    set_target_properties(mpi_sposv_rl PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(mpi_sposv_rl dplasma-mpi mpi_sposv ${MPI_LIBRARIES} ${SPOTRF_REQUIRED_LIBRARIES})
  endif( DPLASMA_MPI AND MPI_FOUND )

  add_library(sposv STATIC data_management.c ${GPU_SUPPORT} sposv.c)
  set_target_properties(sposv PROPERTIES COMPILE_FLAGS "-DADD_")

  add_executable(sposv_ll cholesky_ll.c)
  set_target_properties(sposv_ll PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(sposv_ll PROPERTIES COMPILE_FLAGS "-DADD_")
  set_target_properties(sposv_ll PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
  target_link_libraries(sposv_ll dplasma sposv ${SPOTRF_REQUIRED_LIBRARIES})

  add_executable(sposv_rl cholesky_rl.c)
  set_target_properties(sposv_rl PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(sposv_rl PROPERTIES COMPILE_FLAGS "-DADD_")
  set_target_properties(sposv_rl PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
  target_link_libraries(sposv_rl dplasma sposv ${SPOTRF_REQUIRED_LIBRARIES})

endif( PLASMA_FOUND )

# add the command to generate the source code
add_custom_command (
  OUTPUT cholesky_rl.c
  COMMAND ../../tools/dpc cholesky_rl.jdf cholesky_rl.c
  MAIN_DEPENDENCY cholesky_rl.jdf
  DEPENDS dpc
  )
 
add_custom_command (
  OUTPUT cholesky_ll.c
  COMMAND ../../tools/dpc cholesky_ll.jdf cholesky_ll.c
  MAIN_DEPENDENCY cholesky_ll.jdf
  DEPENDS dpc
  )
