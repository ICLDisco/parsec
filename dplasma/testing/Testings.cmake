#
# Shared Memory Testings
#
# if (MPI_FOUND)
#   set(SHM_TEST_CMD mpirun -x LD_LIBRARY_PATH -np 1 -hostfile /etc/hostfile -bynode)
# else()
#   unset(SHM_TEST_CMD )
# endif()

# check the control in shared memory
add_test(print  ${SHM_TEST_CMD} ./testing_dprint -N 40 -t 7 -x -v=5)

#
# Check BLAS/Lapack subroutines in shared memory
#
foreach(prec ${DPLASMA_PRECISIONS})

  # check the norms that are used in all other testings
  add_test(${prec}lange ${SHM_TEST_CMD} ./testing_${prec}lange -N 1500 -t 233 -x -v=5)

  # Need to add here check on lacpy (Tile => Lapack) and geadd

  # BLAS Shared memory
  add_test(${prec}trmm  ${SHM_TEST_CMD} ./testing_${prec}trmm          -N 1500 -K 987 -t 56 -x -v=5)
  add_test(${prec}trsm  ${SHM_TEST_CMD} ./testing_${prec}trsm          -N 1500 -K 987 -t 56 -x -v=5)
  add_test(${prec}gemm  ${SHM_TEST_CMD} ./testing_${prec}gemm  -M 1067 -N 2873 -K 987 -t 56 -x -v=5)
  add_test(${prec}symm  ${SHM_TEST_CMD} ./testing_${prec}symm  -M 1067 -N 2873 -K 987 -t 56 -x -v=5)
  add_test(${prec}syrk  ${SHM_TEST_CMD} ./testing_${prec}syrk  -M 2873 -N 2873 -K 987 -t 56 -x -v=5)
  add_test(${prec}syr2k ${SHM_TEST_CMD} ./testing_${prec}syr2k -M 2873 -N 2873 -K 987 -t 56 -x -v=5)

  if ( "${prec}" STREQUAL "c" OR "${prec}" STREQUAL "z" )
    add_test(${prec}hemm  ${SHM_TEST_CMD} ./testing_${prec}hemm  -M 1067 -N 2873 -K 987 -t 56 -x -v=5)
    add_test(${prec}herk  ${SHM_TEST_CMD} ./testing_${prec}herk  -M 2873 -N 2873 -K 987 -t 56 -x -v=5)
    add_test(${prec}her2k ${SHM_TEST_CMD} ./testing_${prec}her2k -M 2873 -N 2873 -K 987 -t 56 -x -v=5)
  endif()

  # LAPACK shared memory
  add_test(${prec}potrf     ${SHM_TEST_CMD} ./testing_${prec}potrf -N 4000 -x -v=5)
  add_test(${prec}posv      ${SHM_TEST_CMD} ./testing_${prec}posv  -N 4000 -K 367 -x -v=5)

  add_test(${prec}getrf        ${SHM_TEST_CMD} ./testing_${prec}getrf        -N 4000 -x -v=5)
  add_test(${prec}getrf_incpiv ${SHM_TEST_CMD} ./testing_${prec}getrf_incpiv -N 4000 -x -v=5)
  add_test(${prec}gesv_incpiv  ${SHM_TEST_CMD} ./testing_${prec}gesv_incpiv  -N 4000 -K 367 -x -v=5)
  add_test(${prec}geqrf        ${SHM_TEST_CMD} ./testing_${prec}geqrf -N 4000 -x -v=5)
  if ( "${prec}" STREQUAL "c" OR "${prec}" STREQUAL "z" )
    add_test(${prec}unmqr        ${SHM_TEST_CMD} ./testing_${prec}unmqr -M 2873 -N 1067 -K 987 -x -v=5)
  else()
    add_test(${prec}ormqr        ${SHM_TEST_CMD} ./testing_${prec}ormqr -M 2873 -N 1067 -K 987 -x -v=5)
  endif()
  add_test(${prec}gelqf        ${SHM_TEST_CMD} ./testing_${prec}gelqf -N 4000 -x -v=5)
  add_test(${prec}geqrf_p0     ${SHM_TEST_CMD} ./testing_${prec}geqrf_param -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 0 --tsrr=0 -v=5)
  add_test(${prec}geqrf_p1     ${SHM_TEST_CMD} ./testing_${prec}geqrf_param -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 1 --tsrr=0 -v=5)
  add_test(${prec}geqrf_p2     ${SHM_TEST_CMD} ./testing_${prec}geqrf_param -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 2 --tsrr=0 -v=5)
  add_test(${prec}geqrf_p3     ${SHM_TEST_CMD} ./testing_${prec}geqrf_param -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 3 --tsrr=0 -v=5)

endforeach()

# Specific cases
# Do we want to test them in all precisions ?
add_test(dpotrf_pbq ${SHM_TEST_CMD} ./testing_dpotrf -N 4000 -x -v=5 -o PBQ)
add_test(dgeqrf_pbq ${SHM_TEST_CMD} ./testing_dgeqrf -N 4000 -x -v=5 -o PBQ)

#
# Distributed Memory Testings
#
set(MPI_TEST_CMD mpirun -x LD_LIBRARY_PATH -np 8 --default-hostfile /etc/hostfile -bynode)
if( MPI_FOUND )
  # Check MPI
  add_test(mpi_test   ${MPI_TEST_CMD} /bin/true)

  # check the control in shared memory
  add_test(mpi_print         ${MPI_TEST_CMD} ./testing_dprint        -p 2 -N 40 -t 7 -x -v=5)

  add_test(mpi_dlange        ${MPI_TEST_CMD} ./testing_dlange        -p 4 -N 1500 -t 233 -x -v=5)

  add_test(mpi_dtrmm         ${MPI_TEST_CMD} ./testing_dtrmm         -p 2 -N 1500 -x -v=5)
  add_test(mpi_dtrsm         ${MPI_TEST_CMD} ./testing_dtrsm         -p 4 -N 1500 -x -v=5)
  add_test(mpi_dgemm         ${MPI_TEST_CMD} ./testing_dgemm         -p 4 -M 1067 -N 2873 -K 987 -t 56 -x -v=5)
  add_test(mpi_dsymm         ${MPI_TEST_CMD} ./testing_dsymm         -p 4 -M 1067 -N 2873 -K 987 -t 56 -x -v=5)
  add_test(mpi_csymm         ${MPI_TEST_CMD} ./testing_csymm         -p 4 -M 1067 -N 2873 -K 987 -t 56 -x -v=5)
  add_test(mpi_chemm         ${MPI_TEST_CMD} ./testing_chemm         -p 4 -M 1067 -N 2873 -K 987 -t 56 -x -v=5)
  add_test(mpi_dsyrk         ${MPI_TEST_CMD} ./testing_dsyrk         -p 4 -M 2873 -N 2873 -K 987 -t 56 -x -v=5)
  add_test(mpi_csyrk         ${MPI_TEST_CMD} ./testing_csyrk         -p 4 -M 2873 -N 2873 -K 987 -t 56 -x -v=5)
  add_test(mpi_cherk         ${MPI_TEST_CMD} ./testing_cherk         -p 4 -M 2873 -N 2873 -K 987 -t 56 -x -v=5)

  add_test(mpi_dpotrf        ${MPI_TEST_CMD} ./testing_dpotrf        -p 2 -N 4000 -x -v=5)
if (CUDA_FOUND)
  add_test(mpi_dpotrf_g1     ${MPI_TEST_CMD} -mca btl_openib_flags 1 ./testing_dpotrf        -p 2 -N 8000 -x -v=5 -g 1)
endif (CUDA_FOUND)
  add_test(mpi_dposv         ${MPI_TEST_CMD} ./testing_dposv         -p 4 -N 4000 -x -v=5)
  add_test(mpi_dpotrf_pbq    ${MPI_TEST_CMD} ./testing_dpotrf        -p 2 -N 4000 -x -v=5 -o PBQ)

  add_test(mpi_dgetrf        ${MPI_TEST_CMD} ./testing_dgetrf        -p 1 -N 4000 -x -v=5)
  add_test(mpi_dgetrf_incpiv ${MPI_TEST_CMD} ./testing_dgetrf_incpiv -p 4 -N 4000 -x -v=5)
  add_test(mpi_dgesv_incpiv  ${MPI_TEST_CMD} ./testing_dgesv_incpiv  -p 4 -N 4000 -x -v=5)

  add_test(mpi_dgeqrf        ${MPI_TEST_CMD} ./testing_dgeqrf        -p 4 -N 4000 -x -v=5)
  add_test(mpi_dgelqf        ${MPI_TEST_CMD} ./testing_dgelqf        -p 4 -N 4000 -x -v=5)
  add_test(mpi_dgeqrf_pbq    ${MPI_TEST_CMD} ./testing_dgeqrf        -p 4 -N 4000 -x -v=5 -o PBQ)

  add_test(mpi_dgeqrf_p0     ${MPI_TEST_CMD} ./testing_dgeqrf_param  -p 4 -N 4000 -t 200 -i 32 -x --qr_p=4 --qr_a=2 --treel 0 --tsrr=0 -v=5)
  add_test(mpi_dgeqrf_p1     ${MPI_TEST_CMD} ./testing_dgeqrf_param  -p 4 -N 4000 -t 200 -i 32 -x --qr_p=4 --qr_a=2 --treel 1 --tsrr=0 -v=5)
  add_test(mpi_dgeqrf_p2     ${MPI_TEST_CMD} ./testing_dgeqrf_param  -p 4 -N 4000 -t 200 -i 32 -x --qr_p=4 --qr_a=2 --treel 2 --tsrr=0 -v=5)
  add_test(mpi_dgeqrf_p3     ${MPI_TEST_CMD} ./testing_dgeqrf_param  -p 4 -N 4000 -t 200 -i 32 -x --qr_p=4 --qr_a=2 --treel 3 --tsrr=0 -v=5)

# The headnode lack GPUs so we need MPI in order to get the test to run on
# one of the nodes.
  if (CUDA_FOUND)
    add_test(dpotrf_g1  ${MPI_TEST_CMD} ./testing_dpotrf -N 8000 -x -v=5 -g 1)
    add_test(dpotrf_g2  ${MPI_TEST_CMD} ./testing_dpotrf -N 8000 -x -v=5 -g 2)
  endif (CUDA_FOUND)
endif( MPI_FOUND )
