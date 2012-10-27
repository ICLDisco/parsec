#
# Shared Memory Testings
#
if (MPI_FOUND)
  set(SHM_TEST_CMD mpirun -x LD_LIBRARY_PATH -np 1 -hostfile /etc/hostfile -bynode)
else()
  unset(SHM_TEST_CMD )
endif()

# check the control in shared memory
add_test(print  ${SHM_TEST_CMD} ./testing_dprint -N 40 -t 7 -x -v=5)

# check the norms that are used in all other testings
add_test(dlange ${SHM_TEST_CMD} ./testing_dlange -N 1500 -t 233 -x -v=5)

# Need to add here check on lacpy (Tile => Lapack) and geadd

# BLAS Shared memory
add_test(dtrmm ${SHM_TEST_CMD} ./testing_dtrmm -N 1500 -x -v=5)
add_test(dtrsm ${SHM_TEST_CMD} ./testing_dtrsm -N 1500 -x -v=5)
add_test(dgemm ${SHM_TEST_CMD} ./testing_dgemm -M 1067 -N 2873 -K 987 -t 56 -x -v=5)
add_test(dsymm ${SHM_TEST_CMD} ./testing_dsymm -M 1067 -N 2873 -K 987 -t 56 -x -v=5)
add_test(csymm ${SHM_TEST_CMD} ./testing_chemm -M 1067 -N 2873 -K 987 -t 56 -x -v=5)

# LAPACK shared memory
add_test(dpotrf     ${SHM_TEST_CMD} ./testing_dpotrf -N 4000 -x -v=5)
if (CUDA_FOUND)
  add_test(dpotrf_g1  ${SHM_TEST_CMD} ./testing_dpotrf -N 8000 -x -v=5 -g 1)
  add_test(dpotrf_g2  ${SHM_TEST_CMD} ./testing_dpotrf -N 8000 -x -v=5 -g 2)
endif (CUDA_FOUND)
add_test(dposv      ${SHM_TEST_CMD} ./testing_dposv  -N 4000 -x -v=5)
add_test(dpotrf_pbq ${SHM_TEST_CMD} ./testing_dpotrf -N 4000 -x -v=5 -o PBQ)

add_test(dgetrf        ${SHM_TEST_CMD} ./testing_dgetrf        -N 4000 -x -v=5)
add_test(dgetrf_incpiv ${SHM_TEST_CMD} ./testing_dgetrf_incpiv -N 4000 -x -v=5)
add_test(dgesv_incpiv  ${SHM_TEST_CMD} ./testing_dgesv_incpiv  -N 4000 -x -v=5)

add_test(dgeqrf     ${SHM_TEST_CMD} ./testing_dgeqrf -N 4000 -x -v=5)
add_test(dgelqf     ${SHM_TEST_CMD} ./testing_dgelqf -N 4000 -x -v=5)
add_test(dgeqrf_pbq ${SHM_TEST_CMD} ./testing_dgeqrf -N 4000 -x -v=5 -o PBQ)

add_test(dgeqrf_p0 ${SHM_TEST_CMD} ./testing_dgeqrf_param -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 0 --tsrr=0 -v=5)
add_test(dgeqrf_p1 ${SHM_TEST_CMD} ./testing_dgeqrf_param -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 1 --tsrr=0 -v=5)
add_test(dgeqrf_p2 ${SHM_TEST_CMD} ./testing_dgeqrf_param -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 2 --tsrr=0 -v=5)
add_test(dgeqrf_p3 ${SHM_TEST_CMD} ./testing_dgeqrf_param -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 3 --tsrr=0 -v=5)

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

endif( MPI_FOUND )
