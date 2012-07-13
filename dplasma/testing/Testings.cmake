#
# Shared Memory Testings
#

# check the control in shared memory
add_test(print  ./testing_dprint -N 40 -t 7 -x -v=5)

# check the norms that are used in all other testings
add_test(dlange ./testing_dlange -N 1500 -t 233 -x -v=5)

# Need to add here check on lacpy (Tile => Lapack) and geadd

# BLAS Shared memory
add_test(dtrmm  ./testing_dtrmm -N 1500 -x -v=5)
add_test(dtrsm  ./testing_dtrsm -N 1500 -x -v=5)
add_test(dgemm  ./testing_dgemm -M 1067 -N 2873 -K 987 -t 56 -x -v=5)

# LAPACK shared memory
add_test(dpotrf    ./testing_dpotrf -N 4000 -x -v=5)
add_test(dposv     ./testing_dposv  -N 4000 -x -v=5)
add_test(dpotrf_tq ./testing_dpotrf -N 4000 -x -v=5 -o LTQ)
add_test(dpotrf_pq ./testing_dpotrf -N 4000 -x -v=5 -o PBQ)

add_test(dgetrf        ./testing_dgetrf        -N 4000 -x -v=5)
add_test(dgetrf_incpiv ./testing_dgetrf_incpiv -N 4000 -x -v=5)
add_test(dgesv_incpiv  ./testing_dgesv_incpiv  -N 4000 -x -v=5)

add_test(dgeqrf    ./testing_dgeqrf -N 4000 -x -v=5)
add_test(dgelqf    ./testing_dgelqf -N 4000 -x -v=5)
add_test(dgeqrf_tq ./testing_dgeqrf -N 4000 -x -v=5 -o LTQ)
add_test(dgeqrf_pq ./testing_dgeqrf -N 4000 -x -v=5 -o PBQ)

add_test(dgeqrf_p0 ./testing_dgeqrf_param -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 0 --tsrr=0 -v=5)
add_test(dgeqrf_p1 ./testing_dgeqrf_param -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 1 --tsrr=0 -v=5)
add_test(dgeqrf_p2 ./testing_dgeqrf_param -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 2 --tsrr=0 -v=5)
add_test(dgeqrf_p3 ./testing_dgeqrf_param -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 3 --tsrr=0 -v=5)

#
# Distributed Memory Testings
#
set(MPI_TEST_CMD mpirun -x LD_LIBRARY_PATH -np 8 -hostfile /etc/hostfile -bynode)
if( MPI_FOUND )
  # Check MPI
  add_test(mpi_test   ${MPI_TEST_CMD} /bin/true)

  # check the control in shared memory
  add_test(mpi_print         ${MPI_TEST_CMD} ./testing_dprint        -p 2 -N 40 -t 7 -x -v=5)

  add_test(mpi_dlange        ${MPI_TEST_CMD} ./testing_dlange        -p 4 -N 1500 -t 233 -x -v=5)

  add_test(mpi_dtrmm         ${MPI_TEST_CMD} ./testing_dtrmm         -p 2 -N 1500 -x -v=5)
  add_test(mpi_dtrsm         ${MPI_TEST_CMD} ./testing_dtrsm         -p 4 -N 1500 -x -v=5)
  add_test(mpi_dgemm         ${MPI_TEST_CMD} ./testing_dgemm         -p 4 -M 1067 -N 2873 -K 987 -t 56 -x -v=5)

  add_test(mpi_dpotrf        ${MPI_TEST_CMD} ./testing_dpotrf        -p 2 -N 4000 -x -v=5)
  add_test(mpi_dposv         ${MPI_TEST_CMD} ./testing_dposv         -p 4 -N 4000 -x -v=5)
  add_test(mpi_dpotrf_tq     ${MPI_TEST_CMD} ./testing_dpotrf        -p 2 -N 4000 -x -v=5 -o LTQ)
  add_test(mpi_dpotrf_pq     ${MPI_TEST_CMD} ./testing_dpotrf        -p 2 -N 4000 -x -v=5 -o PBQ)

  add_test(mpi_dgetrf        ${MPI_TEST_CMD} ./testing_dgetrf        -p 1 -N 4000 -x -v=5)
  add_test(mpi_dgetrf_incpiv ${MPI_TEST_CMD} ./testing_dgetrf_incpiv -p 4 -N 4000 -x -v=5)
  add_test(mpi_dgesv_incpiv  ${MPI_TEST_CMD} ./testing_dgesv_incpiv  -p 4 -N 4000 -x -v=5)

  add_test(mpi_dgeqrf        ${MPI_TEST_CMD} ./testing_dgeqrf        -p 4 -N 4000 -x -v=5)
  add_test(mpi_dgelqf        ${MPI_TEST_CMD} ./testing_dgelqf        -p 4 -N 4000 -x -v=5)
  add_test(mpi_dgeqrf_tq     ${MPI_TEST_CMD} ./testing_dgeqrf        -p 4 -N 4000 -x -v=5 -o LTQ)
  add_test(mpi_dgeqrf_pq     ${MPI_TEST_CMD} ./testing_dgeqrf        -p 4 -N 4000 -x -v=5 -o PBQ)

  add_test(mpi_dgeqrf_p0     ${MPI_TEST_CMD} ./testing_dgeqrf_param  -p 4 -N 4000 -t 200 -i 32 -x --qr_p=4 --qr_a=2 --treel 0 --tsrr=0 -v=5)
  add_test(mpi_dgeqrf_p1     ${MPI_TEST_CMD} ./testing_dgeqrf_param  -p 4 -N 4000 -t 200 -i 32 -x --qr_p=4 --qr_a=2 --treel 1 --tsrr=0 -v=5)
  add_test(mpi_dgeqrf_p2     ${MPI_TEST_CMD} ./testing_dgeqrf_param  -p 4 -N 4000 -t 200 -i 32 -x --qr_p=4 --qr_a=2 --treel 2 --tsrr=0 -v=5)
  add_test(mpi_dgeqrf_p3     ${MPI_TEST_CMD} ./testing_dgeqrf_param  -p 4 -N 4000 -t 200 -i 32 -x --qr_p=4 --qr_a=2 --treel 3 --tsrr=0 -v=5)

endif( MPI_FOUND )
