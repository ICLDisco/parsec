#
# Shared Memory Testings
#

# check the control in shared memory
add_test(print  ${SHM_TEST_CMD_LIST} ./testing_dprint -N 40 -t 7 -x -v=5)

#
# Check BLAS/Lapack subroutines in shared memory
#
foreach(prec ${DPLASMA_PRECISIONS})

  # check the norms that are used in all other testings
  add_test(${prec}lange ${SHM_TEST_CMD_LIST} ./testing_${prec}lange -N 1500 -t 233 -x -v=5)

  # Need to add here check on lacpy (Tile => Lapack) and geadd

  # BLAS Shared memory
  add_test(${prec}trmm  ${SHM_TEST_CMD_LIST} ./testing_${prec}trmm          -N 1500 -K 987 -t 56 -x -v=5)
  add_test(${prec}trsm  ${SHM_TEST_CMD_LIST} ./testing_${prec}trsm          -N 1500 -K 987 -t 56 -x -v=5)
  add_test(${prec}gemm  ${SHM_TEST_CMD_LIST} ./testing_${prec}gemm  -M 1067 -N 2873 -K 987 -t 56 -x -v=5)
  add_test(${prec}symm  ${SHM_TEST_CMD_LIST} ./testing_${prec}symm  -M 1067 -N 2873 -K 987 -t 56 -x -v=5)
  add_test(${prec}syrk  ${SHM_TEST_CMD_LIST} ./testing_${prec}syrk  -M 2873 -N 2873 -K 987 -t 56 -x -v=5)
  add_test(${prec}syr2k ${SHM_TEST_CMD_LIST} ./testing_${prec}syr2k -M 2873 -N 2873 -K 987 -t 56 -x -v=5)

  if ( "${prec}" STREQUAL "c" OR "${prec}" STREQUAL "z" )
    add_test(${prec}hemm  ${SHM_TEST_CMD_LIST} ./testing_${prec}hemm  -M 1067 -N 2873 -K 987 -t 56 -x -v=5)
    add_test(${prec}herk  ${SHM_TEST_CMD_LIST} ./testing_${prec}herk  -M 2873 -N 2873 -K 987 -t 56 -x -v=5)
    add_test(${prec}her2k ${SHM_TEST_CMD_LIST} ./testing_${prec}her2k -M 2873 -N 2873 -K 987 -t 56 -x -v=5)
  endif()

  # LAPACK shared memory
  add_test(${prec}potrf     ${SHM_TEST_CMD_LIST} ./testing_${prec}potrf -N 4000 -x -v=5)
  add_test(${prec}posv      ${SHM_TEST_CMD_LIST} ./testing_${prec}posv  -N 4000 -K 367 -x -v=5)

  add_test(${prec}getrf        ${SHM_TEST_CMD_LIST} ./testing_${prec}getrf        -N 4000 -x -v=5)
  add_test(${prec}getrf_incpiv ${SHM_TEST_CMD_LIST} ./testing_${prec}getrf_incpiv -N 4000 -x -v=5)
  add_test(${prec}gesv_incpiv  ${SHM_TEST_CMD_LIST} ./testing_${prec}gesv_incpiv  -N 4000 -K 367 -x -v=5)
  add_test(${prec}geqrf        ${SHM_TEST_CMD_LIST} ./testing_${prec}geqrf -N 4000 -x -v=5)
  if ( "${prec}" STREQUAL "c" OR "${prec}" STREQUAL "z" )
    add_test(${prec}unmqr        ${SHM_TEST_CMD_LIST} ./testing_${prec}unmqr -M 2873 -N 1067 -K 987 -x -v=5)
  else()
    add_test(${prec}ormqr        ${SHM_TEST_CMD_LIST} ./testing_${prec}ormqr -M 2873 -N 1067 -K 987 -x -v=5)
  endif()
  add_test(${prec}gelqf        ${SHM_TEST_CMD_LIST} ./testing_${prec}gelqf -N 4000 -x -v=5)
  add_test(${prec}geqrf_p0     ${SHM_TEST_CMD_LIST} ./testing_${prec}geqrf_param -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 0 --tsrr=0 -v=5)
  add_test(${prec}geqrf_p1     ${SHM_TEST_CMD_LIST} ./testing_${prec}geqrf_param -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 1 --tsrr=0 -v=5)
  add_test(${prec}geqrf_p2     ${SHM_TEST_CMD_LIST} ./testing_${prec}geqrf_param -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 2 --tsrr=0 -v=5)
  add_test(${prec}geqrf_p3     ${SHM_TEST_CMD_LIST} ./testing_${prec}geqrf_param -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 3 --tsrr=0 -v=5)

endforeach()

# Specific cases
# Do we want to test them in all precisions ?
add_test(dpotrf_pbq ${SHM_TEST_CMD_LIST} ./testing_dpotrf -N 4000 -x -v=5 -o PBQ)
add_test(dgeqrf_pbq ${SHM_TEST_CMD_LIST} ./testing_dgeqrf -N 4000 -x -v=5 -o PBQ)

# The headnode lack GPUs so we need MPI in order to get the test to run on
# one of the nodes.
if (CUDA_FOUND AND MPI_FOUND)
  add_test(dpotrf_g1  ${SHM_TEST_CMD_LIST} ./testing_dpotrf -N 8000 -x -v=5 -g 1)
  add_test(dpotrf_g2  ${SHM_TEST_CMD_LIST} ./testing_dpotrf -N 8000 -x -v=5 -g 2)
endif (CUDA_FOUND AND MPI_FOUND)

#
# Distributed Memory Testings
#
if( MPI_FOUND )
  # Check MPI
  add_test(mpi_test   ${MPI_TEST_CMD_LIST} /bin/true)

  # check the control in shared memory
  add_test(mpi_print         ${MPI_TEST_CMD_LIST} ./testing_dprint        -p 2 -N 40 -t 7 -x -v=5)
  SET_TESTS_PROPERTIES("mpi_print" PROPERTIES DEPENDS "mpi_test")

  add_test(mpi_dlange        ${MPI_TEST_CMD_LIST} ./testing_dlange        -p 4 -N 1500 -t 233 -x -v=5)
  SET_TESTS_PROPERTIES("mpi_dlange" PROPERTIES DEPENDS "mpi_test")

  add_test(mpi_dtrmm         ${MPI_TEST_CMD_LIST} ./testing_dtrmm         -p 2 -N 1500 -x -v=5)
  SET_TESTS_PROPERTIES("mpi_dtrmm" PROPERTIES DEPENDS "mpi_test")

  add_test(mpi_dtrsm         ${MPI_TEST_CMD_LIST} ./testing_dtrsm         -p 4 -N 1500 -x -v=5)
  SET_TESTS_PROPERTIES("mpi_dtrsm" PROPERTIES DEPENDS "mpi_test")

  add_test(mpi_dgemm         ${MPI_TEST_CMD_LIST} ./testing_dgemm         -p 4 -M 1067 -N 2873 -K 987 -t 56 -x -v=5)
  SET_TESTS_PROPERTIES("mpi_dgemm" PROPERTIES DEPENDS "mpi_test")

  add_test(mpi_dsymm         ${MPI_TEST_CMD_LIST} ./testing_dsymm         -p 4 -M 1067 -N 2873 -K 987 -t 56 -x -v=5)
  SET_TESTS_PROPERTIES("mpi_dsymm" PROPERTIES DEPENDS "mpi_test")

  add_test(mpi_csymm         ${MPI_TEST_CMD_LIST} ./testing_csymm         -p 4 -M 1067 -N 2873 -K 987 -t 56 -x -v=5)
  SET_TESTS_PROPERTIES("mpi_csymm" PROPERTIES DEPENDS "mpi_test")

  add_test(mpi_chemm         ${MPI_TEST_CMD_LIST} ./testing_chemm         -p 4 -M 1067 -N 2873 -K 987 -t 56 -x -v=5)
  SET_TESTS_PROPERTIES("mpi_chemm" PROPERTIES DEPENDS "mpi_test")

  add_test(mpi_dsyrk         ${MPI_TEST_CMD_LIST} ./testing_dsyrk         -p 4 -M 2873 -N 2873 -K 987 -t 56 -x -v=5)
  SET_TESTS_PROPERTIES("mpi_dsyrk" PROPERTIES DEPENDS "mpi_test")

  add_test(mpi_csyrk         ${MPI_TEST_CMD_LIST} ./testing_csyrk         -p 4 -M 2873 -N 2873 -K 987 -t 56 -x -v=5)
  SET_TESTS_PROPERTIES("mpi_csyrk" PROPERTIES DEPENDS "mpi_test")

  add_test(mpi_cherk         ${MPI_TEST_CMD_LIST} ./testing_cherk         -p 4 -M 2873 -N 2873 -K 987 -t 56 -x -v=5)
  SET_TESTS_PROPERTIES("mpi_cherk" PROPERTIES DEPENDS "mpi_test")

  add_test(mpi_dpotrf        ${MPI_TEST_CMD_LIST} ./testing_dpotrf        -p 2 -N 4000 -x -v=5)
  SET_TESTS_PROPERTIES("mpi_dpotrf" PROPERTIES DEPENDS "mpi_test")

if (CUDA_FOUND)
  add_test(mpi_dpotrf_g1     ${MPI_TEST_CMD_LIST} -mca btl_openib_flags 1 ./testing_dpotrf        -p 2 -N 8000 -x -v=5 -g 1)
  SET_TESTS_PROPERTIES("mpi_dpotrf_g1" PROPERTIES DEPENDS "mpi_test")

endif (CUDA_FOUND)
  add_test(mpi_dposv         ${MPI_TEST_CMD_LIST} ./testing_dposv         -p 4 -N 4000 -x -v=5)
  SET_TESTS_PROPERTIES("mpi_dposv" PROPERTIES DEPENDS "mpi_test")

  add_test(mpi_dpotrf_pbq    ${MPI_TEST_CMD_LIST} ./testing_dpotrf        -p 2 -N 4000 -x -v=5 -o PBQ)
  SET_TESTS_PROPERTIES("mpi_dpotrf_pbq" PROPERTIES DEPENDS "mpi_test")


  add_test(mpi_dgetrf        ${MPI_TEST_CMD_LIST} ./testing_dgetrf        -p 1 -N 4000 -x -v=5)
  SET_TESTS_PROPERTIES("mpi_dgetrf" PROPERTIES DEPENDS "mpi_test")

  add_test(mpi_dgetrf_incpiv ${MPI_TEST_CMD_LIST} ./testing_dgetrf_incpiv -p 4 -N 4000 -x -v=5)
  SET_TESTS_PROPERTIES("mpi_dgetrf_incpiv" PROPERTIES DEPENDS "mpi_test")

  add_test(mpi_dgesv_incpiv  ${MPI_TEST_CMD_LIST} ./testing_dgesv_incpiv  -p 4 -N 4000 -x -v=5)
  SET_TESTS_PROPERTIES("mpi_dgesv_incpiv" PROPERTIES DEPENDS "mpi_test")


  add_test(mpi_dgeqrf        ${MPI_TEST_CMD_LIST} ./testing_dgeqrf        -p 4 -N 4000 -x -v=5)
  SET_TESTS_PROPERTIES("mpi_dgeqrf" PROPERTIES DEPENDS "mpi_test")

  add_test(mpi_dgelqf        ${MPI_TEST_CMD_LIST} ./testing_dgelqf        -p 4 -N 4000 -x -v=5)
  SET_TESTS_PROPERTIES("mpi_dgelqf" PROPERTIES DEPENDS "mpi_test")

  add_test(mpi_dgeqrf_pbq    ${MPI_TEST_CMD_LIST} ./testing_dgeqrf        -p 4 -N 4000 -x -v=5 -o PBQ)
  SET_TESTS_PROPERTIES("mpi_dgeqrf_pbq" PROPERTIES DEPENDS "mpi_test")

  add_test(mpi_dgeqrf_p0     ${MPI_TEST_CMD_LIST} ./testing_dgeqrf_param  -p 4 -N 4000 -t 200 -i 32 -x --qr_p=4 --qr_a=2 --treel 0 --tsrr=0 -v=5)
  SET_TESTS_PROPERTIES("mpi_dgeqrf_p0" PROPERTIES DEPENDS "mpi_test")

  add_test(mpi_dgeqrf_p1     ${MPI_TEST_CMD_LIST} ./testing_dgeqrf_param  -p 4 -N 4000 -t 200 -i 32 -x --qr_p=4 --qr_a=2 --treel 1 --tsrr=0 -v=5)
  SET_TESTS_PROPERTIES("mpi_dgeqrf_p1" PROPERTIES DEPENDS "mpi_test")

  add_test(mpi_dgeqrf_p2     ${MPI_TEST_CMD_LIST} ./testing_dgeqrf_param  -p 4 -N 4000 -t 200 -i 32 -x --qr_p=4 --qr_a=2 --treel 2 --tsrr=0 -v=5)
  SET_TESTS_PROPERTIES("mpi_dgeqrf_p2" PROPERTIES DEPENDS "mpi_test")

  add_test(mpi_dgeqrf_p3     ${MPI_TEST_CMD_LIST} ./testing_dgeqrf_param  -p 4 -N 4000 -t 200 -i 32 -x --qr_p=4 --qr_a=2 --treel 3 --tsrr=0 -v=5)
  SET_TESTS_PROPERTIES("mpi_dgeqrf_p3" PROPERTIES DEPENDS "mpi_test")

endif( MPI_FOUND )
