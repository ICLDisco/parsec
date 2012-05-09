if(MPI_FOUND)
  add_library(common-mpi common.c common_timing.h flops.h)
  set_target_properties(common-mpi PROPERTIES COMPILE_FLAGS
      "${MPI_COMPILE_FLAGS}")
else()
  add_library(common common.c common_timing.h flops.h)
endif()

include(RulesTestings.cmake)

set(TESTS
  testing_zgemm.c
  testing_ztrmm.c
  testing_ztrsm.c
  testing_zpotrf.c
  testing_zhebut.c
  testing_zposv.c
  testing_zgetrf.c
  testing_zgetrf_incpiv.c
  testing_zgetrf_incpiv_sd.c
  testing_zgesv_incpiv.c
  testing_zgeqrf.c
  testing_zgeqrf_param.c
  testing_zgetrf_hincpiv.c
  testing_zgetrf_hpp.c
  testing_zgetrf_hpp_multithrd.c
  testing_zgelqf.c
  testing_zherbt.c
  testing_zhbrdt.c
  testing_zheev.c
  testing_zlange.c
  #testing_zgesvd.c
)
if( DPLASMA_DEBUG_QR_PIVGEN )
  list(APPEND TESTS testing_zpivgen.c)
  include(TestsQRPivgen.cmake)
endif()

testings_addexec(targets "${DPLASMA_PRECISIONS}" "${TESTS}")

#Add the LL define to potrf LL
# testings_addexec(lltargets "+;${DPLASMA_PRECISIONS}" "testing_zpotrf_ll.c")
# foreach(target ${lltargets})
#   get_target_property(${target}_COMPILE_FLAGS ${target} COMPILE_FLAGS)
#   set_target_properties(${target} PROPERTIES COMPILE_FLAGS "${${target}_COMPILE_FLAGS} -DLLT_LL")
# endforeach()

Add_Subdirectory(TSQR)

# BLAS Shared memory
add_test(dgemm  ./testing_dgemm -N 1500 -x -v=4)
add_test(dtrmm  ./testing_dtrmm -N 1500 -x -v=4)
add_test(dtrsm  ./testing_dtrsm -N 1500 -x -v=4)

# LAPACK shared memory
add_test(dpotrf    ./testing_dpotrf -N 4000 -x -v=4)
add_test(dposv     ./testing_dposv  -N 4000 -x -v=4)
add_test(dgeqrf    ./testing_dgeqrf -N 4000 -x -v=4)
add_test(dpotrf_tq ./testing_dpotrf -N 4000 -x -v=4 -o LTQ)
add_test(dpotrf_pq ./testing_dpotrf -N 4000 -x -v=4 -o PBQ)
add_test(dgeqrf_tq ./testing_dgeqrf -N 4000 -x -v=4 -o LTQ)
add_test(dgeqrf_pq ./testing_dgeqrf -N 4000 -x -v=4 -o PBQ)
add_test(dgeqrf_p0 ./testing_dgeqrf_param -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 0 -v=4)
add_test(dgeqrf_p1 ./testing_dgeqrf_param -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 1 -v=4)
add_test(dgeqrf_p2 ./testing_dgeqrf_param -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 2 -v=4)
add_test(dgeqrf_p3 ./testing_dgeqrf_param -N 4000 -t 200 -i 32 -x --qr_a=2 --treel 3 -v=4)

set(MPI_TEST_CMD mpirun -x LD_LIBRARY_PATH -np 8 -hostfile /etc/hostfile -bynode)
if( MPI_FOUND )
  # Check MPI
  add_test(mpi_test   ${MPI_TEST_CMD} /bin/true)

  # No BLAS checked in distributed memory otherwise we will need to link against PBLAS
  add_test(mpi_dgemm  ${MPI_TEST_CMD} ./testing_dgemm -N 4000 -p 2)

  # LAPACK distributed memory
  add_test(mpi_dpotrf    ${MPI_TEST_CMD} ./testing_dpotrf -N 4000 -p 2 -x -v=4)
  add_test(mpi_dposv     ${MPI_TEST_CMD} ./testing_dposv  -N 4000 -p 4 -x -v=4)
  add_test(mpi_dgeqrf    ${MPI_TEST_CMD} ./testing_dgeqrf -N 4000 -p 4 -x -v=4)
  add_test(mpi_dpotrf_tq ${MPI_TEST_CMD} ./testing_dpotrf -N 4000 -p 2 -x -v=4 -o LTQ)
  add_test(mpi_dgeqrf_tq ${MPI_TEST_CMD} ./testing_dgeqrf -N 4000 -p 4 -x -v=4 -o LTQ)
  add_test(mpi_dpotrf_pq ${MPI_TEST_CMD} ./testing_dpotrf -N 4000 -p 2 -x -v=4 -o PBQ)
  add_test(mpi_dgeqrf_pq ${MPI_TEST_CMD} ./testing_dgeqrf -N 4000 -p 4 -x -v=4 -o PBQ)
  add_test(mpi_dgeqrf_p0 ${MPI_TEST_CMD} ./testing_dgeqrf_param -N 4000 -t 200 -i 32 -p 4 -x --qr_p=8 --qr_a=2 --treel 0 -v=4)
  add_test(mpi_dgeqrf_p1 ${MPI_TEST_CMD} ./testing_dgeqrf_param -N 4000 -t 200 -i 32 -p 4 -x --qr_p=8 --qr_a=2 --treel 1 -v=4)
  add_test(mpi_dgeqrf_p2 ${MPI_TEST_CMD} ./testing_dgeqrf_param -N 4000 -t 200 -i 32 -p 4 -x --qr_p=8 --qr_a=2 --treel 2 -v=4)
  add_test(mpi_dgeqrf_p3 ${MPI_TEST_CMD} ./testing_dgeqrf_param -N 4000 -t 200 -i 32 -p 4 -x --qr_p=8 --qr_a=2 --treel 3 -v=4)

endif( MPI_FOUND )
