add_library(common common.c common_timing.h flops.h)
add_dependencies(common dplasma_includes)

if(MPI_C_FOUND)
  set_target_properties(common PROPERTIES COMPILE_FLAGS
                        "${MPI_C_COMPILE_FLAGS}")
endif(MPI_C_FOUND)

include(RulesTestings.cmake)

set(TESTS
  # Lapack Auxiliary
  testing_zgeadd.c
  testing_zprint.c
  testing_zlange.c
  testing_zlanm2.c
  # Level 3 Blas
  testing_zgemm.c
  testing_zhemm.c
  testing_zsymm.c
  testing_zherk.c
  testing_zsyrk.c
  testing_zher2k.c
  testing_zsyr2k.c
  testing_ztrmm.c
  testing_ztrsm.c
  #
  # Lapack
  #
  # Cholesky
  testing_zpotrf.c
  testing_zposv.c
  testing_ztrtri.c
  testing_zpoinv.c
  # QR/LQ
  testing_zgelqf.c
  testing_zgeqrf.c
  testing_zunmqr.c
  testing_zunmlq.c
  testing_zgeqrf_hqr.c
  testing_zgelqf_hqr.c
  testing_zunmqr_hqr.c
  testing_zunmlq_hqr.c
  testing_zgeqrf_systolic.c
  testing_zgelqf_systolic.c
  testing_zunmqr_systolic.c
  testing_zunmlq_systolic.c
  # LU
  testing_zgetrf.c
  testing_zgetrf_incpiv.c
  testing_zgetrf_nopiv.c
  testing_zgetrf_qrf.c
  #testing_zgesv.c
  testing_zgesv_incpiv.c
  # Extra testings
  testing_zgesvd.c
  testing_zgebrd_ge2gb.c
  testing_zgeqrf_rd.c
  testing_zhebut.c
  testing_zhbrdt.c
  testing_zheev.c
  # insert_task interface
  testing_zpotrf_dtd.c
  testing_zgetrf_incpiv_dtd.c
  testing_zgeqrf_dtd.c
)
if( DPLASMA_DEBUG_QR_PIVGEN )
  list(APPEND TESTS testing_zpivgen.c)
  include(TestsQRPivgen.cmake)
endif()

link_directories(${COREBLAS_LIBRARY_DIRS})
testings_addexec(targets "${DPLASMA_PRECISIONS}" "${TESTS}")

# Define Testings
include(Testings.cmake)
