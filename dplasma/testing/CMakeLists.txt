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
  testing_zhemm.c
  testing_zsymm.c
  testing_zherk.c
  testing_zsyrk.c
  testing_zher2k.c
  testing_zsyr2k.c
  testing_ztrmm.c
  testing_ztrsm.c
  #
  testing_zpotrf.c
  testing_zhebut.c
  testing_zposv.c
  testing_zgetrf.c
  testing_zgetrf_incpiv.c
  testing_zgetrf_incpiv_sd.c
#  testing_zgetrf_nopiv.c
  testing_zgesv_incpiv.c
  testing_zgeqrf.c
  testing_zgeqrf_systolic.c
  testing_zgeqrf_param.c
  testing_zgeqrf_rd.c
  testing_zunmqr.c
  testing_zgelqf.c
  testing_zherbt.c
  testing_zhbrdt.c
  testing_zheev.c
  testing_zlange.c
  testing_zprint.c
  #testing_zgesvd.c
)
if( DPLASMA_DEBUG_QR_PIVGEN )
  list(APPEND TESTS testing_zpivgen.c)
  include(TestsQRPivgen.cmake)
endif()

link_directories(${PLASMA_LIBRARY_DIRS})
testings_addexec(targets "${DPLASMA_PRECISIONS}" "${TESTS}")

#Add the LL define to potrf LL
# testings_addexec(lltargets "+;${DPLASMA_PRECISIONS}" "testing_zpotrf_ll.c")
# foreach(target ${lltargets})
#   get_target_property(${target}_COMPILE_FLAGS ${target} COMPILE_FLAGS)
#   set_target_properties(${target} PROPERTIES COMPILE_FLAGS "${${target}_COMPILE_FLAGS} -DLLT_LL")
# endforeach()

# Define Testings
include(Testings.cmake)
