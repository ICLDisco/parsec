if(DAGUE_MPI AND MPI_FOUND)
  add_library(common-mpi common.c common_timing.h)
  set_target_properties(common-mpi PROPERTIES COMPILE_FLAGS "-DUSE_MPI")
else()
  add_library(common common.c common_timing.h)
endif()
  
include(TestingRules.cmake)

set(TESTS
  testing_zgemm.c
  testing_ztrmm.c
  testing_ztrsm.c
  testing_zpotrf.c
  testing_zgetrf.c
  testing_zgetrf_sd.c
  testing_zgeqrf.c
)
testings_addexec(targets "+;${DPLASMA_PRECISIONS}" "${TESTS}")

#Add the LL define to potrf LL 
testings_addexec(lltargets "+;${DPLASMA_PRECISIONS}" "testing_zpotrf_ll.c")
foreach(target ${lltargets})
  get_target_property(${target}_COMPILE_FLAGS ${target} COMPILE_FLAGS)
  set_target_properties(${target} PROPERTIES COMPILE_FLAGS "${${target}_COMPILE_FLAGS} -DLLT_LL")
endforeach()

Add_Subdirectory(TSQR)

