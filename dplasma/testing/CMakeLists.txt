Add_Subdirectory(common)

include(testings)

set(TESTS
  testing_zgemm.c
  testing_ztrmm.c
  testing_ztrsm.c
  testing_zgetrf.c
)

testings_addexec(targets "+;${DPLASMA_PRECISIONS}" "${TESTS}")

Add_Subdirectory(LLT)

Add_Subdirectory(QR)
Add_Subdirectory(TSQR)
Add_Subdirectory(psgeqrf)

