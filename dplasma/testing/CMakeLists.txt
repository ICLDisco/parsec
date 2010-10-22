Add_Subdirectory(common)

include(testings)

set(TESTS
  testing_zgemm.c
  testing_ztrmm.c
  testing_ztrsm.c
)

testings_addexec(targets "+;${DPLASMA_PRECISIONS}" "${TESTS}")

Add_Subdirectory(LLT)
Add_Subdirectory(LU)
Add_Subdirectory(QR)
Add_Subdirectory(TSQR)
Add_Subdirectory(psgeqrf)
Add_Subdirectory(psgemm)
Add_Subdirectory(pdgemm)

