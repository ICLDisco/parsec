Add_Subdirectory(ptg-compiler)

IF (CMAKE_CXX_COMPILER_WORKS)
  Add_Subdirectory(q2j)
ENDIF (CMAKE_CXX_COMPILER_WORKS)
