include(ParsecCompilePTG)

parsec_addtest_executable(C async)
target_ptg_sources(async PRIVATE "async.jdf")

if(MPI_Fortran_FOUND AND CMAKE_Fortran_COMPILER_WORKS)
  if(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
    parsec_addtest_executable(Fortran generate_f SOURCES generate_f.F90)
  endif(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
endif(MPI_Fortran_FOUND AND CMAKE_Fortran_COMPILER_WORKS)

