if(TARGET parsec-ptgpp)
  parsec_addtest_executable(C touch_ex SOURCES touch_ex.c)
  target_ptg_sources(touch_ex PRIVATE "touch.jdf")

  parsec_addtest_executable(C touch_ex_inline SOURCES touch_ex.c)
  target_ptg_sources(touch_ex_inline PRIVATE "touch.jdf")
  target_compile_definitions(touch_ex_inline PRIVATE BUILDING_PARSEC)
  target_compile_options(touch_ex_inline PRIVATE ${PARSEC_ATOMIC_SUPPORT_OPTIONS})

  if(MPI_Fortran_FOUND AND CMAKE_Fortran_COMPILER_WORKS)
    if(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
      parsec_addtest_executable(Fortran touch_exf SOURCES touch_exf.F90)
      target_link_libraries(touch_exf PRIVATE $<$<BOOL:${MPI_Fortran_FOUND}>:MPI::MPI_Fortran>)
      target_ptg_sources(touch_exf PRIVATE "touch.jdf")
      # In some corner cases (using clang to compile the C parsec library
      # but gfortran to compile touch_exf.F90), for a reason we did not
      # understand, touch.c.o needs to be compiled with the option -fPIE
      # when linking with touch_exf. As this does not seem to impact negatively
      # in other cases, we decided to set this property here)
      set_target_properties(touch_exf PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
    endif(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
  endif(MPI_Fortran_FOUND AND CMAKE_Fortran_COMPILER_WORKS)
endif(TARGET parsec-ptgpp)

parsec_addtest_executable(C operator SOURCES operator.c)
parsec_addtest_executable(C compose SOURCES compose.c)
