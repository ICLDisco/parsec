include(FortranCInterface)
FortranCInterface_HEADER(f2c_mangle.h
                             MACRO_NAMESPACE "PARSEC_F2C_"
                             SYMBOL_NAMESPACE "parsec_f2c_")

set(sources
    parsecf.F90 parsecf.c)

if (PARSEC_PROF_TRACE)
  list(APPEND sources parsec_profilef.F90 parsec_profilef.c)
endif (PARSEC_PROF_TRACE)

target_sources(parsec PRIVATE ${sources})

target_include_directories(parsec
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_Fortran_MODULE_DIRECTORY}>)
