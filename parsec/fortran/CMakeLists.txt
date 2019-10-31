set(sources
    parsecf.F90 parsecf.c)

IF (PARSEC_PROF_TRACE)
  list(APPEND sources parsec_profilef.F90 parsec_profilef.c)
ENDIF (PARSEC_PROF_TRACE)

if(NOT TARGET parsec_fortran)
  add_library(parsec_fortran OBJECT ${sources})
endif(NOT TARGET parsec_fortran)

set_target_properties(parsec_fortran PROPERTIES POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS})
# CMake should be smart enough to set the correct linker language by itself

target_include_directories(parsec_fortran
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_Fortran_MODULE_DIRECTORY}>)
