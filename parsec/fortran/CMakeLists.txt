set(sources
    parsecf.F90 parsecf.c)

IF (PARSEC_PROF_TRACE)
  list(APPEND sources parsec_profilef.F90 parsec_profilef.c)
ENDIF (PARSEC_PROF_TRACE)

if(NOT TARGET parsec_fortran)
  add_library(parsec_fortran OBJECT ${sources})
endif(NOT TARGET parsec_fortran)

set_target_properties(parsec_fortran PROPERTIES POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS})

target_include_directories(parsec_fortran
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_Fortran_MODULE_DIRECTORY}>)
target_link_libraries(parsec_fortran
  INTERFACE
  $<$<BOOL:${MPI_Fortran_FOUND}>:MPI::MPI_Fortran>
)
