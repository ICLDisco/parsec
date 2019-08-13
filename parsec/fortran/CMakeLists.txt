set(sources
    parsecf.F90 parsecf.c)

IF (PARSEC_PROF_TRACE)
  list(APPEND sources parsec_profilef.F90 parsec_profilef.c)
ENDIF (PARSEC_PROF_TRACE)

if(NOT TARGET parsecf)
  add_library(parsecf OBJECT ${sources})
endif(NOT TARGET parsecf)

if(BUILD_SHARED_LIBS)
  set_target_properties(parsecf PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif(BUILD_SHARED_LIBS)

if(NOT TARGET parsec_fortran)
  add_library(parsec_fortran INTERFACE)
endif(NOT TARGET parsec_fortran)

target_sources(parsec_fortran INTERFACE $<TARGET_OBJECTS:parsecf>)

target_link_libraries(parsec_fortran
  INTERFACE
  $<$<BOOL:${MPI_Fortran_FOUND}>:MPI::MPI_Fortran>
)
