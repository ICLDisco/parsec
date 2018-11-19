set(sources
    parsecf.F90 parsecf.c)

IF (PARSEC_PROF_TRACE)
  list(APPEND sources parsec_profilef.F90 parsec_profilef.c)
ENDIF (PARSEC_PROF_TRACE)

add_library(parsecf OBJECT ${sources})
set_target_properties(parsecf PROPERTIES COMPILE_FLAGS "${MPI_Fortran_COMPILE_FLAGS}")
if(BUILD_SHARED_LIBS)
  set_target_properties(parsecf PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif(BUILD_SHARED_LIBS)

add_library(parsec_fortran INTERFACE)
target_sources(parsec_fortran INTERFACE $<TARGET_OBJECTS:parsecf>)

