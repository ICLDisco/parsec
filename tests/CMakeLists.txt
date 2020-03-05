function(parsec_addtest lang target)
  add_executable(${target} ${ARGN})
  set_target_properties(${target} PROPERTIES
    LINKER_LANGUAGE ${lang})
  target_link_libraries(${target}
    PRIVATE
      parsec
      $<$<STREQUAL:Fortran,${lang}>:parsec_fortran>)
endfunction(parsec_addtest)

add_Subdirectory(unit)
add_Subdirectory(interfaces/superscalar)
add_Subdirectory(interfaces/ptg)
add_Subdirectory(jdf/new_null)
add_Subdirectory(haar-tree-project)
add_Subdirectory(pingpong)
add_Subdirectory(branching)
add_Subdirectory(choice)
add_Subdirectory(controlgather)
add_Subdirectory(scheduling)
add_Subdirectory(all2all)
add_Subdirectory(generalized_reduction)
add_Subdirectory(merge_sort)
add_Subdirectory(two_dim_band)
add_Subdirectory(stencil)
if(PARSEC_PROF_TRACE)
  add_Subdirectory(profiling)
  add_Subdirectory(standalone-profiling)
endif(PARSEC_PROF_TRACE)

parsec_addtest(C operator operator.c)
parsec_addtest(C reduce reduce.c)
parsec_addtest(C dtt_bug_replicator "dtt_bug_replicator_ex.c")
target_ptg_sources(dtt_bug_replicator PRIVATE "dtt_bug_replicator.jdf")

parsec_addtest(C write_check "vector.c")
target_ptg_sources(write_check PRIVATE "write_check.jdf")

parsec_addtest(C touch_ex "touch_ex.c")
target_ptg_sources(touch_ex PRIVATE "touch.jdf")
parsec_addtest(C touch_ex_inline "touch_ex.c")
target_ptg_sources(touch_ex_inline PRIVATE "touch.jdf")
target_compile_definitions(touch_ex_inline PRIVATE BUILDING_PARSEC)
target_compile_options(touch_ex_inline PRIVATE ${PARSEC_ATOMIC_SUPPORT_OPTIONS})
if(MPI_Fortran_FOUND AND CMAKE_Fortran_COMPILER_WORKS)
  if(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
    parsec_addtest(Fortran touch_exf "touch_exf.F90")
    target_link_libraries(touch_exf PRIVATE $<$<BOOL:${MPI_Fortran_FOUND}>:MPI::MPI_Fortran>)
    target_ptg_sources(touch_exf PRIVATE "touch.jdf")
  endif(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
endif(MPI_Fortran_FOUND AND CMAKE_Fortran_COMPILER_WORKS)

parsec_addtest(C strange)
target_ptg_sources(strange PRIVATE "strange.jdf")

parsec_addtest(C startup)
target_ptg_sources(startup PRIVATE "startup.jdf")

parsec_addtest(C complex_deps)
target_ptg_sources(complex_deps PRIVATE "complex_deps.jdf")

if( MPI_C_FOUND )
  parsec_addtest(C multichain)
  target_ptg_sources(multichain PRIVATE "multichain.jdf")
endif( MPI_C_FOUND )

parsec_addtest(C compose "compose.c")

# Define Testings
include(Testings.cmake)
