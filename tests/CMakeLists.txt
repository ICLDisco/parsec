macro(parsec_addtest lang target)
  add_executable(${target} ${ARGN})
  set_target_properties(${target} PROPERTIES
    LINKER_LANGUAGE ${lang})
  target_link_libraries(${target}
    PRIVATE
      parsec
      $<$<STREQUAL:Fortran,${lang}>:parsec_fortran>)
endmacro(parsec_addtest)

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
parsec_compile_ptg(dtt_generated "${CMAKE_CURRENT_SOURCE_DIR}/dtt_bug_replicator.jdf")
parsec_addtest(C dtt_bug_replicator "dtt_bug_replicator_ex.c;${dtt_generated}")

parsec_compile_ptg(wc_generated "${CMAKE_CURRENT_SOURCE_DIR}/write_check.jdf")
string( REGEX REPLACE "write_check.[hc]" "" include_dir "${wc_generated}" )
include_directories(${include_dir})
parsec_addtest(C write_check "vector.c;${wc_generated}")

parsec_compile_ptg(touch_generated "${CMAKE_CURRENT_SOURCE_DIR}/touch.jdf")
string( REGEX REPLACE "touch.[hc]" "" include_dir "${touch_generated}" )
include_directories(${include_dir})

parsec_addtest(C touch_ex "touch_ex.c;${touch_generated}")
if(MPI_Fortran_FOUND AND CMAKE_Fortran_COMPILER_WORKS)
  if(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
    parsec_addtest(Fortran touch_exf "touch_exf.F90;${touch_generated}")
  endif(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
endif(MPI_Fortran_FOUND AND CMAKE_Fortran_COMPILER_WORKS)

parsec_compile_ptg(strange_generated "${CMAKE_CURRENT_SOURCE_DIR}/strange.jdf")
string( REGEX REPLACE "strange.[hc]" "" include_dir "${strange_generated}" )
include_directories(${include_dir})
parsec_addtest(C strange "${strange_generated}")

parsec_compile_ptg(startup_generated "${CMAKE_CURRENT_SOURCE_DIR}/startup.jdf")
parsec_addtest(C startup "${startup_generated}")

parsec_compile_ptg(complex_deps_generated "${CMAKE_CURRENT_SOURCE_DIR}/complex_deps.jdf")
parsec_addtest(C complex_deps "${complex_deps_generated}")

if( MPI_C_FOUND )
  parsec_compile_ptg(multichain_generated "${CMAKE_CURRENT_SOURCE_DIR}/multichain.jdf")
  parsec_addtest(C multichain "${multichain_generated}")
endif( MPI_C_FOUND )

parsec_addtest(C compose "compose.c")

# Define Testings
include(Testings.cmake)
