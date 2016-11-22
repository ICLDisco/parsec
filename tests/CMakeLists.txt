
macro(dague_addtest lang target input)
  list(APPEND EXTRA_LIBS m)

  if( MPI_C_FOUND )
    set(${target}_${lang}FLAGS  "${MPI_C_COMPILE_FLAGS} ${${target}_${lang}FLAGS}")
    set(${target}_LDFLAGS "${MPI_C_LINK_FLAGS} ${${target}_LDFLAGS}")
  endif( MPI_C_FOUND )
  set(${target}_LIBS    dague dague_distribution_matrix ${EXTRA_LIBS} ${${target}_LIBS})

  add_executable(${target} ${input})
  set_target_properties(${target} PROPERTIES
    LINKER_LANGUAGE ${lang}
    COMPILE_FLAGS "${${target}_${lang}FLAGS} ${LOCAL_${lang}_LINK_FLAGS}"
    LINK_FLAGS "${${target}_LDFLAGS}")
  target_link_libraries(${target} ${${target}_LIBS})
  #  install(TARGETS ${target} RUNTIME DESTINATION bin)

endmacro(dague_addtest)

add_Subdirectory(haar-tree-project)
add_Subdirectory(pingpong)
add_Subdirectory(branching)
add_Subdirectory(choice)
add_Subdirectory(controlgather)
add_Subdirectory(scheduling)
add_Subdirectory(all2all)
add_Subdirectory(generalized_reduction)
add_Subdirectory(merge_sort)
add_Subdirectory(profiling)
add_Subdirectory(interfaces/superscalar)
add_Subdirectory(jdf/new_null)
add_Subdirectory(standalone-profiling)

Add_Subdirectory(unit)
dague_addtest(C operator operator.c)
dague_addtest(C reduce reduce.c)
jdf_rules(dtt_generated "${CMAKE_CURRENT_SOURCE_DIR}/dtt_bug_replicator.jdf")
dague_addtest(C dtt_bug_replicator "dtt_bug_replicator_ex.c;${dtt_generated}")

jdf_rules(wc_generated "${CMAKE_CURRENT_SOURCE_DIR}/write_check.jdf")
string( REGEX REPLACE "write_check.[hc]" "" include_dir "${wc_generated}" )
include_directories(${include_dir})
dague_addtest(C write_check "vector.c;${wc_generated}")

jdf_rules(touch_generated "${CMAKE_CURRENT_SOURCE_DIR}/touch.jdf")
string( REGEX REPLACE "touch.[hc]" "" include_dir "${touch_generated}" )
include_directories(${include_dir})

dague_addtest(C touch_ex "touch_ex.c;${touch_generated}")
if(CMAKE_Fortran_COMPILER_WORKS)
  if(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
    dague_addtest(Fortran touch_exf "touch_exf.F90;${touch_generated}")
    target_link_libraries(touch_exf daguef)
    SET_TARGET_PROPERTIES(touch_exf PROPERTIES COMPILE_FLAGS "-I${PROJECT_BINARY_DIR}/dague/fortran")
    SET_TARGET_PROPERTIES(touch_exf PROPERTIES LINKER_LANGUAGE Fortran)
  endif(CMAKE_Fortran_COMPILER_SUPPORTS_F90)
endif(CMAKE_Fortran_COMPILER_WORKS)

jdf_rules(strange_generated "${CMAKE_CURRENT_SOURCE_DIR}/strange.jdf")
string( REGEX REPLACE "strange.[hc]" "" include_dir "${strange_generated}" )
include_directories(${include_dir})
dague_addtest(C strange "${strange_generated}")

#Most of the test need -lm. Add it and trim the list down.
list(APPEND EXTRA_LIBS m)
list(REMOVE_DUPLICATES EXTRA_LIBS)

jdf_rules(startup_generated "${CMAKE_CURRENT_SOURCE_DIR}/startup.jdf")
dague_addtest(C startup "${startup_generated}")

jdf_rules(complex_deps_generated "${CMAKE_CURRENT_SOURCE_DIR}/complex_deps.jdf")
dague_addtest(C complex_deps "${complex_deps_generated}")

if( MPI_C_FOUND )
  jdf_rules(multichain_generated "${CMAKE_CURRENT_SOURCE_DIR}/multichain.jdf")
  dague_addtest(C multichain "${multichain_generated}")
endif( MPI_C_FOUND )

# Define Testings
include(Testings.cmake)
