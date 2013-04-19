
macro(dague_addtest lang target input)
  if(CMAKE_${lang}_COMPILER_WORKS)
    if( MPI_FOUND )
      set(${target}_${lang}FLAGS  "${MPI_COMPILE_FLAGS} ${${target}_${lang}FLAGS}")
      set(${target}_LDFLAGS "${MPI_LINK_FLAGS} ${${target}_LDFLAGS}")
      set(${target}_LIBS    dague-mpi dague_distribution_matrix-mpi dague-base ${EXTRA_LIBS} ${${target}_LIBS} ${MPI_LIBRARIES})
    else ( MPI_FOUND )
      list(APPEND EXTRA_LIBS m)
      set(${target}_LIBS    dague dague_distribution_matrix dague-base ${EXTRA_LIBS} ${${target}_LIBS})
    endif()


    add_executable(${target} ${input})
    set_target_properties(${target} PROPERTIES
                          LINKER_LANGUAGE ${lang}
                          COMPILE_FLAGS "${${target}_${lang}FLAGS} ${LOCAL_${lang}_LINK_FLAGS}"
                          LINK_FLAGS "${${target}_LDFLAGS}")
    target_link_libraries(${target} ${${target}_LIBS})
    install(TARGETS ${target} RUNTIME DESTINATION bin)
  endif(CMAKE_${lang}_COMPILER_WORKS)
endmacro(dague_addtest)



#set(CMAKE_C_COMPILER "vtcc")
#set(CMAKE_Fortran_COMPILER "vtf77")
Add_Subdirectory(pingpong)
Add_Subdirectory(branching)
Add_Subdirectory(controlgather)
if (DAGUE_DEBUG_BUILD_UNIT_TESTS )
  Add_Subdirectory(unit)
  dague_addtest(C operator operator.c)
  dague_addtest(C reduce reduce.c)
endif(DAGUE_DEBUG_BUILD_UNIT_TESTS )

jdf_rules(touch_generated "${CMAKE_CURRENT_SOURCE_DIR}/touch.jdf")
string( REGEX REPLACE "touch.[hc]" "" include_dir "${touch_generated}" )
include_directories(${include_dir})

dague_addtest(C touch_ex "touch_ex.c;${touch_generated}")
