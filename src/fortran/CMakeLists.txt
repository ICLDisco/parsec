set(SOURCES daguef.F90 daguef.c)

add_library(daguef STATIC ${SOURCES})
set_target_properties(daguef PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
target_link_libraries(daguef dague ${EXTRA_LIBS})
install(TARGETS daguef ARCHIVE DESTINATION lib)

set(PROF_SOURCES dague_profilef.F90 dague_profilef.c)
add_library(dague_profilef STATIC ${PROF_SOURCES})
set_target_properties(dague_profilef PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
target_link_libraries(dague_profilef dague ${EXTRA_LIBS})
install(TARGETS dague_profilef ARCHIVE DESTINATION lib)

