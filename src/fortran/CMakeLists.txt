set(SOURCES daguef.F90 daguef.c)

add_library(daguef STATIC ${SOURCES})
set_target_properties(daguef PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
target_link_libraries(daguef dague ${EXTRA_LIBS})
install(TARGETS daguef ARCHIVE DESTINATION lib)

