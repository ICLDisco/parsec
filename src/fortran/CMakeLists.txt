set(SOURCES daguef.F90 daguef.c)

add_library(daguef STATIC ${SOURCES})
set_target_properties(daguef PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
if (MPI_FOUND)
  target_link_libraries(daguef dague-mpi ${EXTRA_LIBS})
else (MPI_FOUND)
  target_link_libraries(daguef dague ${EXTRA_LIBS})
endif (MPI_FOUND)
install(TARGETS daguef ARCHIVE DESTINATION lib)
install(FILES daguef.h DESTINATION include)

