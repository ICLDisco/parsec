set(SOURCES parsecf.F90 parsecf.c)

add_library(parsecf STATIC ${SOURCES})
set_target_properties(parsecf PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
target_link_libraries(parsecf parsec ${EXTRA_LIBS})
install(TARGETS parsecf ARCHIVE DESTINATION lib)

set(PROF_SOURCES parsec_profilef.F90 parsec_profilef.c)
add_library(parsec_profilef STATIC ${PROF_SOURCES})
set_target_properties(parsec_profilef PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
target_link_libraries(parsec_profilef parsec ${EXTRA_LIBS})
install(TARGETS parsec_profilef ARCHIVE DESTINATION lib)

