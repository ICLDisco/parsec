if(NOT BUILD_TOOLS OR NOT PARSEC_PROF_TRACE)
  return()
endif()
# when crosscompiling the executable is imported from the
# export file.
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(parsec-dbp2xml dbp2xml.c dbpreader.c)
set_target_properties(parsec-dbp2xml PROPERTIES LINKER_LANGUAGE C)
target_link_libraries(parsec-dbp2xml parsec-base)
install(TARGETS parsec-dbp2xml RUNTIME DESTINATION ${PARSEC_INSTALL_BINDIR})

add_executable(parsec-dbpinfos dbpinfos.c dbpreader.c)
set_target_properties(parsec-dbpinfos PROPERTIES LINKER_LANGUAGE C)
target_link_libraries(parsec-dbpinfos parsec-base)
install(TARGETS parsec-dbpinfos RUNTIME DESTINATION ${PARSEC_INSTALL_BINDIR})

add_executable(parsec-dbp2mem dbp2mem.c dbpreader.c)
set_target_properties(parsec-dbp2mem PROPERTIES LINKER_LANGUAGE C)
target_link_libraries(parsec-dbp2mem parsec-base)
install(TARGETS parsec-dbp2mem RUNTIME DESTINATION ${PARSEC_INSTALL_BINDIR})

find_package(GTG)

if(GTG_FOUND)
  add_executable(parsec-dbp2paje dbp2paje.c dbpreader.c)
  set_target_properties(parsec-dbp2paje PROPERTIES LINKER_LANGUAGE C)
  target_include_directories(parsec-dbp2paje PRIVATE ${GTG_INCLUDE_DIR})
  target_link_libraries(parsec-dbp2paje parsec-base ${GTG_LIBRARIES})

  install(TARGETS parsec-dbp2paje RUNTIME DESTINATION ${PARSEC_INSTALL_BINDIR})
endif()

find_package(Graphviz QUIET)

if(Graphviz_FOUND)

  add_executable(parsec-graph testgraph.c graph.c)
  set_target_properties(parsec-graph PROPERTIES LINKER_LANGUAGE C)
  target_include_directories(parsec-graph PRIVATE ${Graphviz_INCLUDE_DIRS})
  target_link_libraries(parsec-graph parsec-base ${Graphviz_LIBRARIES})

  find_package(GD QUIET)

  if(GD_FOUND AND GD_SUPPORTS_PNG AND GD_SUPPORTS_GIF)

    add_executable(parsec-dbp-dot2png dbp-dot2png.c dbpreader.c graph.c animation.c)
    set_target_properties(parsec-dbp-dot2png PROPERTIES LINKER_LANGUAGE C)
    target_include_directories(parsec-dbp-dot2png PRIVATE ${Graphviz_INCLUDE_DIRS} ${GD_INCLUDE_DIR})
    target_link_libraries(parsec-dbp-dot2png parsec-base ${Graphviz_LIBRARIES} ${GD_LIBRARY})
    install(TARGETS parsec-dbp-dot2png RUNTIME DESTINATION ${PARSEC_INSTALL_BINDIR})
  endif()

endif(Graphviz_FOUND)

Add_Subdirectory(python)

