# when crosscompiling the executable is imported from the
# export file.
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(dbp2xml dbp2xml.c dbpreader.c)
set_target_properties(dbp2xml PROPERTIES LINKER_LANGUAGE C)
set_target_properties(dbp2xml PROPERTIES LINK_FLAGS "${LOCAL_C_LINK_FLAGS}")
target_link_libraries(dbp2xml dague-base ${GTG_LIBRARIES})

add_executable(dbpinfos dbpinfos.c dbpreader.c)
set_target_properties(dbpinfos PROPERTIES LINKER_LANGUAGE C)
set_target_properties(dbpinfos PROPERTIES LINK_FLAGS "${LOCAL_C_LINK_FLAGS}")
target_link_libraries(dbpinfos dague-base ${GTG_LIBRARIES})

find_package(GTG)

if(GTG_FOUND)
  add_executable(dbp2paje dbp2paje.c dbpreader.c)
  set_target_properties(dbp2paje PROPERTIES LINKER_LANGUAGE C)
  include_directories(${GTG_INCLUDE_DIR})
  target_link_libraries(dbp2paje dague-base ${GTG_LIBRARIES})
endif()

find_package(Graphviz QUIET)

if(GRAPHVIZ_FOUND)
  include_directories(${GRAPHVIZ_INCLUDE_DIR})

  add_executable(testgraph testgraph.c graph.c)
  set_target_properties(testgraph PROPERTIES LINKER_LANGUAGE C)
  set_target_properties(testgraph PROPERTIES LINK_FLAGS "${LOCAL_C_LINK_FLAGS} ${GRAPHVIZ_LIBRARY}")
  target_link_libraries(testgraph dague-base)

  find_package(GD QUIET)

  if(GD_FOUND AND GD_SUPPORTS_PNG AND GD_SUPPORTS_GIF)
    include_directories(${GD_INCLUDE_DIR})

    add_executable(dbp-dot2png dbp-dot2png.c dbpreader.c graph.c animation.c)
    set_target_properties(dbp-dot2png PROPERTIES LINKER_LANGUAGE C)
    set_target_properties(dbp-dot2png PROPERTIES LINK_FLAGS "${LOCAL_C_LINK_FLAGS} ${GRAPHVIZ_LIBRARY} ${GD_LIBRARY}")
    target_link_libraries(dbp-dot2png dague-base)
  endif()

endif(GRAPHVIZ_FOUND)

# Python support for profiling (requires Cython)
find_package(Cython)
if(CYTHON_EXECUTABLE)
  Add_Subdirectory(python)
endif(CYTHON_EXECUTABLE)

install(TARGETS dbp2xml RUNTIME DESTINATION bin)
