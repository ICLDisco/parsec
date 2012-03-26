# when crosscompiling the executable is imported from the
# export file.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_executable(dbp2xml dbp2xml.c)
set_target_properties(dbp2xml PROPERTIES LINKER_LANGUAGE C)
set_target_properties(dbp2xml PROPERTIES LINK_FLAGS "${LOCAL_C_LINK_FLAGS}")

if(GTG_FOUND)
  add_executable(dbp2paje dbp2paje.c)
  set_target_properties(dbp2xml PROPERTIES LINKER_LANGUAGE C)
  include_directories(${GTG_INCLUDE_DIR})
  set_target_properties(dbp2paje PROPERTIES LINK_FLAGS "${LOCAL_C_LINK_FLAGS} ${GTG_LIBRARIES}")
  target_link_libraries(dbp2paje -lgtg)
endif()

install(TARGETS dbp2xml RUNTIME DESTINATION bin)
