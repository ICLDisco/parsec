if(NOT BUILD_TOOLS OR NOT PARSEC_PROF_TRACE OR PARSEC_HAVE_OTF2)
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

if(CMAKE_CXX_COMPILER)
  include(FetchContent)
  FetchContent_Declare(perfetto_source
                       GIT_REPOSITORY https://android.googlesource.com/platform/external/perfetto
                       GIT_TAG v41.0)
  FetchContent_MakeAvailable(perfetto_source)
  FetchContent_GetProperties(perfetto_source SOURCE_DIR PERFETTO_SOURCE_DIR)

  if( EXISTS ${PERFETTO_SOURCE_DIR}/sdk/perfetto.h AND EXISTS ${PERFETTO_SOURCE_DIR}/sdk/perfetto.cc )
    message(STATUS "Found Perfetto library sources in ${PERFETTO_SOURCE_DIR}")

    add_executable(dbp2pftrace dbp2pftrace.cc "${PERFETTO_SOURCE_DIR}/sdk/perfetto.cc" "${PROJECT_SOURCE_DIR}/tools/profiling/dbpreader.c")
    target_include_directories(dbp2pftrace PRIVATE ${PERFETTO_SOURCE_DIR}/sdk)
    target_link_libraries(dbp2pftrace parsec-base)

    if (WIN32)
      # The perfetto library contains many symbols, so it needs the big object
      # format.
      target_compile_options(dbp2pftrace PRIVATE "/bigobj")
      # Disable legacy features in windows.h.
      target_compile_definitions(dbp2pftrace PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
      # On Windows we should link to WinSock2.
      target_link_libraries(dbp2pftrace ws2_32)
    endif (WIN32)

    # Enable standards-compliant mode when using the Visual Studio compiler.
    if (MSVC)
      target_compile_options(dbp2pftrace PRIVATE "/permissive-")
    endif (MSVC)
  else()
    message(WARNING "Unable to download the Perfetto library source from the Android website in '${PERFETTO_SOURCE_DIR}'; Perfetto is disabled")
  endif()
endif(CMAKE_CXX_COMPILER)

Add_Subdirectory(python)
