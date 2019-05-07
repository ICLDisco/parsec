

find_package(LibXml2 QUIET)
find_package(Threads QUIET)
#if(Threads_FOUND)
#  set(PARSEC_HAVE_PTHREAD true)
#  list(APPEND EXTRA_LIBS ${CMAKE_THREAD_LIBS_INIT})
#endif(Threads_FOUND)

include_directories(${LIBXML2_INCLUDE_DIR})

add_executable(reader reader.c)
target_link_libraries(reader ${LIBXML2_LIBRARIES} ${EXTRA_LIBS} -lrt)
