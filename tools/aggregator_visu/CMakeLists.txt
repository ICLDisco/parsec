find_package(LibXml2 QUIET)
find_package(Threads QUIET)
#if(Threads_FOUND)
#  set(PARSEC_HAVE_PTHREAD true)
#  list(APPEND EXTRA_LIBS ${CMAKE_THREAD_LIBS_INIT})
#endif(Threads_FOUND)

# Drop visualization support if we can't find libxml2.
if(NOT LIBXML2_FOUND)
  return()
endif(NOT LIBXML2_FOUND)

include_directories(${LIBXML2_INCLUDE_DIR})

add_executable(reader reader.c)
target_link_libraries(reader ${LIBXML2_LIBRARIES} ${EXTRA_LIBS})

if(NOT PARSEC_HAVE_SHM_OPEN)
  check_function_exists(shm_open PARSEC_HAVE_SHM_OPEN)
  if( NOT PARSEC_HAVE_SHM_OPEN )
    check_library_exists(rt shm_open "" PARSEC_HAVE_SHM_OPENRT)
    if(PARSEC_HAVE_SHM_OPENRT)
      target_link_libraries(reader rt)
      set(PARSEC_HAVE_SHM_OPEN true)
    endif(PARSEC_HAVE_SHM_OPENRT)
  endif( NOT PARSEC_HAVE_SHM_OPEN )
endif(NOT PARSEC_HAVE_SHM_OPEN)

