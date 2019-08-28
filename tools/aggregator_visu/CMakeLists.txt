find_package(LibXml2 QUIET)
find_package(Threads REQUIRED)

# Drop visualization support if we can't find libxml2.
if(NOT LIBXML2_FOUND)
  return()
endif(NOT LIBXML2_FOUND)

add_executable(reader reader.c)
target_include_directories(reader PRIVATE ${LIBXML2_INCLUDE_DIRS})
target_link_libraries(reader PRIVATE ${LIBXML2_LIBRARIES} Threads::Threads ${EXTRA_LIBS})

if(NOT PARSEC_HAVE_SHM_OPEN)
  check_function_exists(shm_open PARSEC_HAVE_SHM_OPEN)
  if( NOT PARSEC_HAVE_SHM_OPEN )
    check_library_exists(rt shm_open "" PARSEC_HAVE_SHM_OPENRT)
    if(PARSEC_HAVE_SHM_OPENRT)
      target_link_libraries(reader PRIVATE rt)
      set(PARSEC_HAVE_SHM_OPEN true)
    endif(PARSEC_HAVE_SHM_OPENRT)
  endif( NOT PARSEC_HAVE_SHM_OPEN )
endif(NOT PARSEC_HAVE_SHM_OPEN)
