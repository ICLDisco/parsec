find_package(LibXml2 QUIET)
find_package(Threads REQUIRED)

# Drop visualization support if we can't find libxml2.
if(NOT LIBXML2_FOUND)
  return()
endif(NOT LIBXML2_FOUND)

add_executable(parsec-reader reader.c)
target_include_directories(parsec-reader PRIVATE ${LIBXML2_INCLUDE_DIRS})
target_link_libraries(parsec-reader PRIVATE
  ${LIBXML2_LIBRARIES}
  Threads::Threads
  $<$<BOOL:${PARSEC_SHM_OPEN_IN_LIBRT}>:rt>
  ${EXTRA_LIBS}
)
