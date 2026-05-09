parsec_addtest_executable(C reduce SOURCES reduce.c)

parsec_addtest_executable(C kcyclic)
target_ptg_sources(kcyclic PRIVATE "kcyclic.jdf")
target_link_libraries(kcyclic PRIVATE m)

add_subdirectory(two_dim_band)

if(PARSEC_HAVE_MPI)
  add_subdirectory(redistribute)
endif(PARSEC_HAVE_MPI)
add_subdirectory(reshape)
