parsec_addtest_executable(C reduce SOURCES reduce.c)

parsec_addtest_executable(C kcyclic)
target_ptg_sources(kcyclic PRIVATE "kcyclic.jdf")
target_link_libraries(kcyclic PRIVATE m)

add_subdirectory(two_dim_band)

add_subdirectory(redistribute)
add_subdirectory(reshape)

