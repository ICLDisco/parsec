include(ParsecCompilePTG)

parsec_addtest_executable(C a2a SOURCES main.c a2a_data.c)
target_ptg_sources(a2a PRIVATE "a2a.jdf")
target_link_libraries(a2a PRIVATE m)

