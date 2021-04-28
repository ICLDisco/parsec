include(ParsecCompilePTG)

parsec_addtest_executable(C branching SOURCES main.c branching_wrapper.c branching_data.c)
target_ptg_sources(branching PRIVATE "branching.jdf")

