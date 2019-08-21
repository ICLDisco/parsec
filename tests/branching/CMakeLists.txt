include(ParsecCompilePTG)

parsec_addtest(C branching "main.c;branching_wrapper.c;branching_data.c")
target_ptg_sources(branching PRIVATE "branching.jdf")

