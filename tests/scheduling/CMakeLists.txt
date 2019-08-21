include(ParsecCompilePTG)

parsec_addtest(C schedmicro "main.c;ep_wrapper.c;schedmicro_data.c")
target_ptg_sources(schedmicro PRIVATE "ep.jdf")

