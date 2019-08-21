include(ParsecCompilePTG)

parsec_addtest(C choice "main.c;choice_wrapper.c;choice_data.c")
target_ptg_sources(choice PRIVATE "choice.jdf")
