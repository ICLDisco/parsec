include(ParsecCompilePTG)

parsec_addtest(C ctlgat "main.c;ctlgat_wrapper.c;ctlgat_data.c")
target_ptg_sources(ctlgat PRIVATE "ctlgat.jdf")

