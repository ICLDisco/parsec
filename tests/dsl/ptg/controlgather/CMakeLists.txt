include(ParsecCompilePTG)

parsec_addtest_executable(C ctlgat SOURCES main.c ctlgat_wrapper.c ctlgat_data.c)
target_ptg_sources(ctlgat PRIVATE "ctlgat.jdf")

