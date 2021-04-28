include(ParsecCompilePTG)

parsec_addtest_executable(C BT_reduction SOURCES main.c BT_reduction_wrapper.c reduc_data.c)
target_ptg_sources(BT_reduction PRIVATE "BT_reduction.jdf")
