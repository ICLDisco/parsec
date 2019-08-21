include(ParsecCompilePTG)

parsec_addtest(C BT_reduction "main.c;BT_reduction_wrapper.c;reduc_data.c")
target_ptg_sources(BT_reduction PRIVATE "BT_reduction.jdf")
