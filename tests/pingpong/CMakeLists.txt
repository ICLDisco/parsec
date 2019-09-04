include(ParsecCompilePTG)

parsec_addtest(C rtt "main.c;rtt_wrapper.c;rtt_data.c")
target_ptg_sources(rtt PRIVATE "rtt.jdf")

parsec_addtest(C bw_test)
target_ptg_sources(bw_test PRIVATE "bandwidth.jdf")

