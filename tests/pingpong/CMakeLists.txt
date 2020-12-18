include(ParsecCompilePTG)

parsec_addtest_executable(C rtt SOURCES main.c rtt_wrapper.c rtt_data.c)
target_ptg_sources(rtt PRIVATE "rtt.jdf")

parsec_addtest_executable(C bw_test)
target_ptg_sources(bw_test PRIVATE "bandwidth.jdf")

