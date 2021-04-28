include(ParsecCompilePTG)

parsec_addtest_executable(C rtt SOURCES main.c rtt_wrapper.c rtt_data.c)
target_ptg_sources(rtt PRIVATE "rtt.jdf")

parsec_addtest_executable(C bw_test)
set_source_files_properties("bandwidth.jdf" PROPERTIES PTGPP_COMPILE_OPTIONS "--Wremoteref")
target_ptg_sources(bw_test PRIVATE "bandwidth.jdf")

