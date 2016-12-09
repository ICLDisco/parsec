include(RulesJDF)

jdf_rules(rtt_generated "${CMAKE_CURRENT_SOURCE_DIR}/rtt.jdf")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

parsec_addtest(C rtt "main.c;${rtt_generated};rtt_wrapper.c;rtt_data.c")

jdf_rules(bw_test_generated "${CMAKE_CURRENT_SOURCE_DIR}/bandwidth.jdf")
parsec_addtest(C bw_test "${bw_test_generated}")

