include(RulesJDF)

jdf_rules(rtt_generated "${CMAKE_CURRENT_SOURCE_DIR}/rtt.jdf")
string( REGEX REPLACE "rtt.[hc]" "" include_dir "${rtt_generated}" )
include_directories(${include_dir})

dague_addtest(C rtt "main.c;${rtt_generated};rtt_wrapper.c;rtt_data.c")
