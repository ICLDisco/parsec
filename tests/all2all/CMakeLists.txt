include(RulesJDF)

jdf_rules(a2a_generated "${CMAKE_CURRENT_SOURCE_DIR}/a2a.jdf")
string( REGEX REPLACE "a2a.[hc]" "" include_dir "${a2a_generated}" )
include_directories(${include_dir})

dague_addtest(C a2a "main.c;${a2a_generated};a2a_data.c")

jdf_rules(bw_test_generated "${CMAKE_CURRENT_SOURCE_DIR}/bandwidth.jdf")
dague_addtest(C bw_test "${bw_test_generated}")

