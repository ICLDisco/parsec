include(RulesJDF)

jdf_rules(branching_generated "${CMAKE_CURRENT_SOURCE_DIR}/branching.jdf")
string( REGEX REPLACE "branching.[hc]" "" include_dir "${branching_generated}" )
include_directories(${include_dir})

dague_addtest(C branching "main.c;${branching_generated};branching_wrapper.c;branching_data.c")

