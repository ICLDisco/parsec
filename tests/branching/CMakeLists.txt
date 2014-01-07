include(RulesJDF)

jdf_rules(branching_generated "${CMAKE_CURRENT_SOURCE_DIR}/branching.jdf")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

dague_addtest(C branching "main.c;${branching_generated};branching_wrapper.c;branching_data.c")

