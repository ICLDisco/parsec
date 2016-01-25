include(RulesJDF)

jdf_rules(project_generated "${CMAKE_CURRENT_SOURCE_DIR}/project.jdf")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

dague_addtest(C project "main.c;${project_generated};tree_dist.c")

add_test(haar_tree_shm "./project" "-x")
