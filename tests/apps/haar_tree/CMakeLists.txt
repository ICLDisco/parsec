include(ParsecCompilePTG)

parsec_addtest_executable(C project SOURCES main.c ${project_generated} tree_dist.c)
target_include_directories(project PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
target_ptg_sources(project PRIVATE "project.jdf;walk.jdf")
target_link_libraries(project PRIVATE Threads::Threads m)

