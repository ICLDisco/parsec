include(ParsecCompilePTG)

parsec_addtest(C project "main.c;tree_dist.c")
target_ptg_sources(project PRIVATE "project.jdf;walk.jdf")
target_include_directories(project PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(project PRIVATE Threads::Threads m)

parsec_addtest(C project_dyn "main.c;tree_dist.c")
target_ptg_sources(project_dyn PRIVATE "project_dyn.jdf")
target_compile_definitions(project_dyn PUBLIC parsec_project_new=parsec_project_dyn_new)
target_include_directories(project_dyn PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)